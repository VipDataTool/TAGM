"""Pipeline: the instrument layer's primary orchestrator.

Owns a loaded model pair, a DeltaStore, and a tokenizer. Provides
parameterized forward passes that capture activations into an
ActivationStore per the caller's CaptureConfig. Supports sequential
base/instruct batch processing for memory-constrained environments.

Lifecycle:
    1. Construct with model IDs and (optionally) an explicit adapter.
    2. load(): instantiate instruct model, auto-detect or use given adapter,
       compute deltas from disk, compute spectral profile.
    3. run(prompt, capture_config): single-prompt forward pass with capture.
    4. Optional: load_base() before runs that need the base model live;
       or use run_pair_batch() for the sequential pattern.
    5. unload() when done.

The Pipeline is the only component that mutates model state. Measurements,
analysis modules, and the service layer read from RunResult/PairRunResult.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import torch

from tagm.core.adapter.registry import find_adapter
from tagm.core.capture.config import CaptureConfig
from tagm.core.capture.installer import install_hooks, remove_hooks
from tagm.core.capture.store import ActivationStore
from tagm.core.deltas.compute import compute_deltas_from_disk
from tagm.core.deltas.spectral import compute_spectral_profile
from tagm.core.types import ProgressCallback, noop_progress

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from tagm.core.adapter.base import ModelAdapter
    from tagm.core.deltas.store import DeltaStore


# ── Run result types ────────────────────────────────────────────────

@dataclass
class ModelStructure:
    """Snapshot of adapter-derived structural info about the loaded model.

    Attached to every RunResult so measurements don't need to re-derive
    head counts, hidden dim, etc. from the model itself (which would
    require carrying a model reference through the measurement layer —
    an unwanted coupling). Populated by Pipeline.run().
    """
    n_layers: int
    hidden_size: int
    n_attention_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int


@dataclass
class RunResult:
    """Result of a single prompt run through one model.

    Holds everything a measurement needs to compute from this prompt:
    tokens (as strings, with tokenizer-specific encodings normalized by the
    Pipeline), token IDs, logits, the ActivationStore filled by hooks
    during the forward pass, and a ModelStructure snapshot from the adapter.
    """
    prompt: str
    tokens: list[str]
    token_ids: torch.Tensor
    seq_len: int
    logits: torch.Tensor
    activations: ActivationStore
    structure: "ModelStructure" = None  # populated by Pipeline.run()
    used_base_model: bool = False

    # Optional: handle back to the originating pipeline, for measurements
    # like LTP that need the model's lm_head or o_proj weights directly.
    # Not serialized; set by the Pipeline and cleared before export.
    pipeline: Optional["Pipeline"] = None

    @classmethod
    def from_pair(cls, pair: "PairRunResult") -> "RunResult":
        """Adapt a PairRunResult into a RunResult referencing its instruct side.

        Used by the orchestrator to dispatch measurements uniformly whether
        the forward pass ran on instruct only or on both models.
        """
        return cls(
            prompt=pair.prompt,
            tokens=pair.tokens,
            token_ids=pair.token_ids,
            seq_len=pair.seq_len,
            logits=pair.instruct_logits,
            activations=pair.instruct_activations,
            structure=pair.structure,
            pipeline=pair.pipeline,
        )


@dataclass
class PairRunResult:
    """Result of a single prompt run through both models.

    Instruct-side activations are always captured (per the instruct
    CaptureConfig). Base-side activations are captured only if a
    base_capture config was provided; otherwise only base logits are
    retained. The `base_activations` attribute is None in the latter case.
    """
    prompt: str
    tokens: list[str]
    token_ids: torch.Tensor
    seq_len: int
    instruct_logits: torch.Tensor
    base_logits: torch.Tensor
    instruct_activations: ActivationStore
    base_activations: Optional[ActivationStore] = None
    structure: Optional["ModelStructure"] = None
    pipeline: Optional["Pipeline"] = None


@dataclass
class BatchBaseCache:
    """Per-prompt base-model outputs cached by the sequential base phase.

    Populated by `run_pair_batch`. The fields here are what the
    measurement layer's base-side extractor functions write; callers
    downstream access them by name.
    """
    prompt: str
    base_logits: Optional[torch.Tensor] = None        # [seq_len, vocab]
    base_log_softmax: Optional[torch.Tensor] = None   # float16 if caller compressed
    per_position_base_alts: list = field(default_factory=list)
    base_counterfactual_tokens: list = field(default_factory=list)
    base_topk: list = field(default_factory=list)


# ── The Pipeline class ──────────────────────────────────────────────

class Pipeline:
    """Owns a loaded model pair and provides parameterized inference.

    All model-family knowledge is delegated to the adapter. All capture
    knowledge is delegated to CaptureConfig/installer. All delta storage
    goes through DeltaStore. The Pipeline itself is a thin coordinator
    that holds references and runs forward passes.
    """

    def __init__(
        self,
        instruct_model_id: str,
        base_model_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        hf_token: Optional[str] = None,
        adapter: Optional["ModelAdapter"] = None,
    ):
        self.instruct_model_id = instruct_model_id
        self.base_model_id = base_model_id
        self.device = device
        self.dtype = dtype
        self.hf_token = hf_token
        self._explicit_adapter = adapter

        self.adapter: Optional["ModelAdapter"] = None
        self.instruct_model: Optional["PreTrainedModel"] = None
        self.base_model: Optional["PreTrainedModel"] = None
        self.tokenizer: Optional["PreTrainedTokenizer"] = None
        self.delta_store: Optional["DeltaStore"] = None

        self._loaded = False

    # ── Lifecycle ───────────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(
        self,
        layer_filter: Optional[list[int]] = None,
        compute_spectral: bool = True,
        svd_k: int = 64,
        progress: Optional[ProgressCallback] = None,
    ) -> None:
        """Load the instruct model, detect adapter, compute deltas, annotate spectral.

        Args:
          layer_filter:     If set, only compute deltas for these layers.
                            Default None means compute for all layers. The
                            DeltaStore records this filter so consumers that
                            request outside-filter layers get a clear error.
          compute_spectral: Whether to run the one-time spectral profile pass.
                            Default True. Disable for faster load when you
                            only need basic measurements.
          svd_k:            Truncation rank for delta SVDs.
          progress:         Optional progress callback.
        """
        from transformers import AutoModelForCausalLM

        log = progress or noop_progress

        log("loading", f"Loading instruct model: {self.instruct_model_id}")
        self.instruct_model = AutoModelForCausalLM.from_pretrained(
            self.instruct_model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
            token=self.hf_token,
            low_cpu_mem_usage=True,
        )

        # Adapter: explicit overrides auto-detection
        if self._explicit_adapter is not None:
            self.adapter = self._explicit_adapter
        else:
            self.adapter = find_adapter(self.instruct_model)
        log("loading", f"Adapter: {self.adapter.family_display_name} "
                       f"({self.adapter.family_id})")

        self.tokenizer = self.adapter.load_tokenizer(
            self.instruct_model_id, hf_token=self.hf_token)

        log("deltas", "Computing weight deltas from base model")
        self.delta_store = compute_deltas_from_disk(
            base_model_id=self.base_model_id,
            instruct_model=self.instruct_model,
            adapter=self.adapter,
            dtype=self.dtype,
            layer_filter=layer_filter,
            hf_token=self.hf_token,
            progress=progress,
        )

        if compute_spectral:
            log("spectral", "Computing delta spectral profile")
            compute_spectral_profile(
                self.delta_store,
                svd_k=svd_k,
                keep_singular_values=True,
                progress=progress,
            )

        self._loaded = True
        log("ready", "Pipeline ready")

    def load_base(self, progress: Optional[ProgressCallback] = None) -> None:
        """Load the base model for measurements that need live base inference.

        Memory-intensive: base and instruct models are both in RAM afterwards.
        Caller should `unload_base()` when finished. For batch workflows
        prefer `run_pair_batch()` which sequences base and instruct loads.
        """
        from transformers import AutoModelForCausalLM

        log = progress or noop_progress

        if self.base_model is not None:
            return
        log("loading", f"Loading base model: {self.base_model_id}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
            token=self.hf_token,
            low_cpu_mem_usage=True,
        )

    def unload_base(self) -> None:
        """Free the base model. Safe to call if base was never loaded."""
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload(self) -> None:
        """Release all model state. After this, load() must be called again."""
        if self.instruct_model is not None:
            del self.instruct_model
            self.instruct_model = None
        self.unload_base()
        self.delta_store = None
        self.tokenizer = None
        self.adapter = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Single-prompt run ───────────────────────────────────────────
    def run(
        self,
        prompt: str,
        capture_config: CaptureConfig,
        use_base: bool = False,
    ) -> RunResult:
        """Single prompt through one model, with the given capture config.

        Validates the capture config against the adapter, installs hooks,
        runs the forward pass, removes hooks, returns a RunResult.
        """
        if not self._loaded:
            raise RuntimeError("Pipeline.load() must be called before run()")

        validation = capture_config.validate(self.adapter)
        if not validation.ok:
            raise ValueError(
                f"CaptureConfig '{capture_config.name}' invalid for adapter "
                f"'{self.adapter.family_id}': {validation.errors}"
            )

        model = self.base_model if use_base else self.instruct_model
        if model is None:
            raise RuntimeError(
                "Requested base-model run but the base model is not loaded. "
                "Call Pipeline.load_base() first."
            )

        store = ActivationStore(capture_config, self.adapter)
        handles = install_hooks(model, self.adapter, capture_config, store)

        # Only ask for attention outputs if someone actually captures them
        needs_attn = any(
            "attention_weights" in p.capture for p in capture_config.points
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            # Normalize tokenizer-specific whitespace encodings
            tokens = [_clean_token(t) for t in tokens]

            with torch.no_grad():
                output = model(**inputs, output_attentions=needs_attn)

            n_heads, n_kv = self.adapter.attention_heads(model)
            structure = ModelStructure(
                n_layers=self.adapter.n_layers(model),
                hidden_size=self.adapter.hidden_size(model),
                n_attention_heads=n_heads,
                n_kv_heads=n_kv,
                head_dim=self.adapter.head_dim(model),
                vocab_size=self.adapter.vocab_size(model),
            )

            return RunResult(
                prompt=prompt,
                tokens=tokens,
                token_ids=inputs["input_ids"][0],
                seq_len=len(tokens),
                logits=output.logits.detach(),
                activations=store,
                structure=structure,
                used_base_model=use_base,
                pipeline=self,
            )
        finally:
            remove_hooks(handles)

    # ── Paired run (both models in RAM) ─────────────────────────────
    def run_pair(
        self,
        prompt: str,
        instruct_capture: CaptureConfig,
        base_capture: Optional[CaptureConfig] = None,
    ) -> PairRunResult:
        """Single prompt through both models, concurrently loaded.

        Requires load_base() to have been called previously. For memory-
        constrained environments use run_pair_batch() instead, which
        loads base → extracts → unloads → loads instruct.
        """
        if self.instruct_model is None or self.base_model is None:
            raise RuntimeError(
                "Both models must be loaded for run_pair(). "
                "Call load() and load_base() first, or use run_pair_batch() "
                "for sequential processing."
            )

        inst_result = self.run(prompt, instruct_capture, use_base=False)

        if base_capture is not None:
            base_result = self.run(prompt, base_capture, use_base=True)
            base_logits = base_result.logits
            base_activations = base_result.activations
        else:
            # Base logits only — no capture
            empty = CaptureConfig.empty(name="base-logits-only")
            base_result = self.run(prompt, empty, use_base=True)
            base_logits = base_result.logits
            base_activations = None

        return PairRunResult(
            prompt=prompt,
            tokens=inst_result.tokens,
            token_ids=inst_result.token_ids,
            seq_len=inst_result.seq_len,
            instruct_logits=inst_result.logits,
            base_logits=base_logits,
            instruct_activations=inst_result.activations,
            base_activations=base_activations,
            structure=inst_result.structure,
            pipeline=self,
        )

    # ── Sequential paired-batch (memory discipline) ─────────────────
    def run_pair_batch(
        self,
        prompts: list[str],
        instruct_capture: CaptureConfig,
        base_extractor: Callable[["Pipeline", str, torch.Tensor], dict],
        base_capture: Optional[CaptureConfig] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> list[tuple[RunResult, BatchBaseCache]]:
        """Run a batch of prompts through base then instruct, sequentially.

        Phase 1: load base model, run each prompt through it, call
                 `base_extractor(pipeline, prompt, base_logits)` to pull
                 out whatever base-side data the caller needs, unload base.
        Phase 2: run each prompt through instruct model with `instruct_capture`.
        Phase 3: zip the two results per prompt.

        The `base_extractor` is where the measurement layer expresses what
        it needs from the base pass — per-position alternatives for LTP,
        per-position base counterfactuals for RD, base top-k for response
        comparison, full log-softmax for KL. It returns a dict keyed by
        field names on BatchBaseCache.

        Args:
          base_capture: Optional CaptureConfig for the base pass. If provided,
                        hooks are installed for base-side activations too
                        (rarely needed; most base-side data is extracted
                        from logits). The captured ActivationStore is passed
                        to `base_extractor` via the Pipeline's internal state
                        but not returned in BatchBaseCache.

        Returns:
          List of (RunResult, BatchBaseCache) pairs, one per prompt, in
          input order.
        """
        log = progress or noop_progress
        n = len(prompts)

        # ── Phase 1: base model ─────────────────────────────────────
        log("base_phase", f"Loading base model for {n} prompt(s) "
                           f"(sequential mode)")
        self.load_base(progress=progress)

        base_caches: list[BatchBaseCache] = []
        try:
            for i, prompt in enumerate(prompts):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Optional base-side capture
                if base_capture is not None:
                    store = ActivationStore(base_capture, self.adapter)
                    handles = install_hooks(
                        self.base_model, self.adapter, base_capture, store)
                else:
                    store = None
                    handles = []

                try:
                    with torch.no_grad():
                        out = self.base_model(**inputs)
                    base_logits = out.logits[0].detach()

                    extracted = base_extractor(self, prompt, base_logits) or {}
                    cache = BatchBaseCache(prompt=prompt)
                    for field_name, value in extracted.items():
                        if hasattr(cache, field_name):
                            setattr(cache, field_name, value)
                    base_caches.append(cache)

                    del out, base_logits
                finally:
                    remove_hooks(handles)

                if (i + 1) % 5 == 0 or (i + 1) == n:
                    log("base_phase", f"Base phase: {i+1}/{n}")

                # Periodic memory cleanup
                if (i + 1) % 20 == 0:
                    gc.collect()
        finally:
            log("base_phase", "Unloading base model")
            self.unload_base()

        # ── Phase 2: instruct model (already loaded) ────────────────
        log("instruct_phase", f"Running instruct phase: {n} prompt(s)")
        instruct_results: list[RunResult] = []
        for i, prompt in enumerate(prompts):
            instruct_results.append(self.run(prompt, instruct_capture))
            if (i + 1) % 5 == 0 or (i + 1) == n:
                log("instruct_phase", f"Instruct phase: {i+1}/{n}")

        return list(zip(instruct_results, base_caches))

    # ── Introspection ───────────────────────────────────────────────
    def describe(self) -> dict:
        """Structured summary of loaded-model state. For the /api/status surface."""
        if not self._loaded:
            return {"loaded": False}
        n_heads, n_kv = self.adapter.attention_heads(self.instruct_model)
        return {
            "loaded": True,
            "adapter": {
                "family_id": self.adapter.family_id,
                "display_name": self.adapter.family_display_name,
            },
            "model_pair": {
                "instruct": self.instruct_model_id,
                "base": self.base_model_id,
                "device": self.device,
                "dtype": str(self.dtype).replace("torch.", ""),
            },
            "structure": {
                "n_layers": self.adapter.n_layers(self.instruct_model),
                "hidden_size": self.adapter.hidden_size(self.instruct_model),
                "n_attention_heads": n_heads,
                "n_kv_heads": n_kv,
                "head_dim": self.adapter.head_dim(self.instruct_model),
                "vocab_size": self.adapter.vocab_size(self.instruct_model),
            },
            "deltas": {
                "n_deltas": len(self.delta_store),
                "layer_filter": self.delta_store.layer_filter,
                "full_deltas_available": self.delta_store.full_deltas_available,
                "total_bytes": self.delta_store.total_bytes(),
                "spectral": self.delta_store.aggregate_spectral_summary(),
            },
            "base_model_loaded": self.base_model is not None,
        }


# ── Helpers ─────────────────────────────────────────────────────────

def _clean_token(token: str) -> str:
    """Normalize tokenizer-specific whitespace encodings for display.

    Different tokenizers use different sentinel characters for a leading
    space (BPE's 'Ġ' / 'Ċ', SentencePiece's '▁'). The display layer
    expects plain spaces and '\\n' placeholders.
    """
    return (token
            .replace("\u0120", " ")   # BPE leading space
            .replace("\u010a", "\\n") # BPE newline
            .replace("\u2581", " "))  # SentencePiece leading space
