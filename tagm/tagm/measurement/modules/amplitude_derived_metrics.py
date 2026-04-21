"""AmplitudeDerivedMetrics: per-token metrics derived from the amplitude heatmap.

Consumes the heatmap produced by AmplitudeTrajectory. Computes:

  - attn_frac:       fraction of correction mass from attention vs MLP per token.
  - coherence:       how peaked (vs uniform) each token's sublayer profile is.
  - sublayer_rank:   effective number of sublayers contributing meaningfully
                     per token. (TASM named this `per_token_spectral_rank`
                     but it is NOT an SVD rank — it's entropy-derived.
                     Renamed here per audit finding #3.)
  - token_similarity: token × token cosine similarity of sublayer profiles.

Capture expectation:
  Empty — this measurement reads from AmplitudeTrajectory's output, not
  from the ActivationStore directly. The orchestrator guarantees the
  dependency is satisfied.
"""
from __future__ import annotations

import numpy as np

from tagm.measurement.base import MeasurementModule
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation
from tagm.measurement.result import FieldSpec, MeasurementResult, padded_per_token


@register_measurement
class AmplitudeDerivedMetrics(MeasurementModule):
    name = "amplitude_derived_metrics"
    display_name = "Amplitude Derived Metrics"
    description = (
        "Per-token metrics derived from the amplitude heatmap: attention/MLP "
        "split, coherence, effective-sublayer count, and token-pair similarity."
    )
    version = "0.2.0"

    parameters = []

    depends_on = ("amplitude_trajectory",)

    def capture_expectation(self, params):
        # No captures of its own — operates on amplitude_trajectory's output
        return CaptureExpectation.empty()

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        seq_len = run_result.seq_len
        deps = params.get("_dependencies") or {}
        at_result = deps.get("amplitude_trajectory")

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={"_depends_on": list(self.depends_on)},
        )

        if at_result is None:
            result.per_token["attn_frac"] = padded_per_token(seq_len)
            result.per_token["coherence"] = padded_per_token(seq_len)
            result.per_token["sublayer_rank"] = padded_per_token(seq_len)
            result.objects["token_similarity"] = []
            return self._annotate(result)

        hm_list = at_result.objects.get("heatmap") or []
        hm = np.array(hm_list, dtype=float) if hm_list else np.zeros((0, seq_len))

        if hm.size == 0 or hm.shape[1] != seq_len:
            result.per_token["attn_frac"] = padded_per_token(seq_len)
            result.per_token["coherence"] = padded_per_token(seq_len)
            result.per_token["sublayer_rank"] = padded_per_token(seq_len)
            result.objects["token_similarity"] = []
            return self._annotate(result)

        n_sub, n_tok = hm.shape

        attn_rows = hm[0::2]
        mlp_rows = hm[1::2]
        attn_sum = attn_rows.sum(axis=0)
        mlp_sum = mlp_rows.sum(axis=0)
        total = attn_sum + mlp_sum
        attn_frac = np.where(total > 0, attn_sum / total, 0.5)
        result.per_token["attn_frac"] = attn_frac.astype(float)

        profiles = hm.T  # (seq_len, n_sublayers)
        coherence = np.zeros(n_tok, dtype=float)
        for i in range(n_tok):
            p = profiles[i]
            s = p.sum()
            if s > 0:
                normed = p / s
                ent = float(-np.sum(normed * np.log(normed + 1e-10)))
                max_ent = float(np.log(n_sub)) if n_sub > 1 else 1.0
                coherence[i] = 1.0 - (ent / max_ent) if max_ent > 0 else 0.0
        result.per_token["coherence"] = coherence

        sublayer_rank = np.zeros(n_tok, dtype=float)
        for i in range(n_tok):
            p = profiles[i]
            nz = p[p > 1e-8]
            if len(nz) > 1:
                q = nz / nz.sum()
                ent = float(-np.sum(q * np.log(q)))
                sublayer_rank[i] = float(np.exp(ent))
            else:
                sublayer_rank[i] = 1.0
        result.per_token["sublayer_rank"] = sublayer_rank

        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        normed_profiles = profiles / norms
        sim = normed_profiles @ normed_profiles.T
        result.objects["token_similarity"] = sim.astype(float).tolist()

        return self._annotate(result)

    def _annotate(self, result: MeasurementResult) -> MeasurementResult:
        result.field_specs["attn_frac"] = FieldSpec(
            name="attn_frac", kind="per_token",
            description="Fraction of correction mass from attention sublayers "
                        "vs MLP at each token.",
            units="fraction in [0,1]", length_invariant=False,
        )
        result.field_specs["coherence"] = FieldSpec(
            name="coherence", kind="per_token",
            description="1 - normalized Shannon entropy of sublayer profile. "
                        "1 = concentrated in one sublayer; 0 = spread uniformly.",
            units="normalized", length_invariant=False,
        )
        result.field_specs["sublayer_rank"] = FieldSpec(
            name="sublayer_rank", kind="per_token",
            description="Effective number of sublayers contributing meaningfully "
                        "to each token's correction. exp(Shannon entropy). "
                        "NOT an SVD rank — see measurement docstring.",
            units="sublayer count", length_invariant=False,
            semantic_note=(
                "Renamed from TASM's 'per_token_spectral_rank' because the "
                "original name implied SVD rank but the computation is an "
                "entropy-derived sublayer count."),
        )
        result.field_specs["token_similarity"] = FieldSpec(
            name="token_similarity", kind="object",
            description="(seq_len, seq_len) cosine similarity of per-token "
                        "sublayer profiles.",
            length_invariant=False,
        )
        return result
