"""RankDisplacement (RD): vocabulary-space displacement between instruct and base.

For each token position, compares top-k counterfactual candidate sets
from instruct vs base models, decomposing displacement into matched,
promoted, demoted pools. Also produces per-position displacement profiles.

Translated from TASM's `engine/sfd.py::compute_rank_displacement`.

Capture expectation:
  Needs base logits (for base-side counterfactuals). No forward-pass
  activations are required; this is purely a logit-space comparison.
  The instruct counterfactuals come from LTP's output via the dependency
  channel; the base counterfactuals come from base_extract.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from tagm.measurement.base import MeasurementModule
from tagm.measurement.parameters import ModuleParameter
from tagm.measurement.registry import register_measurement
from tagm.measurement.requirements import CaptureExpectation
from tagm.measurement.result import FieldSpec, MeasurementResult, padded_per_token

logger = logging.getLogger("tagm")


@register_measurement
class RankDisplacement(MeasurementModule):
    name = "rank_displacement"
    display_name = "Rank Displacement"
    description = (
        "Vocabulary-space displacement between instruct and base models' "
        "counterfactual candidate sets. Decomposes into matched / promoted / "
        "demoted pools per position."
    )
    version = "0.2.0"

    parameters = [
        ModuleParameter(
            name="k",
            display_name="Candidates per bank (k)",
            description="Number of top-k candidates compared per position.",
            kind="int", default=8, min_value=2, max_value=32,
        ),
        ModuleParameter(
            name="min_shared",
            display_name="Min shared candidates for tau",
            description="Minimum candidates in both banks to compute Kendall's tau.",
            kind="int", default=2, min_value=0, max_value=10, advanced=True,
        ),
    ]

    depends_on = ("lateral_tension_profile",)

    def capture_expectation(self, params):
        return CaptureExpectation(needs_base_logits=True)

    def base_extract(self, pipeline, prompt, base_logits, params):
        """Extract per-position base counterfactuals.

        If LTP is also active and emits its own base_extract, the
        orchestrator merges them by field name. Both emit
        'base_counterfactual_tokens' and the values are compatible.
        """
        k = int(params.get("k", 8))
        overfetch = 5
        tokenizer = pipeline.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(pipeline.device)
        token_ids = inputs["input_ids"][0]
        seq_len = token_ids.shape[0]

        base_cf = []
        for i in range(seq_len):
            chosen_id = token_ids[i].item()
            topk = torch.topk(base_logits[i], k + overfetch)
            probs = torch.softmax(topk.values, dim=-1)
            alts = []
            for j, tid in enumerate(topk.indices.tolist()):
                if tid != chosen_id and len(alts) < k:
                    alts.append((
                        tokenizer.decode(tid).strip(),
                        round(probs[j].item(), 8),
                    ))
            base_cf.append(alts)
        return {"base_counterfactual_tokens": base_cf}

    def compute(self, run_result, adapter, delta_store, params,
                probes=None, base_cache=None):
        seq_len = run_result.seq_len
        k = int(params.get("k", 8))
        min_shared = int(params.get("min_shared", 2))

        deps = params.get("_dependencies") or {}
        ltp_result = deps.get("lateral_tension_profile")

        instruct_cf = (ltp_result.objects.get("counterfactual_tokens")
                       if ltp_result is not None else [])
        base_cf = (base_cache.get("base_counterfactual_tokens")
                   if base_cache else []) or []

        result = MeasurementResult(
            measurement_name=self.name,
            measurement_version=self.version,
            parameters={"k": k, "min_shared": min_shared,
                        "_depends_on": list(self.depends_on)},
        )

        if not instruct_cf or not base_cf:
            result.per_token["total_disp"] = padded_per_token(seq_len)
            result.per_token["replacement_ratio"] = padded_per_token(seq_len)
            result.scalars["mean_disp_per_token"] = float("nan")
            self._annotate(result)
            return result

        n_pos = min(len(instruct_cf), len(base_cf))

        per_pos = []
        instruct_disp_profiles: list[list[float]] = []
        base_disp_profiles: list[list[float]] = []
        taus: list[float] = []
        overlaps: list[float] = []

        try:
            from scipy.stats import kendalltau
            _have_scipy = True
        except ImportError:
            _have_scipy = False
            logger.warning("[RD] scipy not available; tau = 0.0 at every position")

        for pos in range(n_pos):
            i_alts = instruct_cf[pos]
            b_alts = base_cf[pos]
            if not i_alts or not b_alts:
                per_pos.append(_empty_pos_record())
                instruct_disp_profiles.append([0.0] * k)
                base_disp_profiles.append([0.0] * k)
                overlaps.append(0.0)
                taus.append(0.0)
                continue

            inst_lookup = {t: (rank, p) for rank, (t, p) in enumerate(i_alts)}
            base_lookup = {t: (rank, p) for rank, (t, p) in enumerate(b_alts)}

            matched = set(inst_lookup) & set(base_lookup)
            promoted = set(inst_lookup) - set(base_lookup)
            demoted = set(base_lookup) - set(inst_lookup)

            matched_disp = sum(abs(inst_lookup[t][1] - base_lookup[t][1])
                               for t in matched)
            promoted_mass = sum(inst_lookup[t][1] for t in promoted)
            demoted_mass = sum(base_lookup[t][1] for t in demoted)
            total_disp = matched_disp + promoted_mass + demoted_mass

            n_m = len(matched)
            replacement_ratio = (
                (promoted_mass + demoted_mass) / total_disp if total_disp > 0 else 0.0
            )
            concentration = total_disp / n_m if n_m > 0 else 0.0

            per_pos.append({
                "n_matched": n_m,
                "n_promoted": len(promoted),
                "n_demoted": len(demoted),
                "matched_disp": round(matched_disp, 8),
                "promoted_mass": round(promoted_mass, 8),
                "demoted_mass": round(demoted_mass, 8),
                "total_disp": round(total_disp, 8),
                "replacement_ratio": round(replacement_ratio, 8),
                "concentration": round(concentration, 8),
            })

            i_profile = []
            for t, p in i_alts[:k]:
                if t in base_lookup:
                    i_profile.append(abs(p - base_lookup[t][1]))
                else:
                    i_profile.append(p)
            while len(i_profile) < k:
                i_profile.append(0.0)
            instruct_disp_profiles.append(i_profile)

            b_profile = []
            for t, p in b_alts[:k]:
                if t in inst_lookup:
                    b_profile.append(abs(p - inst_lookup[t][1]))
                else:
                    b_profile.append(p)
            while len(b_profile) < k:
                b_profile.append(0.0)
            base_disp_profiles.append(b_profile)

            i_tokens = [tok for tok, _ in i_alts]
            b_tokens = [tok for tok, _ in b_alts]
            all_tokens = set(i_tokens) | set(b_tokens)
            overlap = len(matched) / len(all_tokens) if all_tokens else 0.0
            overlaps.append(overlap)

            shared = [tok for tok in i_tokens if tok in b_tokens]
            if _have_scipy and len(shared) >= min_shared:
                i_ranks = [i_tokens.index(t) for t in shared]
                b_ranks = [b_tokens.index(t) for t in shared]
                tau, _ = kendalltau(i_ranks, b_ranks)
                taus.append(float(tau) if not np.isnan(tau) else 0.0)
            else:
                taus.append(0.0)

        total_disp_arr = padded_per_token(seq_len)
        replacement_ratio_arr = padded_per_token(seq_len)
        for i, rec in enumerate(per_pos):
            if i < seq_len:
                total_disp_arr[i] = rec["total_disp"]
                replacement_ratio_arr[i] = rec["replacement_ratio"]
        result.per_token["total_disp"] = total_disp_arr
        result.per_token["replacement_ratio"] = replacement_ratio_arr

        n = len(per_pos)
        if n > 0:
            result.scalars["mean_matched"] = float(
                sum(p["n_matched"] for p in per_pos) / n)
            result.scalars["mean_replacement"] = float(
                sum(p["replacement_ratio"] for p in per_pos) / n)
            result.scalars["mean_concentration"] = float(
                sum(p["concentration"] for p in per_pos) / n)
            result.scalars["mean_disp_per_token"] = float(
                sum(p["total_disp"] for p in per_pos) / n)
            result.scalars["total_displacement"] = float(
                sum(p["total_disp"] for p in per_pos))
            result.scalars["high_replacement_frac"] = float(
                sum(1 for p in per_pos if p["replacement_ratio"] > 0.5) / n)
            result.scalars["low_match_frac"] = float(
                sum(1 for p in per_pos if p["n_matched"] < 5) / n)
        result.scalars["mean_tau"] = float(np.mean(taus)) if taus else 0.0
        result.scalars["mean_overlap"] = float(np.mean(overlaps)) if overlaps else 0.0
        result.scalars["n_comparable"] = len(taus)
        result.scalars["n_positions"] = n_pos

        result.objects["per_position"] = per_pos
        result.objects["instruct_disp_profiles"] = instruct_disp_profiles
        result.objects["base_disp_profiles"] = base_disp_profiles
        result.objects["per_position_tau"] = [round(t, 8) for t in taus]
        result.objects["per_position_overlap"] = [round(o, 8) for o in overlaps]

        self._annotate(result)
        return result

    def _annotate(self, result: MeasurementResult) -> None:
        result.field_specs["total_disp"] = FieldSpec(
            name="total_disp", kind="per_token",
            description="Per-position total displacement "
                        "(matched + promoted + demoted).",
            length_invariant=False,
        )
        result.field_specs["replacement_ratio"] = FieldSpec(
            name="replacement_ratio", kind="per_token",
            description="Fraction of per-position displacement from set-composition "
                        "changes vs in-set probability shifts.",
            length_invariant=False,
        )
        for name, desc in (
            ("mean_disp_per_token", "Mean of per-position total displacement."),
            ("mean_replacement", "Mean of per-position replacement_ratio."),
            ("mean_tau", "Mean Kendall's tau."),
            ("mean_overlap", "Mean Jaccard overlap of candidate sets."),
        ):
            result.field_specs[name] = FieldSpec(
                name=name, kind="scalar", description=desc, length_invariant=True,
            )


def _empty_pos_record():
    return {
        "n_matched": 0, "n_promoted": 0, "n_demoted": 0,
        "matched_disp": 0.0, "promoted_mass": 0.0, "demoted_mass": 0.0,
        "total_disp": 0.0, "replacement_ratio": 0.0, "concentration": 0.0,
    }
