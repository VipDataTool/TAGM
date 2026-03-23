"""
Dataset Manager: single-session storage for TASM analysis results.

Architecture:
  - One session directory: datasets/current/
  - New session wipes prior data automatically
  - results.json is the single source of truth (full per-token data)
  - summary.csv provides scalar-only view for quick inspection
  - No plot files on disk — all rendering is client-side
  - session.json stores metadata (model, timestamp)
"""

import csv
import json
import time
import shutil
import zipfile
from pathlib import Path
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger("tasm")


def _sanitize(obj):
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


class DatasetSession:
    """A single experimental session that accumulates prompt results.

    Only one session exists at a time. Starting a new session clears
    the previous one. All data lives in datasets/current/.
    """

    SESSION_DIR_NAME = "current"

    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / self.SESSION_DIR_NAME
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Wipe prior session
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
            logger.info(f"[SESSION] Cleared previous session at {self.session_dir}")

        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Clean up any old timestamped dirs from prior versions
        self._cleanup_old_sessions()

        self.results: list = []
        self.csv_path = self.session_dir / "summary.csv"
        self.json_path = self.session_dir / "results.json"
        self.model_name = ""
        self._csv_initialized = False

    def _cleanup_old_sessions(self):
        """Remove any timestamped session directories from prior versions."""
        if not self.base_dir.exists():
            return
        for child in self.base_dir.iterdir():
            if child.is_dir() and child.name != self.SESSION_DIR_NAME:
                try:
                    shutil.rmtree(child)
                    logger.info(f"[SESSION] Removed old session: {child.name}")
                except Exception as e:
                    logger.warning(f"[SESSION] Could not remove {child.name}: {e}")

    # ─── Properties ───

    @property
    def n_results(self):
        return len(self.results)

    @property
    def categories(self):
        cats = {}
        for r in self.results:
            c = r.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        return cats

    # ─── Session lifecycle ───

    def set_model(self, name: str):
        self.model_name = name
        meta = {
            "model": name,
            "started": self.timestamp,
            "session_dir": str(self.session_dir),
        }
        with open(self.session_dir / "session.json", "w") as f:
            json.dump(meta, f, indent=2)

    def add_result(self, result_dict: dict, plots: dict = None):
        """Add an analyzed prompt to the session.

        Plots dict is accepted for API compatibility but not written
        to disk — all plot rendering happens client-side.
        """
        idx = len(self.results)
        result_dict["_index"] = idx
        self.results.append(result_dict)
        self._write_csv_row(result_dict)
        return idx

    def save_comparative_plots(self, plots: dict):
        """No-op. Comparative plots are rendered client-side.

        Kept for API compatibility with app.py dashboard pipeline.
        """
        pass

    def save_aggregate_json(self, agg: dict):
        """Save aggregate statistics."""
        path = self.session_dir / "aggregate_statistics.json"
        with open(path, "w") as f:
            json.dump(agg, f, indent=2, default=str)

    def save_results_json(self, include_arrays: bool = True):
        """Save per-prompt results to JSON.

        Args:
            include_arrays: if True (default), preserves all per-token arrays
                (signed_attr, per_position_tau, LTP profiles, SFD per-token,
                etc.) for full offline analysis.  If False, strips arrays to
                produce a compact scalar-only export.
        """
        EXCLUDE_ALWAYS = {
            # Internal metadata (not analysis data)
            "per_layer_amplitude", "signal_layer_indices", "spectral_summary",
            "delta_scale", "full_capture_enabled", "per_layer_signed_attr",
            # Obsolete fields from old baseline normalization
            "entropy_ln", "stress_score_ln", "top2_share_ln", "middle_share_ln",
        }
        ARRAY_FIELDS = {
            "heatmap", "amplitude_trajectory", "amplitude_normalized",
            "per_token_stress", "signed_attr",
            "per_token_kl", "per_token_coherence", "per_token_spectral_rank",
            "attn_frac", "token_similarity", "base_counterfactual_tokens",
            "proof1_checks",
        }
        LTP_ARRAY_FIELDS = {
            "profiles", "base_profiles", "tension_magnitudes",
            "counterfactual_tokens", "semantic_trajectory_2d",
            "tension_trajectory_2d", "offset_magnitude", "offset_consistency",
            "offset_variance", "prc_per_token",
        }
        RD_ARRAY_FIELDS = {
            "per_position", "per_position_tau", "per_position_overlap",
            "instruct_disp_profiles", "base_disp_profiles",
        }
        SFD_ARRAY_FIELDS = {
            "per_token_energy", "per_token_entropy", "per_token_density",
        }

        exclude = EXCLUDE_ALWAYS | (ARRAY_FIELDS if not include_arrays else set())

        slim = []
        for r in self.results:
            row = {k: v for k, v in r.items() if k not in exclude}
            if not include_arrays:
                if "ltp" in row and isinstance(row["ltp"], dict):
                    row["ltp"] = {k: v for k, v in row["ltp"].items()
                                  if k not in LTP_ARRAY_FIELDS}
                if "rank_displacement" in row and isinstance(row["rank_displacement"], dict):
                    row["rank_displacement"] = {k: v for k, v in row["rank_displacement"].items()
                                                if k not in RD_ARRAY_FIELDS}
                if "sfd" in row and isinstance(row["sfd"], dict):
                    row["sfd"] = {k: v for k, v in row["sfd"].items()
                                  if k not in SFD_ARRAY_FIELDS}
            slim.append(row)
        with open(self.json_path, "w") as f:
            json.dump(_sanitize(slim), f, indent=1, default=str)

    def get_results_by_category(self):
        """Group results by category."""
        groups = {}
        for r in self.results:
            cat = r.get("category", "unknown")
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(r)
        return groups

    # ─── Export ───

    def export_zip(self, exclude_plots=False, exclude_pdf=False, exclude_json=False) -> bytes:
        """Package the session as a ZIP, written to disk for download endpoint."""
        if not exclude_json:
            self.save_results_json()

        zip_path = self.session_dir / "tasm_session.zip"
        plots_dir = self.session_dir / "plots"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filepath in self.session_dir.rglob("*"):
                if not filepath.is_file():
                    continue
                if filepath.name == "tasm_session.zip":
                    continue
                if exclude_json and filepath.name == "results.json":
                    continue
                if exclude_pdf and filepath.name == "report.pdf":
                    continue
                if exclude_plots and plots_dir in filepath.parents:
                    continue
                arcname = filepath.relative_to(self.session_dir)
                zf.write(filepath, arcname)
        return zip_path.read_bytes()

    # ─── CSV (scalar summary only) ───

    CSV_FIELDS = [
        "index", "prompt", "category", "seq_len",
        "stress_score", "net_correction",
        "entropy", "gini", "top2_share", "middle_share", "interior_cv",
        "n_negative_tokens", "has_negative_tokens",
        "kl_divergence",
        "instruct_top1", "instruct_top1_prob",
        "base_top1", "base_top1_prob",
        # LTP summary (ltp_mean_L excluded: constant 1.0, zero variance)
        "ltp_mean_M", "ltp_mean_C", "ltp_mean_V",
        "ltp_max_prc", "ltp_n_directional",
        "ltp_layer_strategy", "ltp_k", "ltp_svd_rank", "ltp_tuned_lens",
        # Rank displacement
        "rd_mean_matched", "rd_mean_replacement", "rd_mean_concentration",
        "rd_mean_tau", "rd_mean_overlap",
        # SFD summary
        "sfd_density_mean", "sfd_energy_mean", "sfd_entropy_mean",
        # Classification
        "predicted_class", "prediction_confidence",
        # Full capture
        "full_capture",
        "mean_coherence", "mean_spectral_rank", "mean_attn_frac",
    ]

    def _write_csv_row(self, result_dict: dict):
        """Append a scalar summary row to the CSV."""
        row = {
            "index": result_dict.get("_index", ""),
            "prompt": result_dict.get("prompt", ""),
            "category": result_dict.get("category", ""),
            "seq_len": result_dict.get("seq_len", ""),
            "stress_score": result_dict.get("stress_score", ""),
            "net_correction": result_dict.get("net_correction", ""),
            "entropy": result_dict.get("entropy", ""),
            "gini": result_dict.get("gini", ""),
            "top2_share": result_dict.get("top2_share", ""),
            "middle_share": result_dict.get("middle_share", ""),
            "interior_cv": result_dict.get("interior_cv", ""),
            "n_negative_tokens": result_dict.get("n_negative_tokens", ""),
            "has_negative_tokens": result_dict.get("has_negative_tokens", ""),
            "kl_divergence": result_dict.get("kl_divergence", ""),
        }

        # Top-1 predictions
        inst_topk = result_dict.get("instruct_topk", [])
        if inst_topk:
            row["instruct_top1"] = inst_topk[0][0]
            row["instruct_top1_prob"] = inst_topk[0][1]
        base_topk = result_dict.get("base_topk", [])
        if base_topk:
            row["base_top1"] = base_topk[0][0]
            row["base_top1_prob"] = base_topk[0][1]

        # LTP summary
        ltp = result_dict.get("ltp")
        if ltp:
            row["ltp_mean_M"] = ltp.get("mean_M", "")
            row["ltp_mean_C"] = ltp.get("mean_C", "")
            row["ltp_mean_V"] = ltp.get("mean_V", "")
            row["ltp_max_prc"] = ltp.get("max_prc", "")
            row["ltp_n_directional"] = ltp.get("n_directional", "")
            row["ltp_layer_strategy"] = ltp.get("layer_strategy", "")
            row["ltp_k"] = ltp.get("k", "")
            row["ltp_svd_rank"] = ltp.get("svd_rank", "")
            row["ltp_tuned_lens"] = ltp.get("tuned_lens", "")

        # Rank displacement summary
        rd = result_dict.get("rank_displacement")
        if rd:
            row["rd_mean_matched"] = rd.get("mean_matched", "")
            row["rd_mean_replacement"] = rd.get("mean_replacement", "")
            row["rd_mean_concentration"] = rd.get("mean_concentration", "")
            row["rd_mean_tau"] = rd.get("mean_tau", "")
            row["rd_mean_overlap"] = rd.get("mean_overlap", "")

        # SFD summary
        sfd = result_dict.get("sfd")
        if sfd:
            row["sfd_density_mean"] = sfd.get("density_mean", "")
            row["sfd_energy_mean"] = sfd.get("energy_mean", "")
            row["sfd_entropy_mean"] = sfd.get("entropy_mean", "")

        # Classification
        cl = result_dict.get("classification")
        if cl:
            row["predicted_class"] = cl.get("predicted", "")
            row["prediction_confidence"] = cl.get("confidence", "")

        # Full capture summary
        row["full_capture"] = result_dict.get("full_capture_enabled", False)
        coherence = result_dict.get("per_token_coherence", [])
        if coherence:
            row["mean_coherence"] = sum(coherence) / len(coherence)
        spectral = result_dict.get("per_token_spectral_rank", [])
        if spectral:
            row["mean_spectral_rank"] = sum(spectral) / len(spectral)
        attn = result_dict.get("attn_frac", [])
        if attn:
            row["mean_attn_frac"] = sum(attn) / len(attn)

        mode = "a" if self._csv_initialized else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS,
                                    extrasaction="ignore")
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    # ─── Cleanup ───

    def clear(self):
        """Remove session data from disk and reset."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
        self.results.clear()
        self._csv_initialized = False

    def get_cache_size(self) -> int:
        """Return total size in bytes of all files in the session directory."""
        if not self.session_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self.session_dir.rglob("*") if f.is_file())

    def clear_plots(self) -> int:
        """Delete all plot PNGs. Returns bytes freed."""
        plots_dir = self.session_dir / "plots"
        if not plots_dir.exists():
            return 0
        freed = sum(f.stat().st_size for f in plots_dir.rglob("*") if f.is_file())
        shutil.rmtree(plots_dir, ignore_errors=True)
        return freed
