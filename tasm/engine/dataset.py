"""
Dataset Manager: accumulates analysis results into a persistent session.
Writes CSV incrementally, saves plots to disk, generates comparative analytics.
Extended with LTP metrics in CSV export.
"""

import os
import io
import csv
import json
import time
import shutil
import zipfile
import base64
from pathlib import Path
from typing import List, Optional
import numpy as np


class DatasetSession:
    """A single experimental session that accumulates prompt results."""

    def __init__(self, base_dir: str = "datasets"):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(base_dir) / self.timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "plots" / "individual").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "plots" / "comparative").mkdir(parents=True, exist_ok=True)

        self.results: list = []  # list of result dicts
        self.csv_path = self.session_dir / "results.csv"
        self.model_name = ""
        self._csv_initialized = False

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

    def set_model(self, name: str):
        self.model_name = name
        # Write session metadata
        meta = {
            "model": name,
            "started": self.timestamp,
            "session_dir": str(self.session_dir),
        }
        with open(self.session_dir / "session.json", "w") as f:
            json.dump(meta, f, indent=2)

    def add_result(self, result_dict: dict, plots: dict = None):
        """Add an analyzed prompt to the session."""
        idx = len(self.results)
        result_dict["_index"] = idx
        self.results.append(result_dict)

        # Append to CSV
        self._write_csv_row(result_dict)

        # Save per-prompt plots
        if plots:
            for name, b64 in plots.items():
                if b64:
                    path = self.session_dir / "plots" / "individual" / f"{idx:04d}_{name}.png"
                    path.write_bytes(base64.b64decode(b64))

        return idx

    def save_comparative_plots(self, plots: dict):
        """Save comparative/aggregate plots."""
        for name, b64 in plots.items():
            if b64:
                path = self.session_dir / "plots" / "comparative" / f"{name}.png"
                path.write_bytes(base64.b64decode(b64))

    def save_aggregate_json(self, agg: dict):
        """Save aggregate statistics."""
        path = self.session_dir / "aggregate_statistics.json"
        with open(path, "w") as f:
            json.dump(agg, f, indent=2, default=str)

    def save_results_json(self):
        """Save the full per-prompt result dicts (including per-token arrays,
        LTP profiles, counterfactual tokens, heatmaps, trajectories, etc.).
        This preserves all data the analyzer produces, not just scalar summaries."""
        path = self.session_dir / "results.json"

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

        with open(path, "w") as f:
            json.dump(_sanitize(self.results), f, indent=1, default=str)

    def get_results_by_category(self):
        """Group results by category."""
        groups = {}
        for r in self.results:
            cat = r.get("category", "unknown")
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(r)
        return groups

    def export_zip(self, exclude_plots=False, exclude_pdf=False, exclude_json=False) -> bytes:
        """Package the session as a ZIP, optionally excluding heavy artifacts."""
        zip_path = self.session_dir / "tasm_session.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(self.session_dir):
                # Prune plots directory tree entirely if charts excluded
                if exclude_plots and "plots" in dirs:
                    dirs.remove("plots")

                for file in files:
                    if file == "tasm_session.zip":
                        continue
                    # Skip PDF report
                    if exclude_pdf and file == "report.pdf":
                        continue
                    # Skip results JSON (but keep aggregate_statistics.json, session.json)
                    if exclude_json and file == "results.json":
                        continue

                    filepath = Path(root) / file
                    arcname = filepath.relative_to(self.session_dir)
                    zf.write(filepath, arcname)
        return zip_path.read_bytes()

    def _write_csv_row(self, result_dict: dict):
        """Append a result to the CSV file."""
        fieldnames = [
            "index", "prompt", "category", "seq_len",
            "stress_score", "net_correction",
            "entropy", "gini", "top2_share", "middle_share", "interior_cv",
            "n_negative_tokens", "has_negative_tokens",
            "kl_divergence",
            "stress_score_ln", "entropy_ln", "top2_share_ln", "middle_share_ln",
            "instruct_top1", "instruct_top1_prob",
            "base_top1", "base_top1_prob",
            # LTP summary statistics
            "ltp_mean_M", "ltp_mean_C", "ltp_mean_V", "ltp_mean_L",
            "ltp_max_prc", "ltp_n_directional",
            "ltp_layer_strategy", "ltp_k", "ltp_svd_rank", "ltp_tuned_lens",
            # Classification
            "predicted_class", "prediction_confidence",
            # Full capture
            "full_capture",
            "mean_coherence", "mean_spectral_rank", "mean_attn_frac",
        ]

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
            "stress_score_ln": result_dict.get("stress_score_ln", ""),
            "entropy_ln": result_dict.get("entropy_ln", ""),
            "top2_share_ln": result_dict.get("top2_share_ln", ""),
            "middle_share_ln": result_dict.get("middle_share_ln", ""),
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
            row["ltp_mean_L"] = ltp.get("mean_L", "")
            row["ltp_max_prc"] = ltp.get("max_prc", "")
            row["ltp_n_directional"] = ltp.get("n_directional", "")
            row["ltp_layer_strategy"] = ltp.get("layer_strategy", "")
            row["ltp_k"] = ltp.get("k", "")
            row["ltp_svd_rank"] = ltp.get("svd_rank", "")
            row["ltp_tuned_lens"] = ltp.get("tuned_lens", "")

        # Classification
        cl = result_dict.get("classification")
        if cl:
            row["predicted_class"] = cl.get("predicted", "")
            row["prediction_confidence"] = cl.get("confidence", "")

        # Full capture summary metrics
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
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def clear(self):
        """Remove session data from disk and reset."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
        self.results.clear()
        self._csv_initialized = False
