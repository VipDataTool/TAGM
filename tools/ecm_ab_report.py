"""ECM diagnostics report — offline A/B analysis and tuning aid.

Reads one or more ECM diagnostics JSON files (the `ecm_diagnostics`
object from a chat response's final SSE event, saved to disk) and
produces a per-file summary plus a comparison figure.

The falsification test from the v2 review: run a benign prompt with
ECM on and check intervention_rate. If it exceeds ~10-15%, the
detector is tracking texture, not instability — raise the deadband
or the agreement requirement.

Usage:
    python tools/ecm_ab_report.py diag_benign.json diag_adversarial.json
    python tools/ecm_ab_report.py diag.json --out report.png

Each input file may be either the bare diagnostics object or a full
chat "done" event containing an "ecm_diagnostics" key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_diagnostics(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if "ecm_diagnostics" in obj:
        obj = obj["ecm_diagnostics"]
    required = {"per_token_entropy", "per_token_temperature",
                "per_token_cascade_signal"}
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)} — "
                         "is this an ECM diagnostics file?")
    return obj


def pctl(xs: list[float], q: float) -> float:
    """Nearest-rank percentile without numpy (works on any list)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, round(q / 100.0 * (len(s) - 1))))
    return s[idx]


def summarize(name: str, d: dict) -> dict:
    ent = d["per_token_entropy"]
    temp = d["per_token_temperature"]
    sig = d["per_token_cascade_signal"]
    n = len(ent)
    cfg = d.get("config", {})
    base = cfg.get("base_temperature")
    floor = cfg.get("floor")
    n_interv = d.get("n_interventions", sum(1 for s in sig if s > 0))
    at_floor = sum(1 for t in temp if floor is not None and t <= floor + 1e-9)
    return {
        "name": name,
        "n_tokens": n,
        "intervention_rate": (n_interv / n) if n else 0.0,
        "n_loop_releases": d.get("n_loop_releases", 0),
        "max_signal": d.get("max_cascade_signal", max(sig, default=0.0)),
        "entropy_mean": sum(ent) / n if n else float("nan"),
        "entropy_p90": pctl(ent, 90),
        "temp_mean": sum(temp) / n if n else float("nan"),
        "temp_p10": pctl(temp, 10),
        "frac_at_floor": (at_floor / n) if n else 0.0,
        "base_temperature": base,
        "signal_units": cfg.get("signal_units", "nats (v1)"),
    }


def print_summary(rows: list[dict]) -> None:
    cols = ["name", "n_tokens", "intervention_rate", "frac_at_floor",
            "n_loop_releases", "max_signal", "entropy_mean", "entropy_p90",
            "temp_mean", "temp_p10", "signal_units"]
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r[c]).ljust(widths[c]) for c in cols))
    print()
    for r in rows:
        rate = r["intervention_rate"]
        if r["signal_units"] == "sigma" and rate > 0.15:
            print(f"[!] {r['name']}: intervention rate {rate:.1%} — if this "
                  "was a benign prompt, raise the deadband or agreement.")
        if r["frac_at_floor"] > 0.05:
            print(f"[!] {r['name']}: {r['frac_at_floor']:.1%} of tokens at the "
                  "temperature floor — the gain may be too hot.")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def plot(diags: list[tuple[str, dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_files = len(diags)
    fig, axes = plt.subplots(3, n_files, figsize=(7 * n_files, 9),
                             squeeze=False, sharex="col")
    for j, (name, d) in enumerate(diags):
        ent = d["per_token_entropy"]
        temp = d["per_token_temperature"]
        sig = d["per_token_cascade_signal"]
        loops = d.get("per_token_loop", [])
        x = range(len(ent))
        cfg = d.get("config", {})

        ax = axes[0][j]
        ax.plot(x, ent, lw=0.8, color="#333", label="entropy (nats)")
        for k, traj in sorted(d.get("ema_trajectories", {}).items(),
                              key=lambda kv: int(kv[0])):
            ax.plot(range(len(traj)), traj, lw=0.9, alpha=0.7,
                    label=f"EWMA λ=0.5^{int(k)+1}")
        ax.set_title(name)
        ax.set_ylabel("entropy (nats)")
        ax.legend(fontsize=7, ncol=2)

        ax = axes[1][j]
        ax.plot(x, temp, lw=1.0, color="#b3541e")
        if cfg.get("base_temperature") is not None:
            ax.axhline(cfg["base_temperature"], ls="--", lw=0.7,
                       color="#888", label="base")
        if cfg.get("floor") is not None:
            ax.axhline(cfg["floor"], ls=":", lw=0.7,
                       color="#888", label="floor")
        for i, is_loop in enumerate(loops):
            if is_loop:
                ax.axvspan(i - 0.5, i + 0.5, color="#c0392b", alpha=0.15)
        ax.set_ylabel("effective temperature")
        ax.legend(fontsize=7)

        ax = axes[2][j]
        ax.plot(x, sig, lw=1.0, color="#1e6fb3")
        units = cfg.get("signal_units", "nats (v1)")
        ax.set_ylabel(f"cascade signal ({units})")
        ax.set_xlabel("token step")

    fig.suptitle("ECM diagnostics — entropy / temperature / cascade signal "
                 "(red bands: loop guard active)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path,
                    help="ECM diagnostics JSON file(s)")
    ap.add_argument("--out", type=Path, default=Path("ecm_report.png"),
                    help="Output figure path (default: ecm_report.png)")
    ap.add_argument("--no-plot", action="store_true",
                    help="Print the summary table only")
    args = ap.parse_args(argv)

    diags = [(p.stem, load_diagnostics(p)) for p in args.files]
    print_summary([summarize(name, d) for name, d in diags])
    if not args.no_plot:
        plot(diags, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
