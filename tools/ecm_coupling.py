"""ECM/geometry coupling — do cascade signals co-locate with the correction ridge?

The linking hypothesis between the ECM paper and the topology module:
per-token entropy rises (and therefore ECM cascade signals) should
co-locate with the per-token KL / stress ridge that the TAGM analyzer
measures on the same text. If they do, "accessory to refusal" is a
measured claim. If ECM instead fires at positions uncorrelated with the
ridge, it is a fluency stabilizer, not an alignment accessory.

Inputs:
  --ecm    JSON containing ECM diagnostics for a generated response
           (bare diagnostics or a chat "done" event or one record from
           tools/ecm_ablation_v2.py results.json via --index)
  --result JSON of the TAGM analyzer result for the SAME text
           (a single result dict with per_token_kl / per_token_stress,
           e.g. one entry from an exported session.json)

Token alignment: the analyzer tokenizes the full analyzed text; ECM
diagnostics cover only generated positions. The script aligns from the
tail (last N positions of the analyzer arrays vs N ECM positions) and
reports the assumed offset. If the analyzed text is not exactly the
generated response, expect misalignment — the script warns on length
mismatch rather than guessing silently.

Outputs Spearman rank correlations (no scipy needed) between:
  ECM per_token_entropy / per_token_cascade_signal
and each available analyzer series (per_token_kl, per_token_stress,
prc_per_token), plus top-decile overlap (do the hottest 10% of
positions coincide?).

Usage:
    python tools/ecm_coupling.py --ecm diag.json --result result.json
    python tools/ecm_coupling.py --ecm ablation_v2_out/results.json \
        --index 12 --result exported_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho via rank transform + Pearson (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def top_decile_overlap(x: list[float], y: list[float]) -> float:
    """Jaccard overlap of the top-10% positions of two series."""
    k = max(1, len(x) // 10)
    tx = set(sorted(range(len(x)), key=lambda i: -x[i])[:k])
    ty = set(sorted(range(len(y)), key=lambda i: -y[i])[:k])
    return len(tx & ty) / len(tx | ty)


def permutation_p(x: list[float], y: list[float], observed: float,
                  n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation p-value for the Spearman correlation."""
    import random
    rng = random.Random(seed)
    y2 = list(y)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(y2)
        if abs(spearman(x, y2)) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def load_ecm(path: Path, index: int | None) -> dict:
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        if index is None:
            raise SystemExit(
                f"{path} is a list ({len(obj)} records) — pass --index")
        obj = obj[index]
    for key in ("ecm", "ecm_diagnostics"):
        if key in obj:
            obj = obj[key]
    if "per_token_entropy" not in obj:
        raise SystemExit(f"{path}: no per_token_entropy — not ECM diagnostics?")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ecm", type=Path, required=True)
    ap.add_argument("--index", type=int, default=None,
                    help="record index when --ecm is an ablation results.json")
    ap.add_argument("--result", type=Path, required=True,
                    help="TAGM analyzer result JSON for the same text")
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()

    ecm = load_ecm(args.ecm, args.index)
    result = json.loads(args.result.read_text())
    if isinstance(result, dict) and "results" in result:
        raise SystemExit("--result looks like a full session export; "
                         "extract a single result dict from .results[]")

    ent = ecm["per_token_entropy"]
    sig = ecm["per_token_cascade_signal"]
    n = len(ent)

    analyzer_series = {}
    for key in ("per_token_kl", "per_token_stress", "prc_per_token",
                "per_token_coherence"):
        v = result.get(key) or (result.get("ltp") or {}).get(key)
        if isinstance(v, list) and len(v) >= 4:
            analyzer_series[key] = v
    if not analyzer_series:
        raise SystemExit("result JSON has no usable per-token series "
                         "(per_token_kl / per_token_stress / prc_per_token)")

    print(f"ECM positions: {n}")
    rows = []
    for key, series in analyzer_series.items():
        m = len(series)
        if m != n:
            print(f"[!] {key}: length {m} != ECM {n} — aligning from the "
                  f"tail over {min(m, n)} positions. Verify the analyzed "
                  "text is exactly the generated response.")
        k = min(m, n)
        a = series[-k:]
        for ecm_name, ecm_series in (("entropy", ent[-k:]),
                                     ("cascade_signal", sig[-k:])):
            if all(v == ecm_series[0] for v in ecm_series) or \
               all(v == a[0] for v in a):
                rho, p, ov = float("nan"), float("nan"), float("nan")
            else:
                rho = spearman(ecm_series, a)
                p = permutation_p(ecm_series, a, rho, args.n_perm)
                ov = top_decile_overlap(ecm_series, a)
            rows.append((key, ecm_name, k, rho, p, ov))

    print(f"\n{'analyzer series':<22} {'ecm series':<15} {'n':>5} "
          f"{'spearman':>9} {'perm_p':>8} {'top10% jaccard':>15}")
    print("-" * 80)
    for key, ecm_name, k, rho, p, ov in rows:
        print(f"{key:<22} {ecm_name:<15} {k:>5} {rho:>9.3f} "
              f"{p:>8.4f} {ov:>15.3f}")

    print("\nInterpretation: positive spearman with per_token_kl / "
          "per_token_stress plus above-chance top-decile overlap (~0.05 "
          "expected under independence) supports the co-location "
          "hypothesis. Near-zero everywhere = ECM is tracking fluency "
          "texture, not the correction ridge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
