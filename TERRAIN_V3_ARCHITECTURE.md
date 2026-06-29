# Correction Field Topology v3 — Architecture

## The Shoebox

The terrain lives in a bounded 3D volume whose dimensions are fixed by the lattice geometry:

- **X (lateral)**: `K × COL_SPACING` per bank side (K=8, COL_SPACING=0.55 → 4.4 units per bank)
- **Z (depth)**: `n_tokens × COL_SPACING`
- **Y (vertical)**: `CEILING = CEILING_RATIO × K × COL_SPACING` (default: 4.4 units)

No measure can escape this box. The ceiling is defined by geometry, not by data.

## Logistic Normalization

All values are logistic-compressed into [0, 1] before mapping to height:

```
σ(v) = 1 / (1 + exp(-steepness × (v - midpoint)))
```

Parameterized **per-prompt** from the distribution's min/max:
- `midpoint = (min + max) / 2`
- `steepness = ln(19) / (max - midpoint)` → maps max to ~0.95, min to ~0.05

This means:
- Every prompt fills roughly the same vertical envelope
- Relative contrast between tokens is preserved
- Switching measures doesn't blow the terrain out of its bounding box
- Outliers saturate rather than dominating

Per-token scalars (brightness, bars) use simple min-max normalization to [0, 1].

## Channel Map

Visual channels bind independently to named measures:

| Channel         | Type              | Data requirement      | Normalization   |
|-----------------|-------------------|-----------------------|-----------------|
| **Height**      | bank-decomposed   | [n_tok][k] per bank   | Logistic → [0,1] → shoebox |
| **Spine**       | per-token scalar  | primary from height measure | Same logistic pool |
| **Surface color** | same as height  | normalized height value | Palette ramp    |
| **Brightness**  | per-token scalar  | any scalar measure    | Min-max → [0,1] |
| **Bar length**  | per-token scalar  | any scalar measure    | Min-max → [0,1] |
| **Bar color**   | categorical       | promoted/demoted/matched | Direct map      |
| **Font color**  | categorical       | promoted/demoted/matched | Direct map      |
| **Filter**      | per-token scalar  | any scalar measure    | Min-max → threshold |

## Available Measures

### Bank-decomposed (height-eligible)
- `rank_displacement` — per-rank probability displacement between base/instruct top-k
- `ltp_tension` — per-rank lateral tension magnitudes

### Per-token scalars (brightness/bar/filter)
- `kl_divergence` — KL(instruct ∥ base) per token
- `stress` — ASM alignment stress per token
- `signed_attr` — signed correction attribution (can be negative)
- `sfd_density` — spectral field density per token

## Payload Structure (v3)

The backend ships **all** available measures in each prompt, not just one.
The renderer picks which measure drives which channel from `channel_config`.

```
{
  geometry: { k, col_spacing, ceiling_ratio },
  height_measure: "rank_displacement",
  channel_config: {
    height: "rank_displacement",
    brightness: "kl_divergence",
    bar_length: "stress",
    filter: "none"
  },
  available_measures: {
    bank: ["rank_displacement", "ltp_tension"],
    scalar: ["kl_divergence", "stress", "signed_attr", "sfd_density"]
  },
  prompts: [{
    tokens: [...],
    banks: {
      rank_displacement: { primary: [...], instruct_bank: [[...]], base_bank: [[...]] },
      ltp_tension: { ... }
    },
    scalars: {
      kl_divergence: [...],
      stress: [...],
      signed_attr: [...],
      sfd_density: [...]
    },
    status: { instruct_status: [[...]], base_status: [[...]] },
    labels: { counterfactual_tokens: [...], base_counterfactual_tokens: [...] }
  }]
}
```

## What Changed from v2

| Aspect | v2 | v3 |
|--------|----|----|
| Height normalization | Linear: `(mag - mn) * HSCALE` | Logistic into shoebox |
| Vertical bounds | Unbounded (HSCALE=180) | Fixed ceiling from geometry |
| Payload | One measure's banks + KL sidecar | All available measures |
| Bar data source | Same as height (redundant) | Independent scalar channel |
| Brightness data source | Hardcoded to KL | Configurable scalar channel |
| Channel bindings | Hardcoded in renderer | Configured via `channel_config` |
| Module params | `measure` (one) | `height_measure` + per-channel selectors |

## Files

- `src/engine/modules/correction_field_topology.py` — v3.0.0 (payload builder)
- `static/correction_field_topology_viz.html` — the terrain viewer
  (standalone page; the shoebox, logistic normalization, and channel
  bindings all live here, not in main.js)
