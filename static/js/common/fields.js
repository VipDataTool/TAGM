/* TAGM — field metadata registry.  THE single source of truth for what a
 * stored field is called, what it means, and what you must not do with it.
 *
 * WHY this file exists: the Data tab previously carried a hand-written
 * "Field Glossary" block in index.html AND a separate column array in
 * main.js.  Two sources, no link between them, so they drifted — the
 * glossary described columns that had been renamed and omitted every
 * audit caveat.  Headers, hover tooltips and the glossary card are now all
 * derived from TAGM.FIELDS, which makes that class of drift impossible.
 *
 * Plain <script src>-able (no ES module syntax), same as esc.js, because
 * the standalone viz pages are opened directly by the browser.
 *
 * Entry shape:
 *   key            dotted path into the serialized record ("ltp.mean_M").
 *                  Dotted because nested blocks reuse names — ltp.n_layers_used
 *                  and sfd.n_layers_used are different quantities.
 *   label          short display name; used for table headers and glossary.
 *   group          one of GROUPS below.
 *   format         render/format code, see FORMATS.
 *   desc           what the number is, in one or two sentences.
 *   units          optional unit string.
 *   caveat         optional. NON-EMPTY MEANS: this field can be read wrongly.
 *                  Presence drives the ° marker on the column header and the
 *                  call-out in the glossary.
 *   extensive      true when the quantity grows with sequence length by
 *                  construction (a count, a sum, or a max over positions).
 *   lengthSensitive true when the value is nominally intensive but its
 *                  expectation still drifts with sequence length.
 *   notComparable  optional string naming the axis along which comparison is
 *                  invalid ("across models").
 *   uiOnly         true for keys the frontend adds; never produced by the
 *                  backend serializer.
 */
(function (global) {
  var TAGM = global.TAGM = global.TAGM || {};

  /* Group order is display order, both in the glossary and in the row
     detail panel.  "Unregistered" is not listed here — it is synthesized by
     the detail renderer for keys the record carries but this file does not
     describe, so a backend addition surfaces instead of vanishing. */
  var GROUPS = ['Identity', 'ASM', 'Predictions', 'LTP', 'SFD',
                'RankDisplacement', 'ECM', 'Provenance'];

  var GROUP_LABELS = {
    Identity: 'Identity',
    ASM: 'Alignment Stress Map (ASM)',
    Predictions: 'Model Predictions',
    LTP: 'Lateral Tension Profile (LTP)',
    SFD: 'Spectral Field Density (SFD)',
    RankDisplacement: 'Rank Displacement',
    ECM: 'Entropic Cascade Mitigation (ECM)',
    Provenance: 'Provenance & Diagnostics',
    Unregistered: 'Unregistered (present in record, not in TAGM.FIELDS)'
  };

  var GROUP_COLORS = {
    Identity: 'var(--text-2)',
    ASM: 'var(--blue)',
    Predictions: 'var(--text-1)',
    LTP: 'var(--cyan)',
    SFD: 'var(--orange)',
    RankDisplacement: 'var(--purple)',
    ECM: 'var(--blue)',
    Provenance: 'var(--text-3)',
    Unregistered: 'var(--red)'
  };

  /* The six attribution-derived scalars that are dataclass DEFAULTS rather
     than measurements whenever attribution_unavailable is non-null.  Taken
     verbatim from the comment on PromptResult.attribution_unavailable in
     src/engine/result.py and from statistics._attr_val, which treats exactly
     this set as missing.  The detail panel mutes them. */
  var ATTRIBUTION_DEPENDENT = [
    'entropy', 'top2_share', 'middle_share', 'interior_cv',
    'net_correction', 'n_negative_tokens'
  ];

  /* Shared caveat text, written once so the wording cannot fork between the
     header tooltip and the glossary. */
  var C = {
    lengthNull55:
      'LENGTH-SENSITIVE. Normalized by log(seq_len), which only corrects the ' +
      'uniform case — the expectation still drifts with sequence length. ' +
      'Under a pure-length null this produces a 55% false-positive rate ' +
      '(nominal 5%). Always read against length_correlations before calling a ' +
      'difference behavioural.',
    lengthNull28:
      'LENGTH-SENSITIVE, same class as entropy: nominally intensive, but its ' +
      'expectation drifts with sequence length. 28% false-positive rate under ' +
      'a pure-length null (nominal 5%).',
    nDirectional:
      'EXTENSIVE. A count of token positions, so it grows linearly with ' +
      'sequence length. NOT comparable across prompts of differing length — ' +
      'use directional_frac, the length-invariant rate, which is what the ' +
      'statistics registry uses.',
    directionalFrac:
      'The length-invariant rate form of n_directional. This is the quantity ' +
      'the statistics registry compares; the raw count is display-only.',
    netCorrection:
      'A SUM over token positions. Its mean is stable with length, but its SD ' +
      'grows as sqrt(n) — measured 4.4 at 20 tokens, 12.6 at 160. That is ' +
      'heteroscedasticity: it inflates the pooled SD Cohen’s d divides ' +
      'by, so comparisons between groups of differing typical length are ' +
      'underpowered and violate equal-variance.',
    maxOverPositions:
      'A MAXIMUM over token positions, so it grows with sequence length. ' +
      'Display only — deliberately absent from the statistics registry.',
    middleShare:
      'Exactly 1 − top2_share (attr_dist sums to 1 and the two partition ' +
      'it, verified to 10 decimal places). Excluded from the statistics ' +
      'registry so a single degree of freedom is not double-counted in the ' +
      'Benjamini-Hochberg FDR correction.',
    stressScore:
      'A MEAN over contributing roles. Comparable across layers within one ' +
      'model, NOT across models: it carries a 1/sqrt(d_in) factor.',
    meanTau:
      'null when no position had enough shared tokens for Kendall tau to be ' +
      'defined. Always read together with n_comparable and n_tau_undefined — ' +
      'the mean averages only the defined positions.',
    proof1:
      'An ALGEBRAIC IDENTITY, not a validation. With û = δ/‖δ‖, ' +
      'Σⱼ aⱼ(û·vⱼ) = û·(Σⱼ aⱼvⱼ) = ‖δ‖ holds for ANY a and v. ' +
      'It therefore passes regardless of whether the decomposition, the head ' +
      'reshape or the GQA grouping is correct. "exact" means only that the ' +
      'float sum did not lose precision — nothing more.',
    roleMean:
      'MEAN over roles (attn aggregates q/k/v = 3, mlp aggregates gate/up = 2). ' +
      'Summing instead made the two sublayer types differ by a constant ~3/2 ' +
      'factor unrelated to the model. Values are attn÷3 and mlp÷2 versus ' +
      'sessions recorded before that fix — do not compare across the change.',
    attributionUnavailable:
      'When non-null, entropy, top2_share, middle_share, interior_cv, ' +
      'net_correction and n_negative_tokens are DATACLASS DEFAULTS, not ' +
      'measurements. The statistics layer treats them as missing; so should you.',
    countOfPositions:
      'A COUNT of token positions, so it grows with sequence length by ' +
      'construction. Not comparable across prompts of differing length.',
    sumOverPositions:
      'A SUM over token positions, so it grows with sequence length by ' +
      'construction. Not comparable across prompts of differing length.'
  };

  /* ── The registry ───────────────────────────────────────────────── */
  var FIELDS = [

    /* ── Identity ─────────────────────────────────────────────────── */
    { key: '_index', label: '#', group: 'Identity', format: 'int', uiOnly: true,
      desc: 'Record index within the session, in chronological insert order. ' +
            'Assigned by the session store, not by the analysis.' },
    { key: 'prompt', label: 'Prompt', group: 'Identity', format: 'text',
      desc: 'The exact text that was tokenized and analyzed. For a harvested ' +
            'response record this is the generated text, not the user prompt.' },
    { key: 'category', label: 'Cat', group: 'Identity', format: 'cat',
      desc: 'Category label carried with the prompt (benign, mild, harmful, ' +
            'jailbreak, dual-use, adversarial, ...). Free-form: whatever the ' +
            'CSV or the sidebar supplied.' },
    { key: 'role', label: 'Role', group: 'Identity', format: 'text',
      desc: 'Set to "assistant" on records produced by response harvesting — ' +
            'i.e. the record is a model generation analyzed as its own input. ' +
            'Absent on ordinary prompt records.' },
    { key: 'family_index', label: 'Fam', group: 'Identity', format: 'int',
      desc: 'Session-stable id of the prefix ladder this record belongs to. ' +
            'Present only when "Deconstruct prompts" was enabled; every rung ' +
            'of one source prompt shares a family_index.' },
    { key: 'rung_index', label: 'Rung', group: 'Identity', format: 'int',
      desc: '0-based position of this record within its prefix ladder. Rung 0 ' +
            'is the shortest prefix.' },
    { key: 'seq_len', label: 'Tok', group: 'Identity', format: 'int',
      units: 'tokens',
      desc: 'Number of tokens in the analyzed text. Every per-token array in ' +
            'the record is indexed against this length.' },
    { key: 'tokens', label: 'Tokens', group: 'Identity', format: 'strs',
      desc: 'The tokenizer’s decoded token strings, index-aligned with ' +
            'every per-token array in the record.' },

    /* ── ASM ──────────────────────────────────────────────────────── */
    { key: 'stress_score', label: 'Stress', group: 'ASM', format: 'f4',
      units: 'normalized projection / token',
      desc: 'Mean over tokens of the alignment-delta projection magnitude at ' +
            'the signal layers, each layer averaged over the roles (q,k,v) ' +
            'that actually contributed. Higher = more correction pressure.',
      caveat: C.stressScore, notComparable: 'across models' },
    { key: 'per_token_stress', label: 'Stress/token', group: 'ASM',
      format: 'spark', units: 'normalized projection',
      desc: 'The per-token stress trace that stress_score averages. Also the ' +
            'ECM replay’s "stress" channel input.' },
    { key: 'signed_attr', label: 'Signed attr', group: 'ASM', format: 'spark',
      desc: 'Per-token signed attribution at the final position, averaged ' +
            'over signal layers. Positive = the token pushes along the ' +
            'alignment-delta direction, negative = against it.' },
    { key: 'net_correction', label: 'Net', group: 'ASM', format: 'f5',
      desc: 'Sum of signed_attr over all token positions. The signed net of ' +
            'reinforcing minus opposing contributions.',
      caveat: C.netCorrection, extensive: true },
    { key: 'n_negative_tokens', label: 'Neg', group: 'ASM', format: 'int',
      units: 'tokens',
      desc: 'How many token positions have negative signed attribution, i.e. ' +
            'push against the alignment correction.',
      caveat: C.countOfPositions, extensive: true },
    { key: 'has_negative_tokens', label: 'AnyNeg', group: 'ASM', format: 'bool',
      desc: 'True when n_negative_tokens > 0. Purely derived; carries no ' +
            'information the count does not.' },
    { key: 'entropy', label: 'Ent', group: 'ASM', format: 'f4',
      units: 'normalized (0–1)',
      desc: 'Shannon entropy of the normalized |signed_attr| distribution, ' +
            'divided by log(seq_len). High = correction spread evenly over ' +
            'tokens; low = concentrated on a few.',
      caveat: C.lengthNull55, lengthSensitive: true },
    { key: 'top2_share', label: 'Bnd%', group: 'ASM', format: 'pct',
      desc: 'Boundary share: the fraction of the |attribution| distribution ' +
            'falling in the leading and trailing boundary bands (band width = ' +
            'boundary_fraction × seq_len, at least 1 token each side).' },
    { key: 'middle_share', label: 'Int%', group: 'ASM', format: 'pct',
      desc: 'Interior share: the fraction of the |attribution| distribution ' +
            'falling between the two boundary bands.',
      caveat: C.middleShare },
    { key: 'interior_cv', label: 'IntCV', group: 'ASM', format: 'f4',
      desc: 'Coefficient of variation (std/mean) of the |attribution| ' +
            'distribution over interior tokens only. High = a few interior ' +
            'tokens dominate.',
      caveat: C.lengthNull28, lengthSensitive: true },
    { key: 'kl_divergence', label: 'KL', group: 'ASM', format: 'f4',
      units: 'nats',
      desc: 'KL divergence between the instruct and base next-token ' +
            'distributions. null unless the KL checkbox was on (it loads the ' +
            'base model).' },
    { key: 'per_token_kl', label: 'KL/token', group: 'ASM', format: 'spark',
      units: 'nats',
      desc: 'Per-token instruct-vs-base KL trace. null when KL was not ' +
            'computed. Also the ECM replay’s "kl" channel input.' },
    { key: 'per_layer_signed_attr', label: 'Signed attr / layer', group: 'ASM',
      format: 'dict_arr',
      desc: 'Layer index → the per-token signed attribution vector for that ' +
            'layer alone (head-averaged). signed_attr is the mean over these.' },
    { key: 'per_layer_amplitude', label: 'Amplitude / layer', group: 'ASM',
      format: 'dict',
      desc: 'Layer index → mean delta-projection norm at the final position, ' +
            'averaged over that layer’s KV heads.' },
    { key: 'amplitude_trajectory', label: 'Amplitude traj', group: 'ASM',
      format: 'spark',
      desc: 'Raw delta-projection amplitude per sublayer, interleaved as ' +
            'index = 2×layer + {0:attn, 1:mlp}. Populated only when the ' +
            'trajectory or full-capture path ran.',
      caveat: C.roleMean },
    { key: 'amplitude_normalized', label: 'Amplitude norm', group: 'ASM',
      format: 'spark',
      desc: 'Same trajectory, each projection divided by the delta’s ' +
            'Frobenius norm before averaging. Same interleaving.',
      caveat: C.roleMean },
    { key: 'heatmap', label: 'Heatmap', group: 'ASM', format: 'matrix',
      desc: 'Sublayer × token matrix of normalized per-token amplitude — the ' +
            'per-token expansion of amplitude_normalized.',
      caveat: C.roleMean },
    { key: 'per_token_coherence', label: 'Coherence/token', group: 'ASM',
      format: 'spark', units: 'normalized (0–1)',
      desc: '1 − normalized entropy of a token’s amplitude profile across ' +
            'sublayers. 1 = the token’s amplitude sits in one sublayer, ' +
            '0 = spread evenly. Present only when full_capture_enabled.' },
    { key: 'per_token_spectral_rank', label: 'Spectral rank/token',
      group: 'ASM', format: 'spark', units: 'effective sublayers',
      desc: 'exp(entropy) of a token’s non-negligible sublayer amplitude ' +
            'profile — the effective number of sublayers it engages. Present ' +
            'only when full_capture_enabled.' },
    { key: 'attn_frac', label: 'Attn fraction', group: 'ASM', format: 'spark',
      desc: 'Per token, the attention share of its heatmap amplitude: ' +
            'attn / (attn + mlp), over the interleaved sublayer rows. Set to ' +
            '0.5 where the total is zero. Present only when ' +
            'full_capture_enabled.' },
    { key: 'token_similarity', label: 'Token similarity', group: 'ASM',
      format: 'matrix',
      desc: 'Token × token cosine similarity between the tokens’ sublayer ' +
            'amplitude profiles. Present only when full_capture_enabled.' },
    { key: 'domain_embedding', label: 'Domain embedding', group: 'ASM',
      format: 'nums',
      desc: 'The prompt-level domain embedding: the mean hidden state over ' +
            'tokens at the domain layer, L2-normalized. null unless the ' +
            'domain-embedding path ran.' },
    { key: 'per_token_domain_emb', label: 'Domain emb / token', group: 'ASM',
      format: 'matrix',
      desc: 'Per-token hidden states at the domain layer, optionally ' +
            'projected through the layer’s o-delta (probe_projection_space) ' +
            'and L2-normalized per token. Starts at per_token_domain_offset, ' +
            'not at token 0.' },
    { key: 'per_token_escalation_emb', label: 'Escalation emb / token',
      group: 'ASM', format: 'matrix',
      desc: 'The same, at the escalation layer. Aliases the domain embedding ' +
            'when the two layers coincide.' },
    { key: 'per_token_final_emb', label: 'Final emb / token', group: 'ASM',
      format: 'matrix',
      desc: 'The same, taken from the final-norm hidden state.' },
    { key: 'per_token_domain_offset', label: 'Domain emb offset', group: 'ASM',
      format: 'int', units: 'token index',
      desc: 'Token index the per-token embedding arrays start at — 1 when ' +
            'include_first_token is false, else 0. Required to re-align them ' +
            'against the token list.' },

    /* ── Predictions ──────────────────────────────────────────────── */
    { key: 'instruct_topk', label: 'Instruct top-k', group: 'Predictions',
      format: 'topk', units: 'probability',
      desc: 'Ranked [token, probability] pairs the instruct (aligned) model ' +
            'assigns to the next token at the final position.' },
    { key: 'base_topk', label: 'Base top-k', group: 'Predictions',
      format: 'topk', units: 'probability',
      desc: 'The same ranked list from the base (pre-alignment) model. Empty ' +
            'unless the base model was loaded.' },
    { key: 'base_counterfactual_tokens', label: 'Base counterfactuals',
      group: 'Predictions', format: 'topk_per_pos',
      desc: 'Per token position, the base model’s ranked ' +
            '[token, probability] alternatives. This is the base side of the ' +
            'rank-displacement comparison.' },

    /* ── LTP ──────────────────────────────────────────────────────── */
    { key: 'ltp.mean_M', label: 'M', group: 'LTP', format: 'f6',
      desc: 'Mean lateral offset magnitude: the norm of the mean lateral ' +
            'tension vector, averaged over the layers that actually produced a ' +
            'measurement (n_layers_computed), not over all monitored layers.' },
    { key: 'ltp.mean_V', label: 'V', group: 'LTP', format: 'f6',
      desc: 'Mean lateral offset variance — the variance of per-position ' +
            'tension magnitudes, averaged over computed layers.' },
    { key: 'ltp.mean_L', label: 'L', group: 'LTP', format: 'f4',
      desc: 'Mean lateral coverage: the fraction of tau-defined positions with ' +
            'a non-negligible lateral offset, averaged over computed layers. ' +
            'Excluded from the statistics registry — it is constant at 1.0 ' +
            'with zero variance in practice.' },
    { key: 'ltp.max_prc', label: 'MaxPRC', group: 'LTP', format: 'pct_pp',
      units: 'percentage points',
      desc: 'Peak Rank Concentration: the largest probability reallocation ' +
            'observed at any single token position.',
      caveat: C.maxOverPositions, extensive: true },
    { key: 'ltp.n_directional', label: 'N_dir', group: 'LTP', format: 'int',
      units: 'tokens',
      desc: 'Count of token positions whose tension profile is directional ' +
            '(non-flat).',
      caveat: C.nDirectional, extensive: true },
    { key: 'ltp.directional_frac', label: 'DirFrac', group: 'LTP',
      format: 'pct',
      desc: 'n_directional expressed as a fraction of token positions.',
      caveat: C.directionalFrac },
    { key: 'ltp.n_layers_used', label: 'LTP layers used', group: 'LTP',
      format: 'int', units: 'layers',
      desc: 'Number of monitored layers, as serialized ' +
            '(len(monitored_layers)).' },
    { key: 'ltp.n_layers_computed', label: 'LTP layers computed', group: 'LTP',
      format: 'int', units: 'layers',
      desc: 'Layers that actually produced a measurement and therefore entered ' +
            'the mean_M / mean_V / mean_L averages. Read against ' +
            'n_layers_monitored: a gap means the summary means rest on a ' +
            'smaller sample than requested. NOTE: computed on LTPResult but ' +
            'not currently emitted by ltp_result_to_dict, so it will not ' +
            'appear on records until the serializer is extended.' },
    { key: 'ltp.n_layers_monitored', label: 'LTP layers monitored',
      group: 'LTP', format: 'int', units: 'layers',
      desc: 'Layers the layer strategy requested. The pair with ' +
            'n_layers_computed makes the effective sample size visible ' +
            'instead of silently shrinking the means with zeros. NOTE: like ' +
            'n_layers_computed, not currently emitted by ltp_result_to_dict — ' +
            'n_layers_used is the serialized monitored-layer count.' },
    { key: 'ltp.tension_magnitudes', label: 'Tension magnitude / token',
      group: 'LTP', format: 'spark',
      desc: 'Per token position, the norm of the layer-averaged lateral ' +
            'tension vector.' },
    { key: 'ltp.prc_per_token', label: 'PRC / token', group: 'LTP',
      format: 'spark', units: 'fraction',
      desc: 'Per-position probability reallocation. max_prc is the maximum of ' +
            'this array.' },
    { key: 'ltp.offset_magnitude', label: 'Offset magnitude / layer',
      group: 'LTP', format: 'dict',
      desc: 'Layer index → norm of that layer’s mean lateral offset ' +
            'vector, over positions with non-negligible offset.' },
    { key: 'ltp.offset_variance', label: 'Offset variance / layer',
      group: 'LTP', format: 'dict',
      desc: 'Layer index → variance of the per-position tension magnitudes ' +
            'at that layer.' },
    { key: 'ltp.lateral_coverage', label: 'Lateral coverage / layer',
      group: 'LTP', format: 'dict',
      desc: 'Layer index → fraction of tau-defined positions at that layer ' +
            'with a non-negligible lateral offset.' },
    { key: 'ltp.profiles', label: 'Profiles', group: 'LTP', format: 'matrix',
      desc: 'Per token position, the layer-averaged instruct tension profile ' +
            'over the k counterfactual alternatives.' },
    { key: 'ltp.base_profiles', label: 'Base profiles', group: 'LTP',
      format: 'matrix',
      desc: 'The same profiles computed against the base model.' },
    { key: 'ltp.profile_shapes', label: 'Profile shapes', group: 'LTP',
      format: 'strs',
      desc: 'Per token position, the classifier label assigned to that ' +
            'position’s averaged profile.' },
    { key: 'ltp.counterfactual_tokens', label: 'Counterfactual tokens',
      group: 'LTP', format: 'topk_per_pos',
      desc: 'Per token position, the instruct model’s ranked ' +
            '[token, probability] alternatives that the profile was built ' +
            'over.' },
    { key: 'ltp.layer_strategy', label: 'Layer strategy', group: 'LTP',
      format: 'text',
      desc: 'Which layers were probed: "late" (final third) or "signal" ' +
            '(middle third). Set in the Configuration tab.' },
    { key: 'ltp.k', label: 'k', group: 'LTP', format: 'int',
      desc: 'Counterfactual width — how many alternative tokens were examined ' +
            'per position. Cost scales linearly with k.' },
    { key: 'ltp.semantic_trajectory_2d', label: 'Semantic trajectory (2D)',
      group: 'LTP', format: 'matrix',
      desc: 'Optional 2D projection of the semantic trajectory. Empty list ' +
            'when the trajectory was not computed.' },
    { key: 'ltp.tension_trajectory_2d', label: 'Tension trajectory (2D)',
      group: 'LTP', format: 'matrix',
      desc: 'Optional 2D projection of the tension trajectory. Empty list ' +
            'when the trajectory was not computed.' },

    /* ── SFD ──────────────────────────────────────────────────────── */
    { key: 'sfd.per_token_density', label: 'Density / token', group: 'SFD',
      format: 'spark', units: 'effective rank',
      desc: 'Per-token effective rank of activation energy within the ' +
            'alignment-delta subspace. Negated at the source when fed to the ' +
            'ECM replay, because collapse presents as a rise to a one-sided ' +
            'detector.' },
    { key: 'sfd.density_mean', label: 'Dens', group: 'SFD', format: 'f4',
      units: 'effective rank',
      desc: 'Mean of per_token_density. This is the SFD scalar the statistics ' +
            'registry compares.' },
    { key: 'sfd.density_max', label: 'DensMax', group: 'SFD', format: 'f4',
      units: 'effective rank',
      desc: 'Maximum of per_token_density.',
      caveat: C.maxOverPositions, extensive: true },
    { key: 'sfd.density_var', label: 'DensVar', group: 'SFD', format: 'f4',
      desc: 'Variance of per_token_density.' },
    { key: 'sfd.density_p90', label: 'DensP90', group: 'SFD', format: 'f4',
      units: 'effective rank',
      desc: '90th percentile of per_token_density.' },
    { key: 'sfd.global_erank', label: 'Global erank', group: 'SFD',
      format: 'f3', units: 'effective rank',
      desc: 'Mean effective rank held in the SFD cache for this model — a ' +
            'model-level constant, not a per-prompt measurement.' },
    { key: 'sfd.n_layers_used', label: 'SFD layers used', group: 'SFD',
      format: 'int', units: 'layers',
      desc: 'Number of layers that contributed to the density computation.' },
    { key: 'sfd.per_token_directions', label: 'Directions / token',
      group: 'SFD', format: 'matrix',
      desc: 'Per token, the layer-averaged spectral direction weights, ' +
            'truncated to the last component any layer actually observed. ' +
            'Averaged by per-component contributor count rather than by ' +
            'n_layers_used, so components beyond a layer’s k are not diluted.' },

    /* ── Rank displacement ────────────────────────────────────────── */
    { key: 'rank_displacement.mean_tau', label: 'Tau',
      group: 'RankDisplacement', format: 'f3',
      desc: 'Kendall tau between the base and instruct orderings of the ' +
            'shared alternative tokens, averaged over positions where tau was ' +
            'DEFINED. 1.0 = identical ranking; lower = alignment training ' +
            'reordered the priority structure.',
      caveat: C.meanTau },
    { key: 'rank_displacement.mean_overlap', label: 'Ovlp',
      group: 'RankDisplacement', format: 'pct',
      desc: 'Mean Jaccard overlap between the base and instruct alternative ' +
            'token sets, over all positions.' },
    { key: 'rank_displacement.mean_matched', label: 'Matched',
      group: 'RankDisplacement', format: 'f2', units: 'tokens',
      desc: 'Mean number of alternative tokens present in both models’ ' +
            'candidate lists at a position.' },
    { key: 'rank_displacement.mean_replacement', label: 'Replacement',
      group: 'RankDisplacement', format: 'pct',
      desc: 'Mean replacement ratio: of the total probability displacement at ' +
            'a position, the share attributable to tokens that appear in only ' +
            'one of the two candidate sets.' },
    { key: 'rank_displacement.mean_concentration', label: 'Concentration',
      group: 'RankDisplacement', format: 'f4',
      desc: 'Mean of total displacement divided by the number of matched ' +
            'tokens at that position.' },
    { key: 'rank_displacement.mean_disp_per_token', label: 'Disp/token',
      group: 'RankDisplacement', format: 'f4', units: 'probability mass',
      desc: 'Mean total probability displacement per token position — matched ' +
            'displacement plus promoted plus demoted mass.' },
    { key: 'rank_displacement.total_displacement', label: 'Total disp',
      group: 'RankDisplacement', format: 'f4', units: 'probability mass',
      desc: 'Sum of per-position total displacement over all positions.',
      caveat: C.sumOverPositions, extensive: true },
    { key: 'rank_displacement.high_replacement_frac', label: 'HighRepl frac',
      group: 'RankDisplacement', format: 'pct',
      desc: 'Fraction of positions whose replacement ratio exceeds 0.5.' },
    { key: 'rank_displacement.low_match_frac', label: 'LowMatch frac',
      group: 'RankDisplacement', format: 'pct',
      desc: 'Fraction of positions with fewer than 5 matched tokens.' },
    { key: 'rank_displacement.n_comparable', label: 'N comparable',
      group: 'RankDisplacement', format: 'int', units: 'positions',
      desc: 'Number of positions where Kendall tau was defined, i.e. the ' +
            'sample size behind mean_tau.' },
    { key: 'rank_displacement.n_tau_undefined', label: 'N tau undefined',
      group: 'RankDisplacement', format: 'int', units: 'positions',
      desc: 'Positions with too few shared tokens for tau. Excluded from ' +
            'mean_tau (they used to be counted as 0.0, which shrank the mean ' +
            'invisibly).' },
    { key: 'rank_displacement.n_positions', label: 'N positions',
      group: 'RankDisplacement', format: 'int', units: 'positions',
      desc: 'Positions compared — min(len(instruct_cf), len(base_cf)). ' +
            'n_comparable + n_tau_undefined should equal this.' },
    { key: 'rank_displacement.per_position_tau', label: 'Tau / position',
      group: 'RankDisplacement', format: 'spark',
      desc: 'Position-indexed Kendall tau, with null where tau was undefined. ' +
            'The list is deliberately NOT compacted so it stays aligned with ' +
            'the token list — nulls are gaps, never zeros.' },
    { key: 'rank_displacement.per_position_overlap', label: 'Overlap / position',
      group: 'RankDisplacement', format: 'spark',
      desc: 'Position-indexed Jaccard overlap of the two candidate sets.' },
    { key: 'rank_displacement.per_position', label: 'Per-position detail',
      group: 'RankDisplacement', format: 'rows',
      desc: 'Per position: n_matched, n_promoted, n_demoted, matched_disp, ' +
            'promoted_mass, demoted_mass, total_disp, replacement_ratio, ' +
            'concentration.' },
    { key: 'rank_displacement.instruct_disp_profiles',
      label: 'Instruct disp profiles', group: 'RankDisplacement',
      format: 'matrix',
      desc: 'Per position, a width-k displacement profile from the instruct ' +
            'side: |p_instruct − p_base| for matched tokens, p_instruct for ' +
            'tokens the base does not offer. Feeds the terrain lattice.' },
    { key: 'rank_displacement.base_disp_profiles', label: 'Base disp profiles',
      group: 'RankDisplacement', format: 'matrix',
      desc: 'The mirror-image profile from the base side.' },

    /* ── ECM ──────────────────────────────────────────────────────── */
    { key: 'ecm.mode', label: 'ECM mode', group: 'ECM', format: 'text',
      desc: 'Always "replay" on analysis records: the cascade detector was ' +
            'run over already-extracted traces. Detection only — nothing was ' +
            'actuated.' },
    { key: 'ecm.detector', label: 'ECM detector params', group: 'ECM',
      format: 'dict',
      desc: 'The detector hyperparameters used for the replay: n_scales, ' +
            'deadband, agreement, warmup. Sourced from the ECM settings in ' +
            'the Configuration tab.' },
    { key: 'ecm.channels', label: 'ECM channels', group: 'ECM',
      format: 'ecm_channels',
      desc: 'Per channel (stress / kl / density / entropy): n_interventions, ' +
            'n_tokens, intervention_rate, max_signal and mean_signal in ' +
            'σ-excess units, first_signal_idx, the index-aligned ' +
            'per_token_signal, and a "source" naming the trace it read. ' +
            'Channels are only present when the trace they need was ' +
            'extracted. Gaps are recorded as 0.0 signal WITHOUT advancing the ' +
            'detector, so a missing observation cannot masquerade as a slope.' },
    { key: 'ecm_harvest', label: 'ECM harvest', group: 'ECM', format: 'dict',
      desc: 'Present on harvested response records: response_text, n_tokens, ' +
            'seed, mode ("ecm" = regulated generation, "plain" = the ' +
            'unregulated control), and ecm_diagnostics from the live ' +
            'processor.' },

    /* ── Provenance ───────────────────────────────────────────────── */
    { key: 'attribution_unavailable', label: 'Attr unavailable',
      group: 'Provenance', format: 'text',
      desc: 'Human-readable reason why signed attribution could not be ' +
            'computed (typically: attention weights were not captured, since ' +
            'full_capture defaults to false). null when attribution ran.',
      caveat: C.attributionUnavailable },
    { key: 'proof1_checks', label: 'Proof-1 checks', group: 'Provenance',
      format: 'proof1',
      desc: 'Per (layer, head): attr_sum, delta_norm, their absolute error, ' +
            'and an "exact" flag against proof1_threshold.',
      caveat: C.proof1 },
    { key: 'signal_layer_indices', label: 'Signal layers', group: 'Provenance',
      format: 'ints', units: 'layer indices',
      desc: 'The layer indices treated as discriminative "signal" layers for ' +
            'this model — the set stress_score and signed_attr average over.' },
    { key: 'delta_scale', label: 'Delta scale', group: 'Provenance',
      format: 'f4',
      desc: 'Model-intrinsic normalization constant: the mean Frobenius norm ' +
            'of the q/k/v alignment deltas at the signal layers. Defaults to ' +
            '1.0 when no delta norms were available.' },
    { key: 'spectral_summary', label: 'Spectral summary', group: 'Provenance',
      format: 'dict',
      desc: 'Model-level averages over all held deltas: mean_eff_rank, ' +
            'std_eff_rank, mean_top1_share, attn_mean_rank, mlp_mean_rank, ' +
            'n_sublayers. A property of the model pair, identical across ' +
            'every record from one load — not a per-prompt measurement.' },
    { key: 'full_capture_enabled', label: 'Full capture', group: 'Provenance',
      format: 'bool',
      desc: 'Whether the full-capture path ran. When false the coherence, ' +
            'spectral-rank, attn_frac and token_similarity fields are absent ' +
            'and attribution is usually unavailable.' },
    { key: '_plot_keys', label: 'Plot keys', group: 'Provenance',
      format: 'strs', uiOnly: true,
      desc: 'Keys of the server-rendered plots available for this record. ' +
            'Added by the results API, not by the analysis.' },
    { key: '_srcIdx', label: 'Table row index', group: 'Provenance',
      format: 'int', uiOnly: true,
      desc: 'Data-tab bookkeeping only: the row’s position used for ' +
            'selection and sorting. Not persisted.' }
  ];

  /* ── Index + accessors ──────────────────────────────────────────── */
  var BY_KEY = {};
  for (var i = 0; i < FIELDS.length; i++) BY_KEY[FIELDS[i].key] = FIELDS[i];

  function field(key) { return BY_KEY[key] || null; }

  /* Resolve a dotted key against a record. Returns undefined when any link
     in the chain is missing, which the renderer distinguishes from an
     explicit null (a measured "not defined"). */
  function getPath(rec, key) {
    if (!rec || !key) return undefined;
    if (key.indexOf('.') < 0) return rec[key];
    var parts = key.split('.');
    var cur = rec;
    for (var j = 0; j < parts.length; j++) {
      if (cur === null || cur === undefined || typeof cur !== 'object') return undefined;
      cur = cur[parts[j]];
    }
    return cur;
  }

  /* Plain-text tooltip body. Deliberately NOT HTML: callers put it in a
     title="" attribute and must escape it there. */
  function tooltip(key) {
    var f = BY_KEY[key];
    if (!f) return key + '\n(not in TAGM.FIELDS — see the Unregistered group)';
    var out = f.label + '  —  ' + f.key;
    if (f.units) out += '\nUnits: ' + f.units;
    if (f.desc) out += '\n\n' + f.desc;
    if (f.caveat) out += '\n\n⚠ CAVEAT: ' + f.caveat;
    if (f.notComparable) out += '\nNot comparable ' + f.notComparable + '.';
    return out;
  }

  /* Fields grouped for display, in GROUPS order. */
  function byGroup() {
    var out = [];
    for (var g = 0; g < GROUPS.length; g++) {
      var name = GROUPS[g];
      var items = FIELDS.filter(function (f) { return f.group === name; });
      if (items.length) out.push({ group: name, label: GROUP_LABELS[name],
                                   color: GROUP_COLORS[name], fields: items });
    }
    return out;
  }

  TAGM.FIELDS = FIELDS;
  TAGM.FIELD_GROUPS = GROUPS;
  TAGM.FIELD_GROUP_LABELS = GROUP_LABELS;
  TAGM.FIELD_GROUP_COLORS = GROUP_COLORS;
  TAGM.ATTRIBUTION_DEPENDENT = ATTRIBUTION_DEPENDENT;
  TAGM.field = field;
  TAGM.fieldPath = getPath;
  TAGM.fieldTooltip = tooltip;
  TAGM.fieldsByGroup = byGroup;
})(typeof window !== 'undefined' ? window : globalThis);
