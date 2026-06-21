# Progress — E_fin1 FinQA Skill-Verifier Replication

**STATUS: COMPLETE — NULL RESULT.** Adversarial lift = 0.99× on FinQA (vs 2.3× coding).
See `results/analysis.md`.

| Stage | Status | Notes |
|-------|--------|-------|
| 0 Carryover audit | ✅ | 1 P0 (operationalize tolerance+schema+env in smoke test), 2 P1 (v009 threshold explicit = v009-only 4/4 unanimous, no v001 gate; cost pre-check), 1 P2 (cross-verifier fallback). All addressed. |
| 1 Data + scaffold | ✅ | czyssrs/FinQA cloned; n=100 dev sampled (seed=42). All 100 numeric. Schema `qa.exe_ans` verified. |
| 0 Smoke test | ✅ | 15/15 scorer checks (incl. FinQA percent/ratio reconciliation), schema, live Bedrock. **Caught a tolerance bug before n=100.** |
| Generate answers | ✅ | Haiku, n=100. Base rate = **0.71**. Gen cost $0.111. |
| Run verifiers | ✅ | Confirmatory (1 call) + adversarial (4-call ensemble). 500 Bedrock calls. |
| Analyze + report | ✅ | See below + analysis.md + lessons.md. |
| RQ2 cross-verifier (Nova Pro) | ✅ | n=40. Haiku AND Nova both 0.92× lift — null is domain-structural, not model-specific. |

## Headline numbers (n=100, Haiku verifier)
| Metric | Confirmatory | Adversarial (4/4) |
|--------|-------------:|------------------:|
| Precision (confident subset) | 0.745 | 0.740 |
| Recall | 0.986 | 0.761 |
| AUC | 0.629 | 0.565 |
| **Lift adv/conf** | — | **0.99×** (coding: 2.3×) |
| Cost/eval | $0.00125 | $0.01341 (under $0.03 ✅) |

## Verdict
- **RQ1 (does adversarial lift transfer?): NO.** Lift 0.99×. Both verifiers at base rate. Applicability-bounding negative result.
- **RQ2 (calibration off-coding?): Claude engages, does not discriminate** (AUC 0.57). Abstention is NOT the failure mode — confident-but-wrong is. Mechanism: no verification asymmetry (verifier must redo the same numeric reasoning as the agent; same-tier → same mistakes).
- **RQ3 (cost ceiling?): holds.** $0.0134/eval = 45% of $0.03.
- **Robustness**: null lift (1.01×) survives label-noise correction (11/29 "wrong" were scoring artifacts — the verifier-reward semantic-mismatch lesson reproduced).

## Carryover audit reflections (carried from verifier-reward / verification-primitives)
- v009 4/4 unanimous used as the confident threshold (no v001 gate) ✅
- Stage-0 smoke test operationalized (scorer + schema + env) ✅ — caught a real bug
- $0.03 ceiling measured, not assumed ✅
- recall ceiling = semantic-mismatch reproduced and audited ✅
