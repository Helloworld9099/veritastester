# M10: Runnable one-encoder, two-output CPU prototype

This [local probe](../research/unified_models/causal_dual_head.py) executes a small, fixed, **untrained** recurrent encoder and two independent numeric output heads on generated 16 kHz PCM. It verifies the data flow and cache mechanics, not speech recognition or translation. The ES→EN and FR→EN labels exercise direction changes; they are not support claims. No model download, fitting, issue of a speech qualification, provider call, GitHub Actions, or app integration occurred.

```text
new PCM → 20 ms frames → recurrent encoder → one immutable acoustic state
                           ↑ previous state     ├─ source projection
                                                  └─ translation projection
```

Each neural step receives only a new frame and the previous eight-value state. The only retained PCM is the incomplete next frame (at most 319 samples); end of capture consumes any short tail exactly once, without padding or reprocessing past audio. Both heads read the identical state object; route-specific bias is applied within a single resident coefficient set. Closing an utterance clears acoustic state; a direction change does not instantiate a second set of coefficients or encoder. Draft outputs are numeric snapshots after each push producing at least one complete frame. Their update frequency therefore depends on push boundaries; encoder steps and the final pair do not.

## Local evidence

```sh
PYTHONPATH=.:src python -m research.unified_models.causal_dual_head
PYTHONPATH=.:src python -m research.unified_models.causal_dual_head --trace
PYTHONPATH=.:src python -m pytest -q tests/test_causal_dual_head.py tests/test_incremental_evidence.py
```

The [reproducible report](../research/unified_models/causal_dual_head_report.json) records one in-memory model load, two direction-labeled utterances, and ten encoder calls covering 2,802 generated samples. Each utterance produced three paired drafts and one paired final. Maximum retained PCM in this run was 314 samples between pushes. The redacted trace passes the M9 structural gate and uses logical ticks, **not latency**. The coefficient digest binds deterministically constructed values, not a trained checkpoint file.

Direct tests inspect the raw input to the actual encoder and the state received by the actual heads. They establish exact encoder and final-output equality across whole, one-sample and irregular push partitions; verify the same prior state object is carried to the next step; show that prior audio affects the next hidden state and that different future audio cannot alter an already emitted pair; and verify the state reset and PCM budget. This tiny probe gives no information about throughput of a learned encoder, meaning preservation, or app readiness. It is deliberately outside the qualified runtime.

## Next useful experiment

Identify one rights-compatible streaming foundation whose acoustic state is reusable and whose training permits a dual-caption head. Before fitting anything, pin the base revision and tokenizer, check warm-start and licensing compatibility, and establish an allowed dual-labeled development recipe plus separate holdout protocol. An ASR-only foundation is not a unified translator. With no qualified checkpoint or authorized training budget, keep the demo numeric and the app unchanged. Rollback is to stop using the probe or revert this research-only commit; there is no production flag or artifact to remove.
