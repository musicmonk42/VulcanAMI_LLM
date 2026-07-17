# Sequence-6 baseline audit

* Date: 2026-07-17 UTC
* Branch/HEAD before this follow-up: `work` / `1de49512ca01b42c25f87bda8ec83d914112a7d9`.
* Canonical Docker entrypoint: `uvicorn vulcan.runtime.app:app`; package root is
  `/app/src` according to `Dockerfile`.
* Current supported domain/locale: NFC-normalized `und` bounded arithmetic;
  deterministic strict response templates.

## Prerequisite evidence

`python -m pytest -q tests/security` exited during collection because
`tests/conftest.py` imports unavailable `numpy`.  This is an environment
limitation, not evidence that Sequences 1–5 pass.  Thus no neural or OpenAI
adapter is attached.  The runtime remains deterministic-only.

## Reachability disposition

| Surface | Disposition |
|---|---|
| `CognitiveKernel` | only canonical consumer of narrow input/output ports |
| `RuntimeContainer` | owns and closes deterministic ports once |
| `LocalGPTProvider`, Graphix/BERT loaders, hybrid executor, OpenAI client | not imported by `vulcan.runtime` production closure |
| training, distillation, self-improvement | offline/non-serving; not imported by canonical runtime |
| `verify_release` | offline artifact check only; not a loader or selector |

No candidate bake-off, model license decision, model download, training data,
or external inference authorization exists in this worktree.  Consequently
Sequence 6 remains incomplete for neural activation, while the deterministic
language-interface baseline is mechanically wired and tested.
