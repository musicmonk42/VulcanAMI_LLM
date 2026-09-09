# NPT flagship experiment preregistration

## Version and corrective history

This document freezes `npt-flagship.v2` before any Vulcan telemetry is analysed.
Version 1 is invalidated: its 0.25 F1 recovery threshold tolerated excessive
false positives, its row-level predictors could observe the same time slice as
the outcome, and intervention families appeared in both training and test data.
No scientific claim may rely on that run. Version 2 repairs those defects and
retains the negative-result rule; the checked report is a synthetic pipeline
qualification, not an observation about Vulcan.

## Dataset and adversarial design

`SyntheticCyclicWorld.v2` uses seed `2201080`, 160 trials, and 16 steps per
trial. It declares a nine-edge lagged graph, including the two autoregressive
state edges actually implemented by the generator. Each even-numbered trial has
current closed-loop control. The following odd trial receives exactly the same
observation and action history but lacks current counterfactual action control.
Each pair also receives the same intervention label.

Center, policy, action, and reafference intervention families are training data.
Boundary, memory, valuation, and no-op intervention families are held out in
full. This produces 1,200 training and 1,200 held-out within-trial transitions;
no row or intervention family crosses the partition. Features come exclusively
from the preceding time step, while outcomes come from the next time step.
Outcomes are ownership, boundary, agency, temporal currentness, and internal
state, and every model is scored separately on all five.

## Structure recovery

Before comparative scoring, the blind instrumentor fits one standardized,
multivariate, lagged ridge model per telemetry variable using only closed-loop
telemetry (`ridge=0.02`). It accepts an edge at absolute standardized
coefficient `>= 0.12`.
Precision and recall must each be at least `0.90` against the declared graph.
Thresholds are fixed and are not selected on the comparative held-out data.

## Rival models and scoring

The NPT model consumes prior center, boundary, value, policy, reafference,
memory, and the current-control condition. Fixed rivals operationalize generic
integration, recurrence, broadcast/global availability, higher-order
representation, and report behavior. All use the same deterministic ridge
implementation (`ridge=0.01`). Score is held-out mean squared error plus `0.002`
per feature. The incremental result compares each model's mean penalized score
across the five preregistered outcomes.

NPT receives incremental support only if structure recovery passes and its mean
penalized held-out score is strictly lower than every rival. Otherwise the
report records falsification: the NPT-specific variables added no incremental
value beyond all rivals and the theory must be revised.

## Outputs, isolation, and limitations

The instrumentor emits only typed closure measurements, the enumerated evidence
classifications, model diagnostics, and the falsification result. It is outside
the authority and candidate-agent loops and cannot commit beliefs, authorize
plans, execute effects, or mutate memory. Input rows contain numeric research
telemetry and bounded identifiers only; prompts, provider text, secrets, and
private reasoning traces are excluded. The entire package remains denied by the
production import policy.

Synthetic success can qualify the measurement pipeline but cannot establish
behavior in Vulcan or an unknown system. Gate F is incomplete, so this version
must not consume Vulcan telemetry and cannot satisfy Gate G or M5.

## Reproduction

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from vulcan.research.npt import run_flagship_experiment
run_flagship_experiment().write(Path("docs/research/npt-flagship-report.json"))
PY
```
