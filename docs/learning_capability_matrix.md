# Learning Capability Matrix

This matrix is descriptive, not promotional. A component is not active unless the
canonical `LearningOwner` reports `ACTIVE` with a verified evaluation or release
identifier. This phase does not claim that the overall learning system is
ready for production use.

| Component | Implemented behavior | Operating status | Governing owner | Acceptance test / evaluation | Known limitations |
|---|---|---|---|---|---|
| Online `/learn` API | Endpoint exists but returns fail-closed 501 | Unavailable | LearningOwner / API middleware | `tests/test_learning_containment.py` | No online learning updates are accepted. |
| Canonical observation + outbox | Validated observation contract and transactional outbox | Shadow-only | LearningOwner | `tests/test_learning_observation_contract.py`, `tests/test_learning_outbox.py` | Does not train or activate policies. |
| Tool-selection bandit | Candidate LinUCB policy learns from verified observations | Shadow-only | LearningOwner | `tests/test_learning_shadow_bandit.py` | Cannot influence live routing without governance. |
| Governed activation | Proposal, CSIU/alignment review, CAS activation boundary | Shadow-only | LearningOwner | `tests/test_learning_governance.py` | Requires external approvals; no automatic promotion. |
| Metacognition | Observation and recommendations | Observe-only | LearningOwner | `tests/test_learning_containment.py` | Runtime mutation application is unavailable. |
| Progressive continual learning | Isolated research implementation and state restore tests | Experimental | None in production; research-only constructor | `tests/test_progressive_research.py` | Production activation is rejected. |
| FOMAML | Isolated first-order MAML research update | Experimental | None in production; research-only MetaLearner | `tests/test_fomaml_research.py` | MAML and PROTO remain unavailable. |
| RLHF shadow reward | Deterministic feedback encoder and shadow reward candidate | Experimental | None in production; research-only trainer | `tests/test_rlhf_shadow_reward.py` | PPO is disabled and cannot affect policy. |
| World-model planning | Isolated deterministic one-planner research candidate | Experimental | None in production; research-only model | `tests/test_world_model_research.py` | Production world model is disabled; CEM/MPPI unproven. |
| MAML | Enum retained for compatibility | Unavailable | LearningOwner | None | Activation returns unavailable until separately proven. |
| PROTO | Enum retained for compatibility | Unavailable | LearningOwner | None | Activation returns unavailable until separately proven. |
| PPO | Historical configuration retained | Unavailable | LearningOwner | None | Policy update path returns unavailable. |
| PackNet | Historical references retained | Unavailable | LearningOwner | None | No verified active implementation. |
| Supervised/federated/transfer learning | Historical/compatibility claims only | Unavailable | LearningOwner | None | No production update authority. |
| Autonomous self-improvement | Governed source-code improvement exists separately | Unavailable as runtime learning | Governance/audit systems | Self-improvement governance tests | Not a learning mutation authority. |
| Hallucination prevention | Safety/audit controls may reduce risk | Unavailable as a guaranteed capability | Safety/audit systems | None as learning capability | No guarantee is claimed. |
