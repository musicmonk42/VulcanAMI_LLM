# ADR 004: Complete Vulcan AMI constitution and target architecture

## Status

Accepted as a constitutional target architecture. Production code remains limited by ADR-003: the Docker image serves `vulcan.runtime.app:app`, and `RuntimeContainer` is the owner of the current production composition. This ADR separates production authority from research aspirations so later prompts can promote capabilities without adding a second authority.

## Context inspected

This decision is grounded in ADR-003, the Graphix philosophy and arena documentation, CSIU phase-9 dataflow, the local-language release boundary, world-model orchestration notes, and the learning capability matrix. Those sources show useful components but also legacy paths that can reason, learn, or influence plans outside one durable owner.

## Decision

Vulcan AMI is a cognitive microkernel architecture. The microkernel is the only production authority that may convert proposals into committed state, plans, effects, memory, episode terminalization, or advertised capabilities. All other components are bounded organs with typed ports:

| Component | Production role | Authority ceiling |
| --- | --- | --- |
| Cognitive microkernel | Owns the control boundary, transaction boundary, capability publication, durable commits, and effect publication. | `EXECUTED_EFFECT` |
| Language cortex, including internal LLMs and OpenAI | Converts between human language and typed proposals or explanations. It never supplies executable semantics, evidence, policy, belief authority, or tool authority. | `UNTRUSTED_PROPOSAL` |
| Graphix | Structured intermediate representation and execution substrate after validation and authorization. Graphix never owns identity, belief, or memory authority. | `VALIDATED_CANDIDATE` unless microkernel-authorized |
| Neurosymbolic workspace | Holds candidate symbols, proofs, simulations, and deliberation traces for review. | `VALIDATED_CANDIDATE` |
| World/self/social models | Produce candidate observations, predictions, user-separateness facts, and risk assessments. | `VALIDATED_CANDIDATE` |
| CSIU homeostasis | Produces typed pressure, safety, utility, and influence proposals under caps. It is proposal-only until governed promotion. | `VALIDATED_CANDIDATE` |
| Learning/development | Runs shadow learning and proposes promotions through CAS-governed review. It may not self-activate. | `VALIDATED_CANDIDATE` |
| Relational extended-self interface | Represents relationships, commitments, consent records, and boundaries without owning or subsuming humans. | `COMMITTED_BELIEF` only through microkernel commit |

## Boundaries

- **Control boundary:** the running AMI consists only of components whose lifecycle and authority are owned by the microkernel/runtime container. Humans, providers, tools, documents, training data, and remote services are outside control and enter as evidence or proposals.
- **Identity-continuity boundary:** durable AMI identity is the microkernel's committed state lineage: code image digest, constitution digest, policy digests, memory root, audit root, and capability registry digest at an exact transaction boundary. Restart reconciliation must choose the last complete transaction or fail closed.
- **Relational-self boundary:** relationships may be represented as commitments and obligations, but humans remain independent actors. A relationship is never system-owned memory, never a license to manipulate, and never a claim that a human is part of the AMI.
- **Moral-concern boundary:** humans and affected non-system actors are inside moral concern and outside ownership/control. Their consent, autonomy, contestability, privacy, and right of exit constrain system action.

## Capability classes

- **Production authority:** implemented in the supported runtime, durable across restart, covered by tests, and listed in the capability registry by the microkernel.
- **Research component:** present in repository or docs but disabled, shadow-only, or proposal-only.
- **Aspirational component:** target architecture with no production claim.
- **Prohibited behavior:** behavior forbidden even if a legacy module currently appears to support it.

## Invariants

Stable machine-checkable invariant identifiers are encoded in `docs/architecture/ami-invariants.yaml`. Later code must cite these IDs instead of inventing new constitutional names. The constitution prohibits human ownership, hidden influence, direct CSIU activation, model-authored executable semantics, and live serving-container source mutation.

## Migration and rollback

Promotion of any research component requires a typed adapter into the microkernel, canonical serialization, full digests, CAS-governed commits, audit evidence, capability-registry update, and fault-injection tests around externally observable transitions. Rollback is selecting a prior image/commit or disabling a governed capability at the microkernel boundary; it is not mutating a live serving container.
