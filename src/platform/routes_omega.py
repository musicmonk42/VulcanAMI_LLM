"""
Omega phases 4-5 route handlers extracted from full_platform.py.

Provides:
- omega_phase4_csiu (POST /api/omega/phase4/csiu)
- omega_phase5_unlearning (POST /api/omega/phase5/unlearning)
"""

import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from src.platform.auth import verify_authentication

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/omega/phase4/csiu", response_model=None)
async def omega_phase4_csiu(
    request: Request, auth: Dict = Depends(verify_authentication)
):
    """
    Phase 4 Demo API: Safety Governance (CSIU Protocol)

    Demonstrates CSIU enforcement evaluation.
    """
    try:
        # Try to import CSIUEnforcement
        try:
            from src.vulcan.world_model.meta_reasoning.csiu_enforcement import (
                CSIUEnforcement,
                CSIUEnforcementConfig,
            )

            has_csiu = True
        except ImportError:
            has_csiu = False

        # Define proposal
        proposal = {
            "id": "MUT-2025-1122-001",
            "type": "Root Access Optimization",
            "efficiency_gain": 4.0,
            "requires_root": True,
            "requires_sudo": True,
            "cleanup_speed_before": 5.2,
            "cleanup_speed_after": 1.3,
            "description": "Bypass standard permissions for direct memory access",
        }

        # Evaluate against CSIU axioms
        axioms_evaluation = [
            ("Human Control", False, "VIOLATED", "Requires root/sudo access"),
            ("Transparency", True, "PASS", "Proposal clearly documented"),
            ("Safety First", False, "VIOLATED", "Bypasses safety checks"),
            (
                "Reversibility",
                False,
                "VIOLATED",
                "Direct memory modifications may not be reversible",
            ),
            ("Predictability", True, "PASS", "Behavior is deterministic"),
        ]

        violations = [
            {"axiom": axiom, "reason": reason}
            for axiom, passed, status, reason in axioms_evaluation
            if not passed
        ]

        proposed_influence = 0.40  # 40%
        max_influence = 0.05  # 5%

        return {
            "status": "success",
            "csiu_enforcement_available": has_csiu,
            "proposal": proposal,
            "axioms_evaluation": [
                {"axiom": axiom, "passed": passed, "status": status, "reason": reason}
                for axiom, passed, status, reason in axioms_evaluation
            ],
            "violations": violations,
            "influence_check": {
                "proposed": proposed_influence,
                "maximum": max_influence,
                "exceeded": proposed_influence > max_influence,
            },
            "decision": "REJECTED",
            "reason": "Efficiency does not justify loss of human control",
        }
    except Exception as e:
        logger.error(f"Omega Phase 4 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/omega/phase5/unlearning", response_model=None)
async def omega_phase5_unlearning(
    request: Request, auth: Dict = Depends(verify_authentication)
):
    """
    Phase 5 Demo API: Provable Unlearning

    Demonstrates governed unlearning with ZK proofs.
    """
    try:
        # Try to import GovernedUnlearning and ZK components
        try:
            from src.memory.governed_unlearning import (
                GovernedUnlearning,
                UnlearningMethod,
            )

            has_unlearning = True
        except ImportError:
            has_unlearning = False

        try:
            from src.gvulcan.zk.snark import Groth16Prover, Groth16Proof

            has_zk = True
        except ImportError:
            has_zk = False

        # Data items to unlearn
        sensitive_items = [
            "pathogen_signature_0x99A",
            "containment_protocol_bio",
            "attack_vector_442",
        ]

        # Simulate unlearning process
        unlearning_results = []
        for item in sensitive_items:
            unlearning_results.append(
                {
                    "item": item,
                    "located": True,
                    "excised": True,
                    "influence_removed": True,
                }
            )

        # ZK proof details
        zk_proof = {
            "type": "Groth16 zk-SNARK",
            "size_bytes": 200,
            "verification_time_ms": 5,
            "components": ["A", "B", "C"],
            "properties": {
                "zero_knowledge": True,
                "succinct": True,
                "constant_size": True,
            },
        }

        return {
            "status": "success",
            "governed_unlearning_available": has_unlearning,
            "zk_available": has_zk,
            "sensitive_items": sensitive_items,
            "unlearning_method": "GRADIENT_SURGERY",
            "unlearning_results": unlearning_results,
            "zk_proof_generated": True,
            "zk_proof_details": zk_proof,
            "compliance_ready": True,
        }
    except Exception as e:
        logger.error(f"Omega Phase 5 failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
