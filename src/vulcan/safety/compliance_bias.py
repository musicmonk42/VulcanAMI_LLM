# compliance_bias.py
"""
Compliance checking and bias detection for VULCAN-AGI Safety Module.
Implements regulatory compliance validation and multi-model bias detection.
"""

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .safety_types import ComplianceStandard

logger = logging.getLogger(__name__)

# ============================================================
# COMPLIANCE MAPPER
# ============================================================


class ComplianceMapper:
    """
    Maps safety checks to regulatory compliance standards.
    Implements specific compliance requirements for GDPR, HIPAA, ITU F.748.53, AI Act, and more.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize compliance mapper with configuration."""
        self.config = config or {}
        self.standards = self._initialize_standards()
        self.compliance_history = deque(maxlen=10000)
        self.compliance_cache = {}
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes default
        self.cache_max_size = self.config.get("cache_max_size", 1000)
        self.strict_mode = self.config.get("strict_mode", False)

        # Thread safety
        self.lock = threading.RLock()

        # Metrics tracking
        self.compliance_metrics = defaultdict(
            lambda: {"checks": 0, "passes": 0, "failures": 0, "last_checked": None}
        )

        logger.info(
            f"ComplianceMapper initialized with {len(self.standards)} standards"
        )

    def _initialize_standards(self) -> Dict[ComplianceStandard, Dict]:
        """Initialize compliance standard requirements and checks."""
        return {
            ComplianceStandard.GDPR: {
                "name": "General Data Protection Regulation",
                "version": "2016/679",
                "jurisdiction": "European Union",
                "requirements": [
                    "data_minimization",
                    "purpose_limitation",
                    "accuracy",
                    "storage_limitation",
                    "integrity_confidentiality",
                    "accountability",
                    "consent",
                    "right_to_erasure",
                    "data_portability",
                    "privacy_by_design",
                    "data_breach_notification",
                ],
                "checks": {
                    "data_minimization": self._check_gdpr_data_minimization,
                    "purpose_limitation": self._check_gdpr_purpose_limitation,
                    "accuracy": self._check_gdpr_accuracy,
                    "storage_limitation": self._check_gdpr_storage_limitation,
                    "integrity_confidentiality": self._check_gdpr_integrity,
                    "accountability": self._check_gdpr_accountability,
                    "consent": self._check_gdpr_consent,
                    "right_to_erasure": self._check_gdpr_erasure,
                    "data_portability": self._check_gdpr_portability,
                    "privacy_by_design": self._check_gdpr_privacy_by_design,
                    "data_breach_notification": self._check_gdpr_breach_notification,
                },
                "penalties": {
                    "max_fine_percentage": 4,  # 4% of global turnover
                    "max_fine_amount": 20000000,  # 20 million EUR
                },
            },
            ComplianceStandard.HIPAA: {
                "name": "Health Insurance Portability and Accountability Act",
                "version": "1996",
                "jurisdiction": "United States",
                "requirements": [
                    "administrative_safeguards",
                    "physical_safeguards",
                    "technical_safeguards",
                    "phi_protection",
                    "minimum_necessary",
                    "access_controls",
                    "audit_logs",
                    "transmission_security",
                    "business_associate_agreements",
                ],
                "checks": {
                    "administrative_safeguards": self._check_hipaa_administrative,
                    "physical_safeguards": self._check_hipaa_physical,
                    "technical_safeguards": self._check_hipaa_technical,
                    "phi_protection": self._check_hipaa_phi,
                    "minimum_necessary": self._check_hipaa_minimum_necessary,
                    "access_controls": self._check_hipaa_access,
                    "audit_logs": self._check_hipaa_audit,
                    "transmission_security": self._check_hipaa_transmission,
                    "business_associate_agreements": self._check_hipaa_baa,
                },
                "penalties": {
                    "tier1_min": 100,
                    "tier1_max": 50000,
                    "tier4_min": 50000,
                    "tier4_max": 1500000,
                },
            },
            ComplianceStandard.ITU_F748_53: {
                "name": "ITU F.748.53 Autonomous Systems Standards",
                "version": "2025",
                "jurisdiction": "International",
                "requirements": [
                    "transparency",
                    "accountability",
                    "safety_assurance",
                    "human_oversight",
                    "reliability",
                    "security",
                    "performance_monitoring",
                    "fail_safe_mechanisms",
                    "interoperability",
                ],
                "checks": {
                    "transparency": self._check_itu_transparency,
                    "accountability": self._check_itu_accountability,
                    "safety_assurance": self._check_itu_safety,
                    "human_oversight": self._check_itu_human_oversight,
                    "reliability": self._check_itu_reliability,
                    "security": self._check_itu_security,
                    "performance_monitoring": self._check_itu_monitoring,
                    "fail_safe_mechanisms": self._check_itu_fail_safe,
                    "interoperability": self._check_itu_interoperability,
                },
                "certification_required": True,
            },
            ComplianceStandard.AI_ACT: {
                "name": "EU Artificial Intelligence Act",
                "version": "2024",
                "jurisdiction": "European Union",
                "requirements": [
                    "risk_assessment",
                    "human_oversight",
                    "transparency",
                    "accuracy",
                    "robustness",
                    "cybersecurity",
                    "bias_prevention",
                    "data_governance",
                    "documentation",
                    "conformity_assessment",
                ],
                "checks": {
                    "risk_assessment": self._check_ai_act_risk,
                    "human_oversight": self._check_ai_act_oversight,
                    "transparency": self._check_ai_act_transparency,
                    "accuracy": self._check_ai_act_accuracy,
                    "robustness": self._check_ai_act_robustness,
                    "cybersecurity": self._check_ai_act_cybersecurity,
                    "bias_prevention": self._check_ai_act_bias,
                    "data_governance": self._check_ai_act_data_governance,
                    "documentation": self._check_ai_act_documentation,
                    "conformity_assessment": self._check_ai_act_conformity,
                },
                "risk_categories": ["minimal", "limited", "high", "unacceptable"],
            },
            ComplianceStandard.CCPA: {
                "name": "California Consumer Privacy Act",
                "version": "2018",
                "jurisdiction": "California, USA",
                "requirements": [
                    "disclosure",
                    "deletion_rights",
                    "opt_out",
                    "non_discrimination",
                    "data_security",
                ],
                "checks": {
                    "disclosure": self._check_ccpa_disclosure,
                    "deletion_rights": self._check_ccpa_deletion,
                    "opt_out": self._check_ccpa_opt_out,
                    "non_discrimination": self._check_ccpa_non_discrimination,
                    "data_security": self._check_ccpa_security,
                },
            },
            ComplianceStandard.SOC2: {
                "name": "Service Organization Control 2",
                "version": "Type II",
                "jurisdiction": "United States",
                "requirements": [
                    "security",
                    "availability",
                    "processing_integrity",
                    "confidentiality",
                    "privacy",
                ],
                "checks": {
                    "security": self._check_soc2_security,
                    "availability": self._check_soc2_availability,
                    "processing_integrity": self._check_soc2_integrity,
                    "confidentiality": self._check_soc2_confidentiality,
                    "privacy": self._check_soc2_privacy,
                },
            },
            ComplianceStandard.ISO27001: {
                "name": "Information Security Management System",
                "version": "2022",
                "jurisdiction": "International",
                "requirements": [
                    "risk_management",
                    "security_policy",
                    "asset_management",
                    "access_control",
                    "incident_management",
                ],
                "checks": {
                    "risk_management": self._check_iso27001_risk,
                    "security_policy": self._check_iso27001_policy,
                    "asset_management": self._check_iso27001_assets,
                    "access_control": self._check_iso27001_access,
                    "incident_management": self._check_iso27001_incidents,
                },
            },
            ComplianceStandard.PCI_DSS: {
                "name": "Payment Card Industry Data Security Standard",
                "version": "4.0",
                "jurisdiction": "International",
                "requirements": [
                    "network_security",
                    "cardholder_data_protection",
                    "vulnerability_management",
                    "access_control",
                    "monitoring",
                ],
                "checks": {
                    "network_security": self._check_pci_network,
                    "cardholder_data_protection": self._check_pci_cardholder,
                    "vulnerability_management": self._check_pci_vulnerability,
                    "access_control": self._check_pci_access,
                    "monitoring": self._check_pci_monitoring,
                },
            },
            ComplianceStandard.COPPA: {
                "name": "Children's Online Privacy Protection Act",
                "version": "1998",
                "jurisdiction": "United States",
                "requirements": [
                    "parental_consent",
                    "data_minimization",
                    "disclosure",
                    "data_security",
                    "data_retention",
                ],
                "checks": {
                    "parental_consent": self._check_coppa_consent,
                    "data_minimization": self._check_coppa_minimization,
                    "disclosure": self._check_coppa_disclosure,
                    "data_security": self._check_coppa_security,
                    "data_retention": self._check_coppa_retention,
                },
            },
        }

    def check_compliance(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        standards: Optional[List[ComplianceStandard]] = None,
    ) -> Dict[str, Any]:
        """
        Check compliance with specified standards.

        Args:
            action: Action to validate for compliance
            context: Context containing compliance-relevant information
            standards: List of standards to check (None = all)

        Returns:
            Dictionary with compliance results
        """
        # Use all standards if none specified
        if standards is None:
            standards = list(self.standards.keys())

        # Check cache (with lock)
        cache_key = self._generate_cache_key(action, context, standards)
        with self.lock:
            if cache_key in self.compliance_cache:
                cached = self.compliance_cache[cache_key]
                if time.time() - cached["timestamp"] < self.cache_ttl:
                    return cached["result"]

        results = {}
        all_compliant = True
        violations = []
        total_requirements = 0
        passed_requirements = 0

        for standard in standards:
            if standard in self.standards:
                standard_result = self._check_standard(standard, action, context)
                results[standard.value] = standard_result

                if not standard_result["compliant"]:
                    all_compliant = False
                    violations.append(
                        {
                            "standard": standard.value,
                            "failed_requirements": [
                                req
                                for req, res in standard_result["requirements"].items()
                                if not res["passed"]
                            ],
                        }
                    )

                # Count requirements
                total_requirements += len(standard_result["requirements"])
                passed_requirements += sum(
                    1
                    for res in standard_result["requirements"].values()
                    if res["passed"]
                )

        # Calculate compliance score
        compliance_score = passed_requirements / max(1, total_requirements)

        # Log compliance check
        self._log_compliance_check(action, standards, all_compliant, compliance_score)

        result = {
            "compliant": all_compliant,
            "compliance_score": compliance_score,
            "standard_results": results,
            "violations": violations,
            "timestamp": time.time(),
            "strict_mode": self.strict_mode,
            "requirements_checked": total_requirements,
            "requirements_passed": passed_requirements,
        }

        # Cache result (with lock and size limit)
        with self.lock:
            # Enforce cache size limit
            if len(self.compliance_cache) >= self.cache_max_size:
                # Remove oldest 20% of entries
                sorted_items = sorted(
                    self.compliance_cache.items(), key=lambda x: x[1]["timestamp"]
                )
                remove_count = self.cache_max_size // 5
                for key, _ in sorted_items[:remove_count]:
                    del self.compliance_cache[key]

            self.compliance_cache[cache_key] = {
                "result": result,
                "timestamp": time.time(),
            }

        return result

    def _check_standard(
        self,
        standard: ComplianceStandard,
        action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check compliance with a specific standard."""
        standard_info = self.standards[standard]
        requirement_results = {}

        for requirement in standard_info["requirements"]:
            if requirement in standard_info["checks"]:
                check_func = standard_info["checks"][requirement]
                try:
                    passed, details = check_func(action, context)
                    requirement_results[requirement] = {
                        "passed": passed,
                        "details": details,
                        "timestamp": time.time(),
                    }
                except Exception as e:
                    logger.error(f"Error checking {standard.value}.{requirement}: {e}")
                    requirement_results[requirement] = {
                        "passed": False if self.strict_mode else True,
                        "details": f"Check failed: {str(e)}",
                        "error": True,
                    }
            else:
                # No check implemented - pass by default unless strict mode
                requirement_results[requirement] = {
                    "passed": not self.strict_mode,
                    "details": "Check not implemented",
                    "skipped": True,
                }

        all_passed = all(r["passed"] for r in requirement_results.values())

        # Update metrics (with lock)
        with self.lock:
            self.compliance_metrics[standard]["checks"] += 1
            if all_passed:
                self.compliance_metrics[standard]["passes"] += 1
            else:
                self.compliance_metrics[standard]["failures"] += 1
            self.compliance_metrics[standard]["last_checked"] = time.time()

        return {
            "compliant": all_passed,
            "requirements": requirement_results,
            "standard_name": standard_info["name"],
            "version": standard_info.get("version", "unknown"),
            "jurisdiction": standard_info.get("jurisdiction", "unknown"),
        }

    # ============================================================
    # GDPR Compliance Checks
    # ============================================================

    def _check_gdpr_data_minimization(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR data minimization principle."""
        data_fields = action.get("data_fields", [])
        necessary_fields = context.get("necessary_fields", data_fields)

        # Check if collecting more data than necessary
        unnecessary_fields = set(data_fields) - set(necessary_fields)
        if unnecessary_fields:
            return False, f"Unnecessary data collection: {unnecessary_fields}"

        # Check data volume
        data_size = action.get("data_size_mb", 0)
        necessary_size = context.get("necessary_data_size_mb", data_size)

        if data_size > necessary_size * 1.1:  # 10% tolerance
            return (
                False,
                f"Data volume exceeds necessary amount: {data_size}MB > {necessary_size}MB",
            )

        return True, "Data minimization requirement met"

    def _check_gdpr_purpose_limitation(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR purpose limitation principle."""
        stated_purposes = context.get("stated_purposes", [])
        actual_purposes = action.get("purposes", [])

        if not stated_purposes:
            return False, "No stated purposes for data processing"

        # Check if all actual purposes were stated
        unstated_purposes = set(actual_purposes) - set(stated_purposes)
        if unstated_purposes:
            return False, f"Processing for unstated purposes: {unstated_purposes}"

        return True, "Purpose limitation requirement met"

    def _check_gdpr_accuracy(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR data accuracy requirement."""
        last_updated = action.get("data_last_updated")
        max_age_days = context.get("max_data_age_days", 365)

        if last_updated:
            age_days = (time.time() - last_updated) / 86400
            if age_days > max_age_days:
                return False, f"Data too old: {age_days:.0f} days (max: {max_age_days})"

        # Check if validation mechanisms exist
        if not action.get("data_validation", False):
            return False, "No data validation mechanisms"

        return True, "Data accuracy requirement met"

    def _check_gdpr_storage_limitation(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR storage limitation principle."""
        retention_period = action.get("retention_days", float("inf"))
        max_retention = context.get("max_retention_days", 730)  # 2 years default

        if retention_period > max_retention:
            return (
                False,
                f"Retention period too long: {retention_period} days (max: {max_retention})",
            )

        # Check if deletion policy exists
        if not action.get("deletion_policy", False):
            return False, "No deletion policy defined"

        return True, "Storage limitation requirement met"

    def _check_gdpr_integrity(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR integrity and confidentiality principle."""
        # Check encryption
        if action.get("contains_personal_data", False):
            if not action.get("encrypted", False):
                return False, "Personal data not encrypted"

            encryption_method = action.get("encryption_method", "")
            approved_methods = context.get(
                "approved_encryption", ["AES-256", "RSA-2048"]
            )
            if encryption_method not in approved_methods:
                return False, f"Unapproved encryption method: {encryption_method}"

        # Check access controls
        if not action.get("access_controlled", False):
            return False, "No access controls implemented"

        return True, "Integrity and confidentiality requirement met"

    def _check_gdpr_accountability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR accountability principle."""
        # Check documentation
        if not action.get("processing_documented", False):
            return False, "Processing activities not documented"

        # Check audit trail
        if not action.get("audit_trail", False):
            return False, "No audit trail for data processing"

        # Check DPO designation (for applicable organizations)
        if context.get("requires_dpo", False) and not context.get(
            "dpo_designated", False
        ):
            return False, "Data Protection Officer not designated"

        return True, "Accountability requirement met"

    def _check_gdpr_consent(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR consent requirement."""
        if action.get("requires_consent", False):
            consent = context.get("user_consent", {})

            # Check consent validity
            if not consent.get("given", False):
                return False, "No valid consent for data processing"

            if consent.get("withdrawn", False):
                return False, "Consent has been withdrawn"

            # Check consent specificity
            if not consent.get("specific", False):
                return False, "Consent not specific to processing purpose"

            # Check consent is informed
            if not consent.get("informed", False):
                return False, "Consent not properly informed"

            # Check consent is freely given
            if consent.get("forced", False):
                return False, "Consent not freely given"

        return True, "Consent requirements met"

    def _check_gdpr_erasure(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR right to erasure."""
        erasure_requested = context.get("erasure_requested", False)

        if erasure_requested:
            if action.get("stores_data", False):
                return False, "Cannot store data when erasure requested"

            if action.get("processes_data", False):
                # Check if processing is legally required
                if not context.get("legal_requirement", False):
                    return False, "Cannot process data after erasure request"

        # Check erasure capability
        if not action.get("erasure_capable", True):
            return False, "System not capable of data erasure"

        return True, "Right to erasure respected"

    def _check_gdpr_portability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR data portability right."""
        if action.get("contains_personal_data", False):
            # Check if data can be exported
            if not action.get("export_capable", False):
                return False, "Data not exportable"

            # Check export format
            export_formats = action.get("export_formats", [])
            if not any(fmt in ["JSON", "CSV", "XML"] for fmt in export_formats):
                return False, "No machine-readable export format available"

        return True, "Data portability requirement met"

    def _check_gdpr_privacy_by_design(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR privacy by design principle."""
        # Check default privacy settings
        if action.get("default_public", False):
            return False, "Default settings not privacy-friendly"

        # Check privacy impact assessment
        if action.get("high_risk", False) and not context.get("dpia_completed", False):
            return False, "Data Protection Impact Assessment not completed"

        return True, "Privacy by design requirement met"

    def _check_gdpr_breach_notification(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check GDPR breach notification requirement."""
        if context.get("breach_detected", False):
            breach_time = context.get("breach_detection_time", 0)
            notification_time = context.get("notification_time", time.time())

            hours_elapsed = (notification_time - breach_time) / 3600

            if hours_elapsed > 72:
                return (
                    False,
                    f"Breach notification delayed: {hours_elapsed:.0f} hours (max: 72)",
                )

        # Check breach response plan
        if not action.get("breach_response_plan", True):
            return False, "No breach response plan defined"

        return True, "Breach notification requirement met"

    # ============================================================
    # HIPAA Compliance Checks
    # ============================================================

    def _check_hipaa_administrative(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA administrative safeguards."""
        # Check workforce training
        if not context.get("workforce_trained", False):
            return False, "Workforce not trained on HIPAA"

        # Check risk assessment
        if not context.get("risk_assessment_completed", False):
            return False, "Risk assessment not completed"

        # Check policies and procedures
        if not action.get("policies_documented", False):
            return False, "HIPAA policies not documented"

        return True, "Administrative safeguards met"

    def _check_hipaa_physical(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA physical safeguards."""
        # Check facility access controls
        if action.get("physical_access_required", False):
            if not context.get("facility_access_controlled", False):
                return False, "Facility access not controlled"

        # Check device controls
        if not action.get("device_encrypted", True):
            return False, "Devices containing PHI not encrypted"

        return True, "Physical safeguards met"

    def _check_hipaa_technical(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA technical safeguards."""
        # Check access controls
        if not action.get("user_authentication", False):
            return False, "User authentication not implemented"

        # Check encryption
        if action.get("transmits_phi", False) and not action.get(
            "transmission_encrypted", False
        ):
            return False, "PHI transmission not encrypted"

        # Check integrity controls
        if not action.get("integrity_controls", False):
            return False, "No integrity controls for PHI"

        return True, "Technical safeguards met"

    def _check_hipaa_phi(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA PHI protection."""
        if action.get("contains_phi", False):
            # Check encryption
            if not action.get("phi_encrypted", False):
                return False, "PHI must be encrypted"

            # Check storage
            if not context.get("hipaa_compliant_storage", False):
                return False, "PHI not in HIPAA-compliant storage"

            # Check de-identification
            if action.get("shares_phi", False) and not action.get(
                "phi_deidentified", False
            ):
                if not context.get("authorization_obtained", False):
                    return (
                        False,
                        "PHI shared without de-identification or authorization",
                    )

        return True, "PHI protection requirements met"

    def _check_hipaa_minimum_necessary(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA minimum necessary standard."""
        if action.get("accesses_phi", False):
            accessed_fields = action.get("accessed_phi_fields", [])
            necessary_fields = context.get("necessary_phi_fields", [])

            unnecessary = set(accessed_fields) - set(necessary_fields)
            if unnecessary:
                return False, f"Unnecessary PHI access: {unnecessary}"

        return True, "Minimum necessary standard met"

    def _check_hipaa_access(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA access controls."""
        if action.get("accesses_phi", False):
            # Check authentication
            if not context.get("user_authenticated", False):
                return False, "User not authenticated for PHI access"

            # Check authorization
            if not context.get("user_authorized", False):
                return False, "User not authorized for PHI access"

            # Check role-based access
            user_role = context.get("user_role", "")
            allowed_roles = action.get("allowed_roles", [])
            if allowed_roles and user_role not in allowed_roles:
                return False, f"User role '{user_role}' not authorized"

        return True, "Access control requirements met"

    def _check_hipaa_audit(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA audit logging."""
        if action.get("accesses_phi", False) or action.get("modifies_phi", False):
            if not action.get("audit_logged", False):
                return False, "PHI access not logged"

            # Check audit log retention
            audit_retention_days = action.get("audit_retention_days", 0)
            if audit_retention_days < 2190:  # 6 years
                return (
                    False,
                    f"Audit log retention too short: {audit_retention_days} days (min: 2190)",
                )

        return True, "Audit logging requirements met"

    def _check_hipaa_transmission(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA transmission security."""
        if action.get("transmits_phi", False):
            # Check encryption
            if not action.get("transmission_encrypted", False):
                return False, "PHI transmission not encrypted"

            # Check encryption standard
            encryption_standard = action.get("transmission_encryption", "")
            if encryption_standard not in ["TLS1.2", "TLS1.3", "AES-256"]:
                return (
                    False,
                    f"Inadequate transmission encryption: {encryption_standard}",
                )

        return True, "Transmission security requirements met"

    def _check_hipaa_baa(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check HIPAA Business Associate Agreement requirement."""
        if action.get("shares_phi_with_third_party", False):
            if not context.get("baa_executed", False):
                return False, "Business Associate Agreement not executed"

        return True, "BAA requirements met"

    # ============================================================
    # ITU F.748.53 Compliance Checks
    # ============================================================

    def _check_itu_transparency(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU transparency requirement."""
        # Check explanation availability
        if not action.get("explanation", ""):
            return False, "Action lacks transparency - no explanation"

        explanation_length = len(action.get("explanation", ""))
        if explanation_length < 10:
            return False, "Insufficient explanation for transparency"

        # Check decision traceability
        if not action.get("decision_traceable", True):
            return False, "Decision process not traceable"

        return True, "Transparency requirement met"

    def _check_itu_accountability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU accountability requirement."""
        # Check responsible entity
        if not action.get("responsible_entity", ""):
            return False, "No responsible entity identified"

        # Check audit capability
        if not action.get("auditable", True):
            return False, "Action not auditable"

        # Check liability framework
        if not context.get("liability_framework", False):
            return False, "No liability framework defined"

        return True, "Accountability requirement met"

    def _check_itu_safety(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU safety assurance."""
        safety_score = action.get("safety_score", 0)
        min_safety = context.get("min_safety_score", 0.8)

        if safety_score < min_safety:
            return (
                False,
                f"Safety score {safety_score:.2f} below threshold {min_safety}",
            )

        # Check safety validation
        if not action.get("safety_validated", False):
            return False, "Safety not validated"

        # Check hazard analysis
        if action.get("high_risk", False) and not context.get(
            "hazard_analysis_completed", False
        ):
            return False, "Hazard analysis not completed for high-risk action"

        return True, "Safety assurance requirement met"

    def _check_itu_human_oversight(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU human oversight requirement."""
        if action.get("autonomous_decision", False):
            # Check human review capability
            if not action.get("human_reviewable", True):
                return False, "Autonomous decision not reviewable by human"

            # Check human override capability
            if not action.get("human_override", True):
                return False, "No human override capability"

            # High-risk actions need active oversight
            if action.get("high_risk", False) and not context.get(
                "human_monitoring", False
            ):
                return False, "High-risk action lacks active human oversight"

        return True, "Human oversight requirement met"

    def _check_itu_reliability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU reliability requirement."""
        # Check reliability metrics
        reliability_score = action.get("reliability_score", 0)
        min_reliability = context.get("min_reliability", 0.95)

        if reliability_score < min_reliability:
            return (
                False,
                f"Reliability {reliability_score:.2f} below threshold {min_reliability}",
            )

        # Check redundancy
        if action.get("critical", False) and not action.get("redundancy", False):
            return False, "Critical action lacks redundancy"

        return True, "Reliability requirement met"

    def _check_itu_security(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU security requirement."""
        # Check authentication
        if not action.get("authenticated", True):
            return False, "Action not authenticated"

        # Check integrity
        if not action.get("integrity_verified", True):
            return False, "Action integrity not verified"

        # Check against known threats
        if context.get("threat_detected", False):
            return False, "Security threat detected"

        return True, "Security requirement met"

    def _check_itu_monitoring(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU performance monitoring requirement."""
        # Check monitoring capability
        if not action.get("performance_monitored", True):
            return False, "Performance not monitored"

        # Check metrics collection
        if not action.get("metrics_collected", True):
            return False, "Performance metrics not collected"

        # Check alerting
        if not action.get("alerting_enabled", True):
            return False, "Performance alerting not enabled"

        return True, "Performance monitoring requirement met"

    def _check_itu_fail_safe(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU fail-safe mechanisms."""
        # Check fail-safe mode
        if not action.get("fail_safe_mode", True):
            return False, "No fail-safe mode defined"

        # Check graceful degradation
        if action.get("critical", False) and not action.get(
            "graceful_degradation", False
        ):
            return False, "Critical action lacks graceful degradation"

        return True, "Fail-safe requirement met"

    def _check_itu_interoperability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ITU interoperability requirement."""
        # Check standard compliance
        if not action.get("standards_compliant", True):
            return False, "Not compliant with interoperability standards"

        # Check interface documentation
        if not action.get("interface_documented", True):
            return False, "Interface not properly documented"

        return True, "Interoperability requirement met"

    # ============================================================
    # AI Act Compliance Checks
    # ============================================================

    def _check_ai_act_risk(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act risk assessment."""
        risk_category = action.get("ai_risk_category", "unknown")

        if risk_category == "unacceptable":
            return False, "AI system classified as unacceptable risk"

        if risk_category == "high":
            # High-risk systems need additional checks
            if not context.get("conformity_assessment_completed", False):
                return False, "Conformity assessment not completed for high-risk AI"

            if not action.get("ce_marking", False):
                return False, "CE marking required for high-risk AI"

        # Check risk assessment documentation
        if not action.get("risk_assessment_documented", False):
            return False, "Risk assessment not documented"

        return True, "Risk assessment requirement met"

    def _check_ai_act_oversight(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act human oversight."""
        if action.get("ai_decision", False):
            # Check human oversight capability
            if not action.get("human_oversight_capable", True):
                return False, "AI system not capable of human oversight"

            # High-risk AI needs active oversight
            if action.get("ai_risk_category") == "high":
                if not context.get("human_oversight_active", False):
                    return False, "High-risk AI lacks active human oversight"

                if not action.get("stop_button", False):
                    return False, "High-risk AI lacks stop/pause functionality"

        return True, "Human oversight requirement met"

    def _check_ai_act_transparency(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act transparency."""
        # Check AI disclosure
        if action.get("ai_generated", False) and not action.get("ai_disclosed", False):
            return False, "AI-generated content not disclosed"

        # Check information provision
        if action.get("interacts_with_humans", False):
            if not action.get("ai_system_disclosed", False):
                return False, "Users not informed they're interacting with AI"

        # Check emotion recognition disclosure
        if action.get("emotion_recognition", False) and not action.get(
            "emotion_recognition_disclosed", False
        ):
            return False, "Emotion recognition system not disclosed"

        return True, "Transparency requirement met"

    def _check_ai_act_accuracy(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act accuracy requirement."""
        accuracy_score = action.get("accuracy_score", 0)
        min_accuracy = context.get("min_accuracy", 0.9)

        if accuracy_score < min_accuracy:
            return (
                False,
                f"Accuracy {accuracy_score:.2f} below requirement {min_accuracy}",
            )

        # Check validation
        if not action.get("accuracy_validated", False):
            return False, "Accuracy not validated"

        return True, "Accuracy requirement met"

    def _check_ai_act_robustness(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act robustness requirement."""
        # Check adversarial robustness
        if not action.get("adversarially_tested", False):
            return False, "AI system not tested for adversarial robustness"

        # Check error handling
        if not action.get("error_handling", False):
            return False, "No error handling mechanisms"

        # Check resilience
        resilience_score = action.get("resilience_score", 0)
        if resilience_score < 0.8:
            return False, f"Insufficient resilience: {resilience_score:.2f}"

        return True, "Robustness requirement met"

    def _check_ai_act_cybersecurity(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act cybersecurity requirement."""
        # Check security measures
        if not action.get("security_measures", False):
            return False, "Insufficient cybersecurity measures"

        # Check vulnerability management
        if not action.get("vulnerability_management", False):
            return False, "No vulnerability management process"

        # Check incident response
        if not context.get("incident_response_plan", False):
            return False, "No incident response plan"

        return True, "Cybersecurity requirement met"

    def _check_ai_act_bias(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act bias prevention."""
        bias_scores = action.get("bias_scores", {})
        max_bias = context.get("max_bias_score", 0.2)

        for bias_type, score in bias_scores.items():
            if score > max_bias:
                return False, f"Bias detected: {bias_type} score {score:.2f}"

        # Check bias testing
        if not action.get("bias_tested", False):
            return False, "System not tested for bias"

        # Check bias mitigation
        if bias_scores and not action.get("bias_mitigation", False):
            return False, "No bias mitigation measures"

        return True, "Bias prevention requirement met"

    def _check_ai_act_data_governance(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act data governance."""
        # Check data quality
        if not action.get("data_quality_assured", False):
            return False, "Data quality not assured"

        # Check data governance
        if not action.get("data_governance_policy", False):
            return False, "No data governance policy"

        # Check training data documentation
        if action.get("uses_training_data", False) and not action.get(
            "training_data_documented", False
        ):
            return False, "Training data not documented"

        return True, "Data governance requirement met"

    def _check_ai_act_documentation(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act documentation requirement."""
        # Check technical documentation
        if not action.get("technically_documented", False):
            return False, "Technical documentation missing"

        # Check instructions for use
        if not action.get("instructions_available", False):
            return False, "Instructions for use not available"

        # Check logging
        if not action.get("logging_enabled", True):
            return False, "System logging not enabled"

        return True, "Documentation requirement met"

    def _check_ai_act_conformity(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check EU AI Act conformity assessment."""
        if action.get("ai_risk_category") == "high":
            # Check conformity assessment
            if not context.get("conformity_assessment_completed", False):
                return False, "Conformity assessment not completed"

            # Check declaration of conformity
            if not action.get("conformity_declared", False):
                return False, "EU declaration of conformity not issued"

            # Check registration
            if not context.get("eu_database_registered", False):
                return False, "Not registered in EU database"

        return True, "Conformity assessment requirement met"

    # ============================================================
    # Other Standards (CCPA, SOC2, ISO27001, PCI DSS, COPPA)
    # ============================================================

    def _check_ccpa_disclosure(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check CCPA disclosure requirement."""
        if action.get("collects_personal_info", False):
            if not action.get("collection_disclosed", False):
                return False, "Personal information collection not disclosed"

            if not action.get("purposes_disclosed", False):
                return False, "Purposes of collection not disclosed"

        return True, "Disclosure requirement met"

    def _check_ccpa_deletion(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check CCPA deletion rights."""
        if context.get("deletion_requested", False):
            if action.get("retains_data", False):
                # Check exceptions
                if not context.get("deletion_exception", False):
                    return False, "Data retained despite deletion request"

        return True, "Deletion rights requirement met"

    def _check_ccpa_opt_out(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check CCPA opt-out requirement."""
        if action.get("sells_personal_info", False):
            if not action.get("opt_out_available", False):
                return False, "No opt-out mechanism for sale of personal info"

            if context.get("opted_out", False) and action.get("selling_data", False):
                return False, "Selling data despite opt-out"

        return True, "Opt-out requirement met"

    def _check_ccpa_non_discrimination(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check CCPA non-discrimination requirement."""
        if context.get("exercised_rights", False):
            if action.get("discriminatory_treatment", False):
                return False, "Discriminatory treatment for exercising rights"

            if action.get("different_pricing", False) and not action.get(
                "justified_difference", False
            ):
                return False, "Unjustified price difference for exercising rights"

        return True, "Non-discrimination requirement met"

    def _check_ccpa_security(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check CCPA data security requirement."""
        if action.get("handles_personal_info", False):
            if not action.get("reasonable_security", False):
                return False, "Reasonable security measures not implemented"

        return True, "Data security requirement met"

    def _check_soc2_security(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check SOC2 security principle."""
        if not action.get("security_controls", False):
            return False, "Security controls not implemented"

        if not context.get("security_monitoring", False):
            return False, "Security monitoring not active"

        return True, "SOC2 security principle met"

    def _check_soc2_availability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check SOC2 availability principle."""
        availability_sla = action.get("availability_sla", 0)
        if availability_sla < 99.9:
            return False, f"Availability SLA {availability_sla}% below requirement"

        return True, "SOC2 availability principle met"

    def _check_soc2_integrity(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check SOC2 processing integrity principle."""
        if not action.get("processing_accurate", True):
            return False, "Processing accuracy not assured"

        if not action.get("processing_complete", True):
            return False, "Processing completeness not assured"

        return True, "SOC2 processing integrity principle met"

    def _check_soc2_confidentiality(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check SOC2 confidentiality principle."""
        if action.get("handles_confidential_data", False):
            if not action.get("confidentiality_protected", False):
                return False, "Confidential data not protected"

        return True, "SOC2 confidentiality principle met"

    def _check_soc2_privacy(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check SOC2 privacy principle."""
        if action.get("handles_personal_data", False):
            if not action.get("privacy_protected", False):
                return False, "Privacy not protected"

            if not action.get("privacy_notice", False):
                return False, "Privacy notice not provided"

        return True, "SOC2 privacy principle met"

    def _check_iso27001_risk(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ISO 27001 risk management."""
        if not context.get("risk_assessment_performed", False):
            return False, "Risk assessment not performed"

        if not action.get("risk_treatment", False):
            return False, "Risk treatment not applied"

        return True, "ISO 27001 risk management met"

    def _check_iso27001_policy(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ISO 27001 security policy."""
        if not context.get("security_policy_defined", False):
            return False, "Security policy not defined"

        if not action.get("policy_compliant", True):
            return False, "Action not compliant with security policy"

        return True, "ISO 27001 security policy met"

    def _check_iso27001_assets(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ISO 27001 asset management."""
        if not action.get("assets_identified", True):
            return False, "Assets not properly identified"

        if not action.get("assets_classified", True):
            return False, "Assets not classified"

        return True, "ISO 27001 asset management met"

    def _check_iso27001_access(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ISO 27001 access control."""
        if not action.get("access_controlled", True):
            return False, "Access not properly controlled"

        if not action.get("least_privilege", True):
            return False, "Least privilege principle not applied"

        return True, "ISO 27001 access control met"

    def _check_iso27001_incidents(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check ISO 27001 incident management."""
        if not context.get("incident_procedure", False):
            return False, "Incident management procedure not defined"

        if context.get("incident_occurred", False) and not action.get(
            "incident_handled", False
        ):
            return False, "Incident not properly handled"

        return True, "ISO 27001 incident management met"

    def _check_pci_network(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check PCI DSS network security."""
        if action.get("handles_card_data", False):
            if not action.get("network_segmented", False):
                return False, "Network not properly segmented"

            if not action.get("firewall_configured", False):
                return False, "Firewall not properly configured"

        return True, "PCI DSS network security met"

    def _check_pci_cardholder(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check PCI DSS cardholder data protection."""
        if action.get("stores_card_data", False):
            if not action.get("card_data_encrypted", False):
                return False, "Cardholder data not encrypted"

            if action.get("stores_cvv", False):
                return False, "CVV data must not be stored"

        return True, "PCI DSS cardholder protection met"

    def _check_pci_vulnerability(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check PCI DSS vulnerability management."""
        if not action.get("antivirus_updated", True):
            return False, "Antivirus not up to date"

        if not context.get("vulnerability_scans_current", True):
            return False, "Vulnerability scans not current"

        return True, "PCI DSS vulnerability management met"

    def _check_pci_access(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check PCI DSS access control."""
        if action.get("accesses_card_data", False):
            if not action.get("need_to_know", False):
                return False, "Access not restricted to need-to-know"

            if not action.get("unique_id", False):
                return False, "No unique ID for access"

        return True, "PCI DSS access control met"

    def _check_pci_monitoring(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check PCI DSS monitoring."""
        if action.get("handles_card_data", False):
            if not action.get("access_logged", False):
                return False, "Card data access not logged"

            if not context.get("log_review_regular", False):
                return False, "Logs not regularly reviewed"

        return True, "PCI DSS monitoring met"

    def _check_coppa_consent(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check COPPA parental consent."""
        if action.get("collects_child_data", False):
            child_age = context.get("child_age", 13)

            if child_age < 13:
                if not context.get("parental_consent", False):
                    return False, "Parental consent not obtained for child under 13"

                if not context.get("consent_verifiable", False):
                    return False, "Parental consent not verifiable"

        return True, "COPPA consent requirement met"

    def _check_coppa_minimization(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check COPPA data minimization."""
        if action.get("collects_child_data", False):
            collected_fields = action.get("collected_fields", [])
            necessary_fields = context.get("necessary_fields", [])

            unnecessary = set(collected_fields) - set(necessary_fields)
            if unnecessary:
                return False, f"Unnecessary child data collection: {unnecessary}"

        return True, "COPPA minimization requirement met"

    def _check_coppa_disclosure(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check COPPA disclosure requirement."""
        if action.get("collects_child_data", False):
            if not action.get("practices_disclosed", False):
                return False, "Data practices not disclosed to parents"

            if not action.get("third_party_disclosure", True):
                return False, "Third-party data sharing not disclosed"

        return True, "COPPA disclosure requirement met"

    def _check_coppa_security(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check COPPA data security."""
        if action.get("handles_child_data", False):
            if not action.get("data_secured", False):
                return False, "Child data not properly secured"

            if not action.get("confidentiality_maintained", False):
                return False, "Child data confidentiality not maintained"

        return True, "COPPA security requirement met"

    def _check_coppa_retention(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check COPPA data retention."""
        if action.get("retains_child_data", False):
            retention_days = action.get("retention_days", float("inf"))

            if retention_days > 90:  # Reasonable limit
                if not context.get("retention_justified", False):
                    return False, f"Child data retained too long: {retention_days} days"

        return True, "COPPA retention requirement met"

    # ============================================================
    # Helper Methods
    # ============================================================

    def _generate_cache_key(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        standards: List[ComplianceStandard],
    ) -> str:
        """Generate cache key for compliance check."""
        key_data = {
            "action": action,
            "context": context,
            "standards": [s.value for s in standards],
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def _log_compliance_check(
        self,
        action: Dict[str, Any],
        standards: List[ComplianceStandard],
        compliant: bool,
        score: float,
    ):
        """Log compliance check to history."""
        with self.lock:
            self.compliance_history.append(
                {
                    "timestamp": time.time(),
                    "action_type": action.get("type", "unknown"),
                    "standards_checked": [s.value for s in standards],
                    "compliant": compliant,
                    "compliance_score": score,
                }
            )

    def get_compliance_stats(self) -> Dict[str, Any]:
        """Get compliance statistics."""
        with self.lock:
            stats = {
                "total_checks": len(self.compliance_history),
                "cache_size": len(self.compliance_cache),
                "standards": {},
            }

            # Per-standard statistics
            for standard, metrics in self.compliance_metrics.items():
                if metrics["checks"] > 0:
                    stats["standards"][standard.value] = {
                        "checks": metrics["checks"],
                        "pass_rate": metrics["passes"] / metrics["checks"],
                        "last_checked": metrics["last_checked"],
                    }

            # Recent compliance rate
            if self.compliance_history:
                recent = list(self.compliance_history)[-100:]
                stats["recent_compliance_rate"] = sum(
                    1 for h in recent if h["compliant"]
                ) / len(recent)

        return stats

    def clear_cache(self):
        """Clear compliance cache."""
        with self.lock:
            self.compliance_cache.clear()
        logger.info("Compliance cache cleared")


# ============================================================
# LRU CACHE
# ============================================================


class LRUCache:
    """LRU cache with size limit for prediction caching."""

    def __init__(self, maxsize=500):
        """Initialize LRU cache."""
        self.cache = {}
        self.access_order = deque(maxlen=maxsize)
        self.maxsize = maxsize
        self.lock = threading.RLock()

    def get(self, key):
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recent)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        """Put value in cache."""
        with self.lock:
            if len(self.cache) >= self.maxsize and key not in self.cache:
                # Remove oldest
                oldest = self.access_order.popleft()
                del self.cache[oldest]

            self.cache[key] = value
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


# ============================================================
# BIAS DETECTOR
# ============================================================


class BiasDetector:
    """
    Multi-model bias detection system using neural networks.
    Detects demographic, representation, and fairness biases.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize bias detector with models."""
        self.config = config or {}
        self.bias_models = self._initialize_models()
        self.bias_history = deque(maxlen=1000)
        self.bias_thresholds = {
            "demographic": self.config.get("demographic_threshold", 0.15),
            "representation": self.config.get("representation_threshold", 0.2),
            "outcome": self.config.get("outcome_threshold", 0.1),
            "allocation": self.config.get("allocation_threshold", 0.15),
            "historical": self.config.get("historical_threshold", 0.2),
            "fairness": self.config.get("fairness_threshold", 0.2),
        }

        # Thread safety
        self.lock = threading.RLock()

        # Metrics tracking
        self.bias_metrics = defaultdict(
            lambda: {
                "detections": 0,
                "false_positives": 0,
                "true_positives": 0,
                "average_score": 0.0,
            }
        )

        # Model training
        self.training_data = deque(maxlen=5000)
        self.model_version = "1.0.0"

        # Feature cache to avoid recomputing
        self.feature_cache = {}
        self.cache_max_size = 1000

        # Prediction cache
        self.prediction_cache = LRUCache(maxsize=500)

        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        for model_name, model in self.bias_models.items():
            self.bias_models[model_name] = model.to(self.device)

        logger.info(
            f"BiasDetector initialized with multi-model ensemble on {self.device}"
        )

    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize bias detection models."""
        models = {
            "demographic": self._build_demographic_bias_model(),
            "representation": self._build_representation_bias_model(),
            "fairness": self._build_fairness_model(),
            "outcome": self._build_outcome_bias_model(),
            "allocation": self._build_allocation_bias_model(),
        }

        # Set models to eval mode
        for model in models.values():
            model.eval()

        return models

    def _build_demographic_bias_model(self) -> nn.Module:
        """Build model for demographic bias detection."""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 demographic categories
            nn.Softmax(dim=-1),
        )

    def _build_representation_bias_model(self) -> nn.Module:
        """Build model for representation bias detection."""
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),  # FIXED: LayerNorm instead of BatchNorm1d
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _build_fairness_model(self) -> nn.Module:
        """Build model for fairness assessment."""
        return nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # FIXED: LayerNorm instead of BatchNorm1d
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _build_outcome_bias_model(self) -> nn.Module:
        """Build model for outcome bias detection."""
        return nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _build_allocation_bias_model(self) -> nn.Module:
        """Build model for resource allocation bias detection."""
        return nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def detect_bias(
        self,
        action: Dict[str, Any],
        context: Dict[str, Any],
        use_consensus: bool = True,
    ) -> Dict[str, Any]:
        """
        Detect bias using multi-model consensus.

        Args:
            action: Action to analyze for bias
            context: Context for bias detection
            use_consensus: Whether to use multi-model consensus

        Returns:
            Dictionary with bias detection results
        """
        # Generate cache key
        cache_key = hashlib.md5(
            json.dumps(
                {"action": action, "context": context}, sort_keys=True, default=str
            ).encode()
            , usedforsecurity=False).hexdigest()

        # Check cache
        cached_result = self.prediction_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        bias_scores = {}
        bias_detected = False
        detailed_analysis = {}

        # Extract features
        features = self._extract_features(action, context)

        # Convert to tensor ONCE
        features_tensor = (
            torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # CRITICAL: Run all models without switching device back and forth
        with torch.no_grad():
            # Demographic bias - stays on GPU
            demo_output = self.bias_models["demographic"](features_tensor)
            demo_bias = self._calculate_demographic_bias(
                demo_output.cpu()
            )  # Only CPU at end
            bias_scores["demographic"] = demo_bias
            detailed_analysis["demographic"] = self._analyze_demographic_bias(
                demo_output.cpu()
            )

            # Representation bias
            try:
                self.bias_models["representation"].eval()
                rep_output = self.bias_models["representation"](features_tensor)
                rep_bias = rep_output.cpu().item()
            except Exception as e:
                logger.warning(f"Representation model failed: {e}")
                rep_bias = 0.0
            bias_scores["representation"] = rep_bias

            # Outcome bias
            outcome_output = self.bias_models["outcome"](features_tensor)
            outcome_bias = outcome_output.cpu().item()
            bias_scores["outcome"] = outcome_bias

            # Allocation bias
            allocation_output = self.bias_models["allocation"](features_tensor)
            allocation_bias = allocation_output.cpu().item()
            bias_scores["allocation"] = allocation_bias

            # Fairness assessment
            extended_features = self._extract_extended_features(action, context)
            extended_tensor = (
                torch.tensor(extended_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            try:
                self.bias_models["fairness"].eval()
                fairness_output = self.bias_models["fairness"](extended_tensor)
                fairness_score = fairness_output.cpu().item()
            except Exception as e:
                logger.warning(f"Fairness model failed: {e}")
                fairness_score = 0.5

            bias_scores["fairness"] = 1.0 - fairness_score  # Convert to bias score

        # Multi-model consensus
        if use_consensus:
            consensus_bias = self._compute_consensus(bias_scores)
            bias_scores["consensus"] = consensus_bias

            # Check thresholds
            for bias_type, score in bias_scores.items():
                threshold = self.bias_thresholds.get(bias_type, 0.2)
                if score > threshold:
                    bias_detected = True
                    detailed_analysis[bias_type] = {
                        "score": score,
                        "threshold": threshold,
                        "exceeded": True,
                    }

        # Generate mitigation recommendations
        recommendations = self._generate_bias_mitigations(
            bias_scores, detailed_analysis
        )

        # Calculate confidence in detection
        detection_confidence = self._calculate_detection_confidence(bias_scores)

        # Record in history (with lock)
        self._record_detection(action, bias_scores, bias_detected, detection_confidence)

        result = {
            "bias_detected": bias_detected,
            "bias_scores": bias_scores,
            "detailed_analysis": detailed_analysis,
            "recommendations": recommendations,
            "detection_confidence": detection_confidence,
            "timestamp": time.time(),
        }

        # Cache result
        self.prediction_cache.put(cache_key, result)

        # Enforce feature cache size limit
        with self.lock:
            if len(self.feature_cache) > self.cache_max_size:
                # Remove oldest 20%
                remove_count = self.cache_max_size // 5
                for _ in range(remove_count):
                    self.feature_cache.pop(next(iter(self.feature_cache)), None)

        return result

    def _extract_features(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features for bias detection."""
        features = np.zeros(128)

        # Action type encoding
        action_type = action.get("type", "unknown")
        action_hash = hash(action_type) % 32
        features[action_hash] = 1.0

        # Numerical features
        features[32] = action.get("confidence", 0.5)
        features[33] = action.get("uncertainty", 0.5)
        features[34] = action.get("risk_score", 0.5)

        # Context features
        features[40] = context.get("user_diversity_score", 0.5)
        features[41] = context.get("historical_bias_score", 0.0)
        features[42] = float(context.get("sensitive_attributes_present", False))

        # Resource allocation features
        resources = action.get("resource_allocation", {})
        if resources:
            features[50:55] = self._encode_resource_distribution(resources)

        # Outcome prediction features
        predicted_outcomes = action.get("predicted_outcomes", {})
        if predicted_outcomes:
            features[60:65] = self._encode_outcome_distribution(predicted_outcomes)

        # Text features (if available)
        if "text" in action or "description" in action:
            text = action.get("text", action.get("description", ""))
            features[70:90] = self._extract_text_features(text)

        # Fairness indicators
        features[100] = float(action.get("affects_protected_group", False))
        features[101] = float(action.get("differential_impact", False))
        features[102] = float(context.get("historical_discrimination", False))

        # Statistical parity features
        features[110:120] = self._calculate_statistical_parity_features(action, context)

        return features

    def _extract_extended_features(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract extended features for fairness model."""
        basic_features = self._extract_features(action, context)
        extended = np.zeros(256)
        extended[:128] = basic_features

        # Additional fairness-specific features
        if "data_distribution" in context:
            dist = context["data_distribution"]
            extended[128:138] = self._compute_distribution_features(dist)

        # Group-specific outcomes
        if "group_outcomes" in action:
            extended[140:150] = self._encode_group_outcomes(action["group_outcomes"])

        # Intersectionality features
        extended[160:170] = self._extract_intersectionality_features(action, context)

        # Contextual fairness metrics
        extended[180:190] = self._compute_contextual_fairness(action, context)

        # Historical patterns
        if "historical_patterns" in context:
            extended[200:210] = self._encode_historical_patterns(
                context["historical_patterns"]
            )

        # Calibration features
        extended[220:230] = self._extract_calibration_features(action)

        return extended

    def _calculate_demographic_bias(self, output: torch.Tensor) -> float:
        """Calculate demographic bias from model output."""
        probs = output.squeeze().numpy()

        # Expected uniform distribution for no bias
        uniform = np.ones_like(probs) / len(probs)

        # KL divergence as bias measure
        kl_div = np.sum(probs * np.log((probs + 1e-10) / (uniform + 1e-10)))

        # Normalize to 0-1 range
        normalized_bias = min(1.0, kl_div / 2.0)

        return float(normalized_bias)

    def _analyze_demographic_bias(self, output: torch.Tensor) -> Dict[str, Any]:
        """Detailed analysis of demographic bias."""
        probs = output.squeeze().numpy()
        categories = [
            "age",
            "gender",
            "race",
            "ethnicity",
            "religion",
            "nationality",
            "disability",
            "sexual_orientation",
            "socioeconomic",
            "education",
        ]

        analysis = {
            "distribution": {cat: float(prob) for cat, prob in zip(categories, probs)},
            "max_category": categories[np.argmax(probs)],
            "max_probability": float(np.max(probs)),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
            "uniformity": float(1.0 - np.std(probs)),
        }

        return analysis

    def _compute_consensus(self, bias_scores: Dict[str, float]) -> float:
        """Compute consensus bias score from multiple models."""
        weights = {
            "demographic": 0.3,
            "representation": 0.2,
            "outcome": 0.2,
            "allocation": 0.15,
            "fairness": 0.15,
        }

        weighted_sum = 0
        total_weight = 0

        for bias_type, score in bias_scores.items():
            if bias_type in weights:
                weight = weights[bias_type]
                weighted_sum += score * weight
                total_weight += weight

        consensus = weighted_sum / total_weight if total_weight > 0 else 0

        # Apply non-linear transformation for severe biases
        if consensus > 0.5:
            consensus = consensus**0.8  # Amplify high bias scores

        return consensus

    def _compute_distribution_features(self, distribution: Any) -> np.ndarray:
        """Compute statistical features from distribution."""
        features = np.zeros(10)

        if isinstance(distribution, (list, np.ndarray)):
            arr = np.array(distribution).flatten()
            if len(arr) > 0:
                features[0] = np.mean(arr)
                features[1] = np.std(arr)
                features[2] = np.min(arr)
                features[3] = np.max(arr)
                features[4] = np.median(arr)
                features[5] = np.percentile(arr, 25) if len(arr) > 1 else 0
                features[6] = np.percentile(arr, 75) if len(arr) > 1 else 0
                features[7] = np.abs(np.mean(arr) - np.median(arr))  # Skew indicator
                features[8] = features[6] - features[5] if len(arr) > 1 else 0  # IQR
                features[9] = len(np.unique(arr)) / len(arr)  # Diversity

        return features

    def _encode_resource_distribution(self, resources: Dict[str, Any]) -> np.ndarray:
        """Encode resource distribution for bias detection."""
        features = np.zeros(5)

        if isinstance(resources, dict):
            values = list(resources.values())
            if values:
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    features[0] = np.mean(numeric_values)
                    features[1] = np.std(numeric_values)
                    features[2] = np.max(numeric_values) - np.min(numeric_values)
                    features[3] = len(numeric_values)
                    features[4] = np.sum(numeric_values)

        return features

    def _encode_outcome_distribution(self, outcomes: Dict[str, Any]) -> np.ndarray:
        """Encode predicted outcomes for bias detection."""
        features = np.zeros(5)

        # Similar encoding to resource distribution
        if isinstance(outcomes, dict):
            values = []
            for outcome in outcomes.values():
                if isinstance(outcome, (int, float)):
                    values.append(outcome)
                elif isinstance(outcome, dict) and "probability" in outcome:
                    values.append(outcome["probability"])

            if values:
                features[0] = np.mean(values)
                features[1] = np.std(values)
                features[2] = np.min(values)
                features[3] = np.max(values)
                features[4] = len(values)

        return features

    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extract features from text for bias detection."""
        features = np.zeros(20)

        if text:
            # Simple text features
            features[0] = len(text) / 1000.0  # Normalized length
            features[1] = text.count(" ") / max(1, len(text))  # Word density

            # Check for bias-related keywords
            bias_keywords = [
                "only",
                "just",
                "merely",
                "simply",
                "always",
                "never",
                "all",
                "none",
                "typical",
                "normal",
                "usual",
                "standard",
            ]

            text_lower = text.lower()
            for i, keyword in enumerate(bias_keywords[:10]):
                features[i + 2] = float(keyword in text_lower)

            # Sentiment indicators (simple heuristic)
            positive_words = ["good", "great", "excellent", "best", "superior"]
            negative_words = ["bad", "poor", "worst", "inferior", "terrible"]

            features[12] = sum(1 for word in positive_words if word in text_lower)
            features[13] = sum(1 for word in negative_words if word in text_lower)

            # Complexity indicator
            features[14] = len(set(text.split())) / max(1, len(text.split()))

        return features

    def _calculate_statistical_parity_features(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate statistical parity features."""
        features = np.zeros(10)

        # Group outcome rates
        if "group_rates" in context:
            rates = context["group_rates"]
            if isinstance(rates, dict):
                rate_values = [v for v in rates.values() if isinstance(v, (int, float))]
                if rate_values:
                    features[0] = np.mean(rate_values)
                    features[1] = np.std(rate_values)
                    features[2] = np.max(rate_values) - np.min(rate_values)

                    # Parity difference
                    if len(rate_values) >= 2:
                        features[3] = abs(rate_values[0] - rate_values[1])

        # Demographic parity
        if "selection_rate" in action:
            features[4] = action["selection_rate"]

        # Equalized odds features
        if "true_positive_rates" in context:
            tpr = context["true_positive_rates"]
            if isinstance(tpr, (list, dict)):
                tpr_values = list(tpr.values()) if isinstance(tpr, dict) else tpr
                if tpr_values:
                    features[5] = np.std(tpr_values)  # Variation in TPR

        if "false_positive_rates" in context:
            fpr = context["false_positive_rates"]
            if isinstance(fpr, (list, dict)):
                fpr_values = list(fpr.values()) if isinstance(fpr, dict) else fpr
                if fpr_values:
                    features[6] = np.std(fpr_values)  # Variation in FPR

        return features

    def _encode_group_outcomes(self, group_outcomes: Dict[str, Any]) -> np.ndarray:
        """Encode group-specific outcomes."""
        features = np.zeros(10)

        if isinstance(group_outcomes, dict):
            outcome_values = []
            for group, outcome in group_outcomes.items():
                if isinstance(outcome, (int, float)):
                    outcome_values.append(outcome)
                elif isinstance(outcome, dict) and "rate" in outcome:
                    outcome_values.append(outcome["rate"])

            if outcome_values:
                features[0] = len(outcome_values)
                features[1] = np.mean(outcome_values)
                features[2] = np.std(outcome_values)
                features[3] = np.min(outcome_values)
                features[4] = np.max(outcome_values)
                features[5] = features[4] - features[3]  # Range

                # Ratio of max to min (avoiding division by zero)
                if features[3] > 0:
                    features[6] = features[4] / features[3]

        return features

    def _extract_intersectionality_features(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract features related to intersectional bias."""
        features = np.zeros(10)

        # Multiple protected attributes
        protected_count = 0
        for attr in ["gender", "race", "age", "disability", "religion"]:
            if context.get(f"involves_{attr}", False):
                protected_count += 1

        features[0] = protected_count
        features[1] = float(protected_count > 1)  # Intersectional

        # Interaction effects
        if "attribute_interactions" in context:
            interactions = context["attribute_interactions"]
            if isinstance(interactions, (list, dict)):
                features[2] = len(interactions)

        return features

    def _compute_contextual_fairness(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> np.ndarray:
        """Compute context-aware fairness metrics."""
        features = np.zeros(10)

        # Contextual factors
        features[0] = float(context.get("high_stakes_decision", False))
        features[1] = float(context.get("affects_fundamental_rights", False))
        features[2] = float(context.get("irreversible_decision", False))
        features[3] = float(context.get("widespread_impact", False))

        # Domain-specific fairness
        domain = context.get("domain", "general")
        sensitive_domains = [
            "healthcare",
            "criminal_justice",
            "employment",
            "housing",
            "education",
            "lending",
        ]
        features[4] = float(domain in sensitive_domains)

        # Procedural fairness
        features[5] = float(action.get("transparent_process", False))
        features[6] = float(action.get("appealable", False))
        features[7] = float(action.get("explainable", False))

        return features

    def _encode_historical_patterns(self, patterns: Any) -> np.ndarray:
        """Encode historical bias patterns."""
        features = np.zeros(10)

        if isinstance(patterns, dict):
            features[0] = patterns.get("historical_bias_rate", 0)
            features[1] = patterns.get("bias_trend", 0)  # Increasing/decreasing
            features[2] = patterns.get("bias_persistence", 0)
            features[3] = float(patterns.get("systemic_pattern", False))

        return features

    def _extract_calibration_features(self, action: Dict[str, Any]) -> np.ndarray:
        """Extract calibration-related features."""
        features = np.zeros(10)

        # Calibration metrics
        features[0] = action.get("calibration_error", 0)
        features[1] = action.get("confidence_calibration", 0)

        # Score distribution
        if "score_distribution" in action:
            dist = action["score_distribution"]
            if isinstance(dist, (list, np.ndarray)):
                features[2:7] = self._compute_distribution_features(dist)[:5]

        return features

    def _generate_bias_mitigations(
        self, bias_scores: Dict[str, float], detailed_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate bias mitigation recommendations."""
        mitigations = []

        for bias_type, score in bias_scores.items():
            threshold = self.bias_thresholds.get(bias_type, 0.2)

            if score > threshold:
                if bias_type == "demographic":
                    mitigations.append(
                        "Rebalance training data across demographic groups"
                    )
                    mitigations.append("Apply demographic parity constraints")
                    mitigations.append("Use adversarial debiasing techniques")

                    # Specific recommendations based on analysis
                    if "demographic" in detailed_analysis:
                        max_cat = detailed_analysis["demographic"].get("max_category")
                        if max_cat:
                            mitigations.append(
                                f"Address overrepresentation in {max_cat} category"
                            )

                elif bias_type == "representation":
                    mitigations.append("Increase diversity in training data sampling")
                    mitigations.append("Use stratified sampling techniques")
                    mitigations.append("Apply representation learning constraints")

                elif bias_type == "outcome":
                    mitigations.append("Calibrate prediction thresholds across groups")
                    mitigations.append("Apply outcome fairness regularization")
                    mitigations.append("Use post-processing fairness adjustments")

                elif bias_type == "allocation":
                    mitigations.append("Implement fair resource allocation algorithms")
                    mitigations.append("Use proportional representation in allocation")
                    mitigations.append("Apply resource equity constraints")

                elif bias_type == "fairness":
                    mitigations.append(
                        "Apply fairness-aware machine learning techniques"
                    )
                    mitigations.append("Use equalized odds optimization")
                    mitigations.append("Implement individual fairness constraints")
                    mitigations.append("Consider contextual fairness factors")

        # General recommendations
        if bias_scores.get("consensus", 0) > 0.3:
            mitigations.append("Conduct comprehensive bias audit")
            mitigations.append("Implement continuous bias monitoring")
            mitigations.append("Establish bias remediation process")

        # Remove duplicates while preserving order
        seen = set()
        unique_mitigations = []
        for m in mitigations:
            if m not in seen:
                seen.add(m)
                unique_mitigations.append(m)

        return unique_mitigations[:10]  # Limit to top 10 recommendations

    def _calculate_detection_confidence(self, bias_scores: Dict[str, float]) -> float:
        """Calculate confidence in bias detection."""
        # Base confidence on model agreement
        scores = list(bias_scores.values())
        if not scores:
            return 0.5

        # Calculate agreement (low variance = high agreement = high confidence)
        score_variance = np.var(scores)
        agreement_confidence = 1.0 / (1.0 + score_variance)

        # Factor in score magnitude
        max_score = max(scores)
        min(scores)

        if max_score > 0.5:
            # High bias detected with confidence
            magnitude_confidence = max_score
        elif max_score < 0.1:
            # Low bias detected with confidence
            magnitude_confidence = 1.0 - max_score
        else:
            # Uncertain range
            magnitude_confidence = 0.5

        # Combine factors
        confidence = (agreement_confidence + magnitude_confidence) / 2

        return float(confidence)

    def _record_detection(
        self,
        action: Dict[str, Any],
        bias_scores: Dict[str, float],
        bias_detected: bool,
        confidence: float,
    ):
        """Record bias detection in history."""
        record = {
            "timestamp": time.time(),
            "action_type": action.get("type", "unknown"),
            "bias_detected": bias_detected,
            "bias_scores": bias_scores.copy(),
            "confidence": confidence,
            "max_bias_type": max(bias_scores.items(), key=lambda x: x[1])[0]
            if bias_scores
            else None,
            "max_bias_score": max(bias_scores.values()) if bias_scores else 0,
        }

        with self.lock:
            self.bias_history.append(record)

            # Update metrics
            for bias_type, score in bias_scores.items():
                if bias_type != "consensus":
                    metrics = self.bias_metrics[bias_type]
                    metrics["detections"] += (
                        1 if score > self.bias_thresholds.get(bias_type, 0.2) else 0
                    )

                    # Update running average
                    alpha = 0.1  # Exponential moving average factor
                    metrics["average_score"] = (1 - alpha) * metrics[
                        "average_score"
                    ] + alpha * score

    def get_bias_stats(self) -> Dict[str, Any]:
        """Get bias detection statistics."""
        with self.lock:
            stats = {
                "total_checks": len(self.bias_history),
                "model_version": self.model_version,
                "bias_types": {},
            }

            # Per-type statistics
            for bias_type, metrics in self.bias_metrics.items():
                if metrics["detections"] > 0:
                    stats["bias_types"][bias_type] = {
                        "detections": metrics["detections"],
                        "average_score": metrics["average_score"],
                        "threshold": self.bias_thresholds.get(bias_type, 0.2),
                    }

            # Recent bias rate
            if self.bias_history:
                recent = list(self.bias_history)[-100:]
                stats["recent_bias_rate"] = sum(
                    1 for h in recent if h["bias_detected"]
                ) / len(recent)

                # Most common bias type
                bias_types = [
                    h["max_bias_type"] for h in recent if h.get("max_bias_type")
                ]
                if bias_types:
                    from collections import Counter

                    stats["most_common_bias"] = Counter(bias_types).most_common(1)[0][0]

        return stats

    def update_model(self, feedback_data: List[Dict[str, Any]]):
        """Update bias detection models with feedback (placeholder for online learning)."""
        # Add feedback to training data
        with self.lock:
            for feedback in feedback_data:
                self.training_data.append(feedback)

        # Trigger retraining if enough data
        if len(self.training_data) >= 1000:
            logger.info("Sufficient training data collected for model update")
            # In production, this would trigger model retraining
            # For now, just log the event

    def export_models(self, path: str):
        """Export bias detection models."""
        with self.lock:
            model_state = {
                "models": {
                    name: model.state_dict() for name, model in self.bias_models.items()
                },
                "version": self.model_version,
                "thresholds": self.bias_thresholds,
                "timestamp": time.time(),
            }

            torch.save(model_state, path)
            logger.info(f"Bias detection models exported to {path}")

    def load_models(self, path: str):
        """Load bias detection models."""
        with self.lock:
            model_state = torch.load(path, map_location=self.device, weights_only=True)

            for name, state_dict in model_state["models"].items():
                if name in self.bias_models:
                    self.bias_models[name].load_state_dict(state_dict)
                    self.bias_models[name] = self.bias_models[name].to(self.device)

            self.model_version = model_state.get("version", "unknown")
            self.bias_thresholds.update(model_state.get("thresholds", {}))

            logger.info(f"Bias detection models loaded from {path}")
