package graphix.vulcan.writebarrier

################################################################################
# VulcanAMI Write Barrier Policy
# Version: 2.0.0
# Bundle: p2025.11
# 
# This policy implements comprehensive write barrier controls based on:
# - Data Quality Score (DQS)
# - PII Detection and Review Status
# - Access Control and Authorization
# - Compliance Requirements (GDPR, HIPAA, SOX)
# - Rate Limiting and Quotas
# - Temporal Constraints
# - Geographic Restrictions
# - Audit and Logging Requirements
################################################################################

import future.keywords.if
import future.keywords.in
import future.keywords.every
import future.keywords.contains

################################################################################
# Default Policies
################################################################################

# Default deny-by-default security posture
default allow = false
default quarantine = false
default reject = true

# Default metadata
default audit_required = true
default encryption_required = false
default redaction_required = false

################################################################################
# Main Decision: Allow Write
################################################################################

# Allow writes for high-quality data with PII review
allow {
    # Quality Requirements
    dqs_check_passed
    
    # PII Requirements
    pii_check_passed
    
    # Access Control
    authorization_check_passed
    
    # Compliance
    compliance_check_passed
    
    # Rate Limiting
    not rate_limit_exceeded
    
    # Temporal Constraints
    temporal_constraints_met
    
    # Geographic Restrictions
    geographic_restrictions_met
    
    # Additional Validations
    not blocked_source
    not blacklisted_user
}

# Allow with warnings for acceptable quality
allow_with_warning {
    # Slightly relaxed DQS requirement
    input.dqs >= data.thresholds.warning
    input.dqs < data.thresholds.accept
    
    # Still requires PII review
    pii_check_passed
    
    # All other checks must pass
    authorization_check_passed
    compliance_check_passed
    not rate_limit_exceeded
}

################################################################################
# Quarantine Decision
################################################################################

# Quarantine data that needs remediation
quarantine {
    # DQS in quarantine range
    input.dqs >= data.thresholds.quarantine
    input.dqs < data.thresholds.warning
    
    # Check if auto-remediation is possible
    not auto_remediation_attempted
}

# Quarantine if PII not reviewed
quarantine {
    dqs_check_passed
    not pii_check_passed
    pii_contains_sensitive_data
}

# Quarantine if compliance review required
quarantine {
    dqs_check_passed
    compliance_review_required
    not compliance_review_completed
}

# Quarantine if rate limit approached
quarantine {
    rate_limit_warning_threshold_reached
    not rate_limit_exceeded
}

################################################################################
# Reject Decision
################################################################################

# Reject low-quality data
reject {
    input.dqs < data.thresholds.quarantine
}

# Reject if critical PII found without review
reject {
    pii_contains_critical_data
    not pii_stage2_reviewed
}

# Reject if authorization fails
reject {
    not authorization_check_passed
}

# Reject if compliance violations detected
reject {
    compliance_violation_detected
}

# Reject if rate limit exceeded
reject {
    rate_limit_exceeded
}

# Reject if from blacklisted source
reject {
    blocked_source
}

# Reject if temporal window closed
reject {
    not temporal_constraints_met
}

# Reject if geographic restrictions violated
reject {
    not geographic_restrictions_met
}

################################################################################
# Quality Checks
################################################################################

# Main DQS check
dqs_check_passed {
    input.dqs >= data.thresholds.accept
}

# DQS with dimension breakdown
dqs_dimension_check_passed {
    # Check critical dimensions
    every dimension in data.critical_dimensions {
        input.dimension_scores[dimension] >= data.dimension_thresholds[dimension]
    }
}

# Quality category check
quality_category_acceptable {
    input.category in data.acceptable_categories
}

################################################################################
# PII Checks
################################################################################

# Basic PII check
pii_check_passed {
    # No PII detected
    not pii_detected
}

pii_check_passed {
    # PII detected but reviewed
    pii_detected
    pii_stage2_reviewed
}

pii_check_passed {
    # PII detected, reviewed, and anonymized
    pii_detected
    pii_stage2_reviewed
    pii_anonymized
}

# PII detection
pii_detected {
    count(input.pii.detected_types) > 0
}

# Stage 2 review check
pii_stage2_reviewed {
    input.pii.stage2_reviewed == true
}

# Anonymization check
pii_anonymized {
    input.pii.anonymized == true
}

# Sensitive PII check
pii_contains_sensitive_data {
    some pii_type in input.pii.detected_types
    pii_type in data.pii.sensitive_types
}

# Critical PII check
pii_contains_critical_data {
    some pii_type in input.pii.detected_types
    pii_type in data.pii.critical_types
}

# Redaction check
pii_redaction_required {
    pii_contains_sensitive_data
    not pii_anonymized
}

################################################################################
# Authorization and Access Control
################################################################################

# Main authorization check
authorization_check_passed {
    # User authenticated
    user_authenticated
    
    # User has required permissions
    user_has_permission
    
    # User not suspended
    not user_suspended
    
    # API key valid
    api_key_valid
}

# User authentication
user_authenticated {
    input.user.id != null
    input.user.authenticated == true
}

# Permission check
user_has_permission {
    # Check role-based permissions
    some role in input.user.roles
    role in data.authorized_roles
}

user_has_permission {
    # Check explicit user permissions
    input.user.id in data.authorized_users
}

# User suspension check
user_suspended {
    input.user.id in data.suspended_users
}

# API key validation
api_key_valid {
    input.api_key != null
    input.api_key != ""
    not api_key_expired
    not api_key_revoked
}

api_key_expired {
    input.api_key_expires_at < time.now_ns()
}

api_key_revoked {
    input.api_key in data.revoked_api_keys
}

# Blacklisted user check
blacklisted_user {
    input.user.id in data.blacklisted_users
}

################################################################################
# Compliance Checks
################################################################################

# Main compliance check
compliance_check_passed {
    # GDPR compliance
    gdpr_compliant
    
    # HIPAA compliance (if applicable)
    hipaa_compliant_if_required
    
    # SOX compliance (if applicable)
    sox_compliant_if_required
    
    # Data residency requirements
    data_residency_compliant
}

# GDPR compliance
gdpr_compliant {
    # Data subject consent obtained
    not requires_gdpr_consent
}

gdpr_compliant {
    requires_gdpr_consent
    input.compliance.gdpr_consent_obtained == true
    input.compliance.gdpr_consent_timestamp != null
}

requires_gdpr_consent {
    # European data subject
    input.data_subject.region in data.gdpr_regions
}

# HIPAA compliance
hipaa_compliant_if_required {
    not requires_hipaa_compliance
}

hipaa_compliant_if_required {
    requires_hipaa_compliance
    input.compliance.hipaa_compliant == true
    input.compliance.phi_protected == true
}

requires_hipaa_compliance {
    # Contains Protected Health Information
    some label in input.labels
    label == "contains_phi"
}

# SOX compliance
sox_compliant_if_required {
    not requires_sox_compliance
}

sox_compliant_if_required {
    requires_sox_compliance
    input.compliance.sox_compliant == true
    input.compliance.financial_data_verified == true
}

requires_sox_compliance {
    # Contains financial data
    input.data_type == "financial"
}

# Data residency
data_residency_compliant {
    not has_residency_requirements
}

data_residency_compliant {
    has_residency_requirements
    input.storage_region in allowed_storage_regions
}

has_residency_requirements {
    input.data_subject.region in data.regions_with_residency_requirements
}

allowed_storage_regions contains region {
    some region in data.allowed_storage_regions[input.data_subject.region]
}

# Compliance review
compliance_review_required {
    # High-risk data
    input.risk_level == "high"
}

compliance_review_required {
    # Large-scale processing
    input.record_count > data.compliance.review_threshold_records
}

compliance_review_completed {
    input.compliance.review_completed == true
    input.compliance.reviewer_id != null
}

compliance_violation_detected {
    # Missing required consent
    requires_gdpr_consent
    not input.compliance.gdpr_consent_obtained
}

compliance_violation_detected {
    # Residency violation
    has_residency_requirements
    not data_residency_compliant
}

################################################################################
# Rate Limiting and Quotas
################################################################################

# Rate limit check
rate_limit_exceeded {
    # Per-user rate limit
    user_rate_limit_exceeded
}

rate_limit_exceeded {
    # Per-source rate limit
    source_rate_limit_exceeded
}

rate_limit_exceeded {
    # Global rate limit
    global_rate_limit_exceeded
}

user_rate_limit_exceeded {
    input.user.id != null
    input.user.request_count >= data.rate_limits.per_user
}

source_rate_limit_exceeded {
    input.source != null
    input.source.request_count >= data.rate_limits.per_source
}

global_rate_limit_exceeded {
    input.system.current_request_rate >= data.rate_limits.global
}

rate_limit_warning_threshold_reached {
    input.user.request_count >= (data.rate_limits.per_user * 0.8)
}

# Quota check
quota_exceeded {
    input.user.storage_used >= input.user.storage_quota
}

quota_exceeded {
    input.user.records_stored >= input.user.records_quota
}

################################################################################
# Temporal Constraints
################################################################################

# Temporal window check
temporal_constraints_met {
    # Always allowed if no restrictions
    not has_temporal_restrictions
}

temporal_constraints_met {
    has_temporal_restrictions
    within_allowed_time_window
}

has_temporal_restrictions {
    data.temporal_restrictions.enabled == true
}

within_allowed_time_window {
    current_hour := time.clock([time.now_ns()])[0]
    current_hour >= data.temporal_restrictions.start_hour
    current_hour < data.temporal_restrictions.end_hour
}

within_allowed_time_window {
    # Check if current day is allowed
    current_day := time.weekday([time.now_ns()])
    current_day in data.temporal_restrictions.allowed_days
}

# Embargo check
embargo_active {
    input.data_subject.country in data.embargoed_countries
}

################################################################################
# Geographic Restrictions
################################################################################

geographic_restrictions_met {
    not has_geographic_restrictions
}

geographic_restrictions_met {
    has_geographic_restrictions
    source_region_allowed
    destination_region_allowed
}

has_geographic_restrictions {
    count(data.geographic_restrictions.blocked_regions) > 0
}

source_region_allowed {
    not input.source.region in data.geographic_restrictions.blocked_regions
}

destination_region_allowed {
    not input.destination.region in data.geographic_restrictions.blocked_regions
}

################################################################################
# Source Validation
################################################################################

blocked_source {
    input.source.id in data.blocked_sources
}

blocked_source {
    # Source credibility too low
    input.source.credibility < data.thresholds.min_source_credibility
}

trusted_source {
    input.source.id in data.trusted_sources
}

################################################################################
# Auto-Remediation
################################################################################

auto_remediation_attempted {
    input.remediation.attempted == true
}

auto_remediation_available {
    input.dqs >= data.thresholds.quarantine
    input.dqs < data.thresholds.warning
    count(available_remediation_strategies) > 0
}

available_remediation_strategies contains strategy {
    some strategy in data.remediation.strategies
    strategy_applicable(strategy)
}

strategy_applicable(strategy) {
    # Check if strategy can fix detected issues
    some label in input.labels
    label in data.remediation.strategy_mappings[strategy]
}

################################################################################
# Audit and Metadata
################################################################################

# Audit requirement
audit_required {
    # Always audit PII
    pii_detected
}

audit_required {
    # Audit compliance-related data
    compliance_review_required
}

audit_required {
    # Audit rejections
    reject
}

audit_required {
    # Audit high-value data
    input.value_classification == "high"
}

# Encryption requirement
encryption_required {
    # Encrypt sensitive PII
    pii_contains_sensitive_data
}

encryption_required {
    # Encrypt PHI
    requires_hipaa_compliance
}

encryption_required {
    # Encrypt financial data
    requires_sox_compliance
}

# Redaction requirement
redaction_required {
    pii_redaction_required
    allow
}

################################################################################
# Decision Metadata
################################################################################

# Reason for allow
allow_reason contains reason {
    dqs_check_passed
    reason := "dqs_threshold_met"
}

allow_reason contains reason {
    pii_check_passed
    reason := "pii_reviewed"
}

allow_reason contains reason {
    authorization_check_passed
    reason := "authorization_granted"
}

# Reason for quarantine
quarantine_reason contains reason {
    input.dqs >= data.thresholds.quarantine
    input.dqs < data.thresholds.warning
    reason := "dqs_in_quarantine_range"
}

quarantine_reason contains reason {
    not pii_check_passed
    reason := "pii_review_required"
}

quarantine_reason contains reason {
    compliance_review_required
    not compliance_review_completed
    reason := "compliance_review_pending"
}

# Reason for reject
reject_reason contains reason {
    input.dqs < data.thresholds.quarantine
    reason := sprintf("dqs_below_threshold: %.3f < %.3f", [input.dqs, data.thresholds.quarantine])
}

reject_reason contains reason {
    pii_contains_critical_data
    not pii_stage2_reviewed
    reason := "critical_pii_not_reviewed"
}

reject_reason contains reason {
    not authorization_check_passed
    reason := "authorization_failed"
}

reject_reason contains reason {
    compliance_violation_detected
    reason := "compliance_violation"
}

reject_reason contains reason {
    rate_limit_exceeded
    reason := "rate_limit_exceeded"
}

################################################################################
# Actions and Recommendations
################################################################################

# Required actions
required_actions contains action {
    pii_redaction_required
    action := {
        "type": "redact_pii",
        "fields": input.pii.detected_fields,
        "priority": "high"
    }
}

required_actions contains action {
    encryption_required
    action := {
        "type": "encrypt_data",
        "algorithm": "AES-256-GCM",
        "priority": "critical"
    }
}

required_actions contains action {
    auto_remediation_available
    action := {
        "type": "auto_remediate",
        "strategies": available_remediation_strategies,
        "priority": "medium"
    }
}

# Recommendations
recommendations contains rec {
    allow_with_warning
    rec := "data_quality_below_optimal"
}

recommendations contains rec {
    quota_exceeded
    rec := "quota_exceeded_upgrade_plan"
}

recommendations contains rec {
    rate_limit_warning_threshold_reached
    rec := "approaching_rate_limit"
}

################################################################################
# Decision Result
################################################################################

# Main decision result
decision := {
    "allow": allow,
    "allow_with_warning": allow_with_warning,
    "quarantine": quarantine,
    "reject": reject,
    "metadata": {
        "audit_required": audit_required,
        "encryption_required": encryption_required,
        "redaction_required": redaction_required,
        "timestamp": time.now_ns(),
        "policy_version": data.policy_version,
        "bundle_version": data.bundle_version
    },
    "reasons": {
        "allow": allow_reason,
        "quarantine": quarantine_reason,
        "reject": reject_reason
    },
    "required_actions": required_actions,
    "recommendations": recommendations,
    "checks": {
        "dqs": dqs_check_passed,
        "pii": pii_check_passed,
        "authorization": authorization_check_passed,
        "compliance": compliance_check_passed,
        "rate_limit": not rate_limit_exceeded,
        "temporal": temporal_constraints_met,
        "geographic": geographic_restrictions_met
    }
}

################################################################################
# Batch Decision Support
################################################################################

# Batch validation
batch_decision contains result {
    some i, item in input.batch_items
    result := {
        "item_id": item.id,
        "decision": item_decision(item),
        "index": i
    }
}

item_decision(item) := result {
    # Evaluate policy for individual item
    result := decision with input as item
}

################################################################################
# Metrics and Telemetry
################################################################################

# Decision metrics
metrics := {
    "dqs_score": input.dqs,
    "pii_detected": pii_detected,
    "pii_count": count(input.pii.detected_types),
    "dimension_scores": input.dimension_scores,
    "processing_time_ms": input.processing_time_ms,
    "decision_path": decision_path
}

decision_path := path {
    allow
    path := "allow"
}

decision_path := path {
    quarantine
    path := "quarantine"
}

decision_path := path {
    reject
    path := "reject"
}
