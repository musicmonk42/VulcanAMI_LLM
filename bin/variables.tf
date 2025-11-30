################################################################################
# VulcanAMI Infrastructure - Terraform Variables
# Version: 4.6.0
# Description: Comprehensive variable definitions for VulcanAMI infrastructure
################################################################################

################################################################################
# Project Metadata
################################################################################

variable "project" {
  description = "Project name/prefix used for resource naming and tagging"
  type        = string
  default     = "vulcanami"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,63}$", var.project))
    error_message = "Project name must start with a letter, contain only lowercase letters, numbers, and hyphens, and be 3-64 characters long."
  }
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "development", "staging", "qa", "uat", "production", "prod"], var.environment)
    error_message = "Environment must be one of: dev, development, staging, qa, uat, production, prod."
  }
}

variable "vulcanami_version" {
  description = "VulcanAMI version for tagging and resource identification"
  type        = string
  default     = "4.6.0"

  validation {
    condition     = can(regex("^\\d+\\.\\d+\\.\\d+$", var.vulcanami_version))
    error_message = "Version must follow semantic versioning format (e.g., 4.6.0)."
  }
}

variable "owner" {
  description = "Owner/team responsible for the infrastructure"
  type        = string
  default     = "data-platform"
}

variable "cost_center" {
  description = "Cost center code for billing and chargeback"
  type        = string
  default     = "engineering"
}

variable "contact_email" {
  description = "Contact email for infrastructure notifications"
  type        = string
  default     = "infrastructure@vulcanami.io"

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.contact_email))
    error_message = "Contact email must be a valid email address."
  }
}

################################################################################
# Regional Configuration
################################################################################

variable "primary_region" {
  description = "Primary AWS region for main infrastructure"
  type        = string
  default     = "us-east-1"

  validation {
    condition = contains([
      "us-east-1", "us-east-2", "us-west-1", "us-west-2",
      "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
      "ap-southeast-1", "ap-southeast-2", "ap-northeast-1", "ap-northeast-2",
      "ap-south-1", "sa-east-1", "ca-central-1"
    ], var.primary_region)
    error_message = "Primary region must be a valid AWS region."
  }
}

variable "secondary_region" {
  description = "Secondary AWS region for disaster recovery and replication"
  type        = string
  default     = "eu-west-1"

  validation {
    condition = contains([
      "us-east-1", "us-east-2", "us-west-1", "us-west-2",
      "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
      "ap-southeast-1", "ap-southeast-2", "ap-northeast-1", "ap-northeast-2",
      "ap-south-1", "sa-east-1", "ca-central-1"
    ], var.secondary_region)
    error_message = "Secondary region must be a valid AWS region."
  }
}

variable "additional_regions" {
  description = "Additional AWS regions for multi-region deployment"
  type        = list(string)
  default     = []
}

variable "availability_zones" {
  description = "Number of availability zones to use (1-6)"
  type        = number
  default     = 3

  validation {
    condition     = var.availability_zones >= 1 && var.availability_zones <= 6
    error_message = "Availability zones must be between 1 and 6."
  }
}

################################################################################
# Networking Configuration
################################################################################

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC"
  type        = bool
  default     = true
}

variable "enable_dns_support" {
  description = "Enable DNS support in the VPC"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Create a VPN gateway for the VPC"
  type        = bool
  default     = false
}

variable "enable_nat_gateway" {
  description = "Create NAT gateways for private subnets"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use a single NAT gateway instead of one per AZ (cost optimization)"
  type        = bool
  default     = false
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC flow logs for network monitoring"
  type        = bool
  default     = true
}

variable "domain_name" {
  description = "Primary domain name for CDN and services"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-\\.]{1,253}[a-z0-9]$", var.domain_name))
    error_message = "Domain name must be a valid DNS name."
  }
}

variable "additional_domains" {
  description = "Additional domain names for CDN aliases"
  type        = list(string)
  default     = []
}

variable "enable_ipv6" {
  description = "Enable IPv6 support"
  type        = bool
  default     = true
}

################################################################################
# S3 Bucket Configuration
################################################################################

variable "bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "vulcanami"
}

variable "enable_bucket_versioning" {
  description = "Enable versioning on S3 buckets"
  type        = bool
  default     = true
}

variable "enable_object_lock" {
  description = "Enable S3 Object Lock for WORM compliance"
  type        = bool
  default     = true
}

variable "object_lock_retention_days" {
  description = "Default retention period for Object Lock (days)"
  type        = number
  default     = 90
}

variable "enable_bucket_encryption" {
  description = "Enable server-side encryption for S3 buckets"
  type        = bool
  default     = true
}

variable "kms_key_rotation" {
  description = "Enable automatic key rotation for KMS keys"
  type        = bool
  default     = true
}

variable "bucket_lifecycle_enabled" {
  description = "Enable lifecycle policies for S3 buckets"
  type        = bool
  default     = true
}

variable "lifecycle_transition_glacier_days" {
  description = "Days before transitioning to Glacier"
  type        = number
  default     = 90
}

variable "lifecycle_transition_deep_archive_days" {
  description = "Days before transitioning to Deep Archive"
  type        = number
  default     = 365
}

variable "lifecycle_expiration_days" {
  description = "Days before object expiration (0 = never)"
  type        = number
  default     = 2555 # 7 years

  validation {
    condition     = var.lifecycle_expiration_days == 0 || var.lifecycle_expiration_days >= 1
    error_message = "Lifecycle expiration must be 0 (never) or >= 1 day."
  }
}

variable "enable_bucket_replication" {
  description = "Enable cross-region replication"
  type        = bool
  default     = true
}

variable "replication_storage_class" {
  description = "Storage class for replicated objects"
  type        = string
  default     = "STANDARD_IA"

  validation {
    condition = contains([
      "STANDARD", "STANDARD_IA", "ONEZONE_IA", "INTELLIGENT_TIERING",
      "GLACIER", "DEEP_ARCHIVE", "GLACIER_IR"
    ], var.replication_storage_class)
    error_message = "Invalid storage class for replication."
  }
}

variable "enable_bucket_logging" {
  description = "Enable S3 access logging"
  type        = bool
  default     = true
}

variable "enable_bucket_notifications" {
  description = "Enable S3 event notifications"
  type        = bool
  default     = true
}

################################################################################
# CloudFront CDN Configuration
################################################################################

variable "enable_cloudfront" {
  description = "Enable CloudFront CDN"
  type        = bool
  default     = true
}

variable "cloudfront_price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_All"

  validation {
    condition = contains([
      "PriceClass_All", "PriceClass_200", "PriceClass_100"
    ], var.cloudfront_price_class)
    error_message = "Invalid CloudFront price class."
  }
}

variable "cloudfront_min_ttl" {
  description = "CloudFront minimum TTL (seconds)"
  type        = number
  default     = 0
}

variable "cloudfront_default_ttl" {
  description = "CloudFront default TTL (seconds)"
  type        = number
  default     = 86400 # 24 hours
}

variable "cloudfront_max_ttl" {
  description = "CloudFront maximum TTL (seconds)"
  type        = number
  default     = 31536000 # 1 year
}

variable "enable_cloudfront_compression" {
  description = "Enable CloudFront compression"
  type        = bool
  default     = true
}

variable "enable_cloudfront_logging" {
  description = "Enable CloudFront access logging"
  type        = bool
  default     = true
}

variable "enable_cloudfront_waf" {
  description = "Enable WAF for CloudFront"
  type        = bool
  default     = true
}

variable "cloudfront_geo_restriction_type" {
  description = "Type of geo restriction (none, whitelist, blacklist)"
  type        = string
  default     = "none"

  validation {
    condition     = contains(["none", "whitelist", "blacklist"], var.cloudfront_geo_restriction_type)
    error_message = "Geo restriction type must be: none, whitelist, or blacklist."
  }
}

variable "cloudfront_geo_restriction_locations" {
  description = "List of country codes for geo restriction"
  type        = list(string)
  default     = []
}

variable "ssl_certificate_arn" {
  description = "ARN of ACM certificate for CloudFront (optional, will use default if not provided)"
  type        = string
  default     = ""
}

variable "enable_http2" {
  description = "Enable HTTP/2 for CloudFront"
  type        = bool
  default     = true
}

variable "enable_http3" {
  description = "Enable HTTP/3 for CloudFront"
  type        = bool
  default     = true
}

################################################################################
# Database Configuration (PostgreSQL RDS)
################################################################################

variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "rds_instance_class" {
  description = "RDS instance type"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "rds_allocated_storage" {
  description = "Allocated storage for RDS (GB)"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling (GB)"
  type        = number
  default     = 1000
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "14.10"
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ for RDS"
  type        = bool
  default     = true
}

variable "rds_backup_retention_days" {
  description = "Backup retention period (days)"
  type        = number
  default     = 30

  validation {
    condition     = var.rds_backup_retention_days >= 1 && var.rds_backup_retention_days <= 35
    error_message = "Backup retention must be between 1 and 35 days."
  }
}

variable "rds_backup_window" {
  description = "Preferred backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

variable "rds_maintenance_window" {
  description = "Preferred maintenance window (UTC)"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection for RDS"
  type        = bool
  default     = true
}

variable "rds_performance_insights_enabled" {
  description = "Enable Performance Insights"
  type        = bool
  default     = true
}

variable "rds_monitoring_interval" {
  description = "Enhanced monitoring interval (seconds, 0 to disable)"
  type        = number
  default     = 60

  validation {
    condition     = contains([0, 1, 5, 10, 15, 30, 60], var.rds_monitoring_interval)
    error_message = "Monitoring interval must be 0, 1, 5, 10, 15, 30, or 60 seconds."
  }
}

################################################################################
# ElastiCache Redis Configuration
################################################################################

variable "enable_redis" {
  description = "Enable ElastiCache Redis cluster"
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the cluster"
  type        = number
  default     = 3

  validation {
    condition     = var.redis_num_cache_nodes >= 1 && var.redis_num_cache_nodes <= 6
    error_message = "Redis nodes must be between 1 and 6."
  }
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "6.2"
}

variable "redis_parameter_group_family" {
  description = "Redis parameter group family"
  type        = string
  default     = "redis6.x"
}

variable "redis_snapshot_retention_limit" {
  description = "Number of days to retain automatic snapshots"
  type        = number
  default     = 7
}

variable "redis_snapshot_window" {
  description = "Daily time range for snapshots (UTC)"
  type        = string
  default     = "03:00-05:00"
}

variable "redis_maintenance_window" {
  description = "Weekly time range for maintenance (UTC)"
  type        = string
  default     = "sun:05:00-sun:07:00"
}

variable "redis_at_rest_encryption_enabled" {
  description = "Enable encryption at rest for Redis"
  type        = bool
  default     = true
}

variable "redis_transit_encryption_enabled" {
  description = "Enable encryption in transit for Redis"
  type        = bool
  default     = true
}

################################################################################
# ECS/Fargate Configuration
################################################################################

variable "enable_ecs" {
  description = "Enable ECS cluster for containerized services"
  type        = bool
  default     = true
}

variable "ecs_launch_type" {
  description = "ECS launch type (EC2 or FARGATE)"
  type        = string
  default     = "FARGATE"

  validation {
    condition     = contains(["EC2", "FARGATE"], var.ecs_launch_type)
    error_message = "ECS launch type must be EC2 or FARGATE."
  }
}

variable "ecs_dqs_task_cpu" {
  description = "CPU units for DQS service (1024 = 1 vCPU)"
  type        = number
  default     = 2048
}

variable "ecs_dqs_task_memory" {
  description = "Memory for DQS service (MB)"
  type        = number
  default     = 4096
}

variable "ecs_dqs_desired_count" {
  description = "Desired number of DQS tasks"
  type        = number
  default     = 3
}

variable "ecs_opa_task_cpu" {
  description = "CPU units for OPA service"
  type        = number
  default     = 1024
}

variable "ecs_opa_task_memory" {
  description = "Memory for OPA service (MB)"
  type        = number
  default     = 2048
}

variable "ecs_opa_desired_count" {
  description = "Desired number of OPA tasks"
  type        = number
  default     = 3
}

variable "enable_ecs_exec" {
  description = "Enable ECS Exec for debugging"
  type        = bool
  default     = true
}

variable "enable_container_insights" {
  description = "Enable Container Insights for ECS"
  type        = bool
  default     = true
}

################################################################################
# Lambda Configuration
################################################################################

variable "enable_lambda_functions" {
  description = "Enable Lambda functions for event processing"
  type        = bool
  default     = true
}

variable "lambda_runtime" {
  description = "Lambda runtime version"
  type        = string
  default     = "python3.11"
}

variable "lambda_memory_size" {
  description = "Lambda function memory size (MB)"
  type        = number
  default     = 512

  validation {
    condition     = var.lambda_memory_size >= 128 && var.lambda_memory_size <= 10240
    error_message = "Lambda memory must be between 128 MB and 10240 MB."
  }
}

variable "lambda_timeout" {
  description = "Lambda function timeout (seconds)"
  type        = number
  default     = 300

  validation {
    condition     = var.lambda_timeout >= 1 && var.lambda_timeout <= 900
    error_message = "Lambda timeout must be between 1 and 900 seconds."
  }
}

variable "lambda_reserved_concurrent_executions" {
  description = "Reserved concurrent executions for Lambda functions"
  type        = number
  default     = 10
}

################################################################################
# Monitoring and Observability
################################################################################

variable "enable_cloudwatch" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch Logs retention period (days)"
  type        = number
  default     = 30

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.cloudwatch_log_retention_days)
    error_message = "Invalid CloudWatch retention period."
  }
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_email_endpoints" {
  description = "Email addresses for alarm notifications"
  type        = list(string)
  default     = []
}

variable "enable_xray" {
  description = "Enable AWS X-Ray distributed tracing"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_elasticsearch" {
  description = "Enable Elasticsearch for logs and search"
  type        = bool
  default     = true
}

variable "elasticsearch_instance_type" {
  description = "Elasticsearch instance type"
  type        = string
  default     = "r6g.large.search"
}

variable "elasticsearch_instance_count" {
  description = "Number of Elasticsearch instances"
  type        = number
  default     = 3
}

variable "elasticsearch_ebs_volume_size" {
  description = "EBS volume size for Elasticsearch (GB)"
  type        = number
  default     = 100
}

################################################################################
# Security Configuration
################################################################################

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty threat detection"
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub"
  type        = bool
  default     = true
}

variable "enable_config" {
  description = "Enable AWS Config for compliance"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "cloudtrail_multi_region" {
  description = "Enable CloudTrail in all regions"
  type        = bool
  default     = true
}

variable "enable_secrets_manager" {
  description = "Enable AWS Secrets Manager"
  type        = bool
  default     = true
}

variable "secrets_rotation_days" {
  description = "Automatic rotation period for secrets (days)"
  type        = number
  default     = 90
}

variable "enable_waf" {
  description = "Enable AWS WAF"
  type        = bool
  default     = true
}

variable "waf_rate_limit" {
  description = "WAF rate limit (requests per 5 minutes)"
  type        = number
  default     = 2000
}

variable "allowed_ip_ranges" {
  description = "IP ranges allowed to access resources"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "enable_bastion_host" {
  description = "Create bastion host for SSH access"
  type        = bool
  default     = false
}

################################################################################
# Backup and Disaster Recovery
################################################################################

variable "enable_aws_backup" {
  description = "Enable AWS Backup service"
  type        = bool
  default     = true
}

variable "backup_plan_frequency" {
  description = "Backup frequency (daily, weekly, monthly)"
  type        = string
  default     = "daily"

  validation {
    condition     = contains(["daily", "weekly", "monthly"], var.backup_plan_frequency)
    error_message = "Backup frequency must be daily, weekly, or monthly."
  }
}

variable "backup_retention_days" {
  description = "Backup retention period (days)"
  type        = number
  default     = 30
}

variable "enable_disaster_recovery" {
  description = "Enable disaster recovery configuration"
  type        = bool
  default     = true
}

variable "dr_rto_minutes" {
  description = "Recovery Time Objective (minutes)"
  type        = number
  default     = 60
}

variable "dr_rpo_minutes" {
  description = "Recovery Point Objective (minutes)"
  type        = number
  default     = 15
}

################################################################################
# Cost Optimization
################################################################################

variable "enable_cost_allocation_tags" {
  description = "Enable cost allocation tags"
  type        = bool
  default     = true
}

variable "enable_savings_plans" {
  description = "Enable Savings Plans recommendations"
  type        = bool
  default     = true
}

variable "enable_spot_instances" {
  description = "Use Spot instances where applicable"
  type        = bool
  default     = false
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for applicable resources"
  type        = bool
  default     = true
}

variable "auto_scaling_min_capacity" {
  description = "Minimum capacity for auto-scaling"
  type        = number
  default     = 2
}

variable "auto_scaling_max_capacity" {
  description = "Maximum capacity for auto-scaling"
  type        = number
  default     = 10
}

variable "auto_scaling_target_cpu" {
  description = "Target CPU utilization for auto-scaling (%)"
  type        = number
  default     = 70

  validation {
    condition     = var.auto_scaling_target_cpu >= 1 && var.auto_scaling_target_cpu <= 100
    error_message = "Target CPU must be between 1 and 100."
  }
}

################################################################################
# Compliance and Governance
################################################################################

variable "compliance_framework" {
  description = "Compliance framework to adhere to"
  type        = string
  default     = "hipaa"

  validation {
    condition     = contains(["none", "hipaa", "gdpr", "sox", "pci_dss", "fedramp"], var.compliance_framework)
    error_message = "Invalid compliance framework."
  }
}

variable "enable_encryption_at_rest" {
  description = "Enforce encryption at rest for all resources"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enforce encryption in transit for all resources"
  type        = bool
  default     = true
}

variable "minimum_tls_version" {
  description = "Minimum TLS version"
  type        = string
  default     = "TLSv1.2"

  validation {
    condition     = contains(["TLSv1.2", "TLSv1.3"], var.minimum_tls_version)
    error_message = "TLS version must be TLSv1.2 or TLSv1.3."
  }
}

variable "data_residency_region" {
  description = "Region for data residency compliance"
  type        = string
  default     = ""
}

################################################################################
# Resource Tagging
################################################################################

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "tag_compliance" {
  description = "Compliance-related tags"
  type        = map(string)
  default = {
    "Compliance:HIPAA" = "true"
    "Compliance:GDPR"  = "true"
    "Compliance:SOX"   = "true"
  }
}

################################################################################
# Feature Flags
################################################################################

variable "enable_experimental_features" {
  description = "Enable experimental features (not for production)"
  type        = bool
  default     = false
}

variable "enable_debug_mode" {
  description = "Enable debug mode with verbose logging"
  type        = bool
  default     = false
}

variable "enable_cost_optimization" {
  description = "Enable aggressive cost optimization features"
  type        = bool
  default     = false
}

variable "enable_high_availability" {
  description = "Enable high availability features"
  type        = bool
  default     = true
}

################################################################################
# Advanced Configuration
################################################################################

variable "custom_kms_key_policy" {
  description = "Custom KMS key policy (JSON string)"
  type        = string
  default     = ""
}

variable "custom_iam_policies" {
  description = "Custom IAM policies to attach"
  type        = list(string)
  default     = []
}

variable "enable_private_link" {
  description = "Enable AWS PrivateLink for services"
  type        = bool
  default     = false
}

variable "enable_transit_gateway" {
  description = "Enable AWS Transit Gateway for network connectivity"
  type        = bool
  default     = false
}

variable "enable_direct_connect" {
  description = "Enable AWS Direct Connect"
  type        = bool
  default     = false
}

################################################################################
# End of Variables
################################################################################
