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

  validation {
    condition     = length(var.owner) > 0 && length(var.owner) <= 100
    error_message = "Owner must be between 1 and 100 characters."
  }
}

variable "cost_center" {
  description = "Cost center code for billing and chargeback"
  type        = string
  default     = "engineering"

  validation {
    condition     = length(var.cost_center) > 0 && length(var.cost_center) <= 50
    error_message = "Cost center must be between 1 and 50 characters."
  }
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

  validation {
    condition     = length(var.additional_regions) <= 5
    error_message = "Maximum of 5 additional regions allowed."
  }
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

  validation {
    condition     = length(var.additional_domains) <= 10
    error_message = "Maximum of 10 additional domains allowed."
  }
}

variable "enable_ipv6" {
  description = "Enable IPv6 support"
  type        = bool
  default     = true
}

# Security: IP Access Control
# Empty default requires explicit specification for security
# IMPORTANT: Resources expecting IP ranges must handle empty list or provide their own defaults
variable "allowed_ip_ranges" {
  description = <<-EOT
    IP ranges allowed to access ALB and other public resources.
    Empty default requires explicit specification for security.
    Example: ["10.0.0.0/8", "172.16.0.0/12"]
    WARNING: Using ["0.0.0.0/0"] allows unrestricted internet access
  EOT
  type        = list(string)
  default     = [] # Empty default requires explicit IP range specification

  validation {
    condition = alltrue([
      for ip in var.allowed_ip_ranges : can(cidrhost(ip, 0))
    ])
    error_message = "All allowed IP ranges must be valid CIDR blocks."
  }
}

################################################################################
# S3 Bucket Configuration
################################################################################

variable "bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "vulcanami"

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{2,61}$", var.bucket_prefix))
    error_message = "Bucket prefix must start with lowercase letter or number, contain only lowercase letters, numbers, and hyphens, and be 3-62 characters long."
  }
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

  validation {
    condition     = var.object_lock_retention_days >= 1 && var.object_lock_retention_days <= 36500
    error_message = "Object lock retention must be between 1 and 36500 days."
  }
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

  validation {
    condition     = var.lifecycle_transition_glacier_days >= 30
    error_message = "Glacier transition must be at least 30 days."
  }
}

variable "lifecycle_transition_deep_archive_days" {
  description = "Days before transitioning to Deep Archive"
  type        = number
  default     = 365

  validation {
    condition     = var.lifecycle_transition_deep_archive_days >= 180
    error_message = "Deep Archive transition must be at least 180 days."
  }
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

################################################################################
# CloudFront Configuration
################################################################################

variable "enable_cloudfront" {
  description = "Enable CloudFront distribution"
  type        = bool
  default     = true
}

variable "cloudfront_price_class" {
  description = "CloudFront distribution price class"
  type        = string
  default     = "PriceClass_200"

  validation {
    condition = contains([
      "PriceClass_All", "PriceClass_200", "PriceClass_100"
    ], var.cloudfront_price_class)
    error_message = "Invalid CloudFront price class."
  }
}

variable "cloudfront_allowed_methods" {
  description = "CloudFront allowed HTTP methods"
  type        = list(string)
  default     = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]

  validation {
    condition = alltrue([
      for method in var.cloudfront_allowed_methods :
      contains(["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"], method)
    ])
    error_message = "Invalid HTTP method in allowed methods list."
  }
}

variable "cloudfront_cached_methods" {
  description = "CloudFront cached HTTP methods"
  type        = list(string)
  default     = ["GET", "HEAD"]

  validation {
    condition = alltrue([
      for method in var.cloudfront_cached_methods :
      contains(["GET", "HEAD"], method)
    ])
    error_message = "Only GET and HEAD methods can be cached."
  }
}

variable "cloudfront_min_ttl" {
  description = "CloudFront minimum TTL"
  type        = number
  default     = 0

  validation {
    condition     = var.cloudfront_min_ttl >= 0
    error_message = "Minimum TTL must be non-negative."
  }
}

variable "cloudfront_default_ttl" {
  description = "CloudFront default TTL"
  type        = number
  default     = 3600

  validation {
    condition     = var.cloudfront_default_ttl >= 0
    error_message = "Default TTL must be non-negative."
  }
}

variable "cloudfront_max_ttl" {
  description = "CloudFront maximum TTL"
  type        = number
  default     = 86400

  validation {
    condition     = var.cloudfront_max_ttl >= 0
    error_message = "Maximum TTL must be non-negative."
  }
}

variable "cloudfront_compress" {
  description = "Enable CloudFront compression"
  type        = bool
  default     = true
}

variable "cloudfront_forward_query_string" {
  description = "Forward query strings to origin"
  type        = bool
  default     = false
}

variable "cloudfront_forward_cookies" {
  description = "Forward cookies to origin (none, whitelist, all)"
  type        = string
  default     = "none"

  validation {
    condition     = contains(["none", "whitelist", "all"], var.cloudfront_forward_cookies)
    error_message = "Cookie forwarding must be none, whitelist, or all."
  }
}

variable "cloudfront_forward_headers" {
  description = "Headers to forward to origin"
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.cloudfront_forward_headers) <= 10
    error_message = "Maximum of 10 headers can be forwarded."
  }
}

variable "cloudfront_geo_restriction_type" {
  description = "CloudFront geo restriction type"
  type        = string
  default     = "none"

  validation {
    condition     = contains(["none", "whitelist", "blacklist"], var.cloudfront_geo_restriction_type)
    error_message = "Geo restriction type must be none, whitelist, or blacklist."
  }
}

variable "cloudfront_geo_restriction_locations" {
  description = "CloudFront geo restriction locations (country codes)"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for location in var.cloudfront_geo_restriction_locations :
      can(regex("^[A-Z]{2}$", location))
    ])
    error_message = "Geo restriction locations must be 2-letter country codes."
  }
}

variable "enable_cloudfront_logging" {
  description = "Enable CloudFront access logging"
  type        = bool
  default     = true
}

variable "cloudfront_log_cookies" {
  description = "Include cookies in CloudFront logs"
  type        = bool
  default     = false
}

variable "enable_cloudfront_waf" {
  description = "Enable WAF for CloudFront"
  type        = bool
  default     = true
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for CloudFront (must be in us-east-1)"
  type        = string
  default     = ""

  validation {
    condition     = var.acm_certificate_arn == "" || can(regex("^arn:aws:acm:us-east-1:[0-9]{12}:certificate/", var.acm_certificate_arn))
    error_message = "ACM certificate ARN must be empty or a valid certificate ARN in us-east-1."
  }
}

variable "alb_certificate_arn" {
  description = "ACM certificate ARN for ALB (must be in the same region as the ALB)"
  type        = string
  default     = ""

  validation {
    condition     = var.alb_certificate_arn == "" || can(regex("^arn:aws:acm:[a-z0-9-]+:[0-9]{12}:certificate/", var.alb_certificate_arn))
    error_message = "ALB certificate ARN must be empty or a valid ACM certificate ARN."
  }
}

################################################################################
# RDS Configuration
################################################################################

variable "enable_rds" {
  description = "Enable RDS database"
  type        = bool
  default     = true
}

variable "rds_engine" {
  description = "RDS database engine"
  type        = string
  default     = "postgres"

  validation {
    condition     = contains(["postgres", "mysql", "mariadb", "aurora-postgresql", "aurora-mysql"], var.rds_engine)
    error_message = "RDS engine must be postgres, mysql, mariadb, aurora-postgresql, or aurora-mysql."
  }
}

variable "rds_engine_version" {
  description = "RDS engine version"
  type        = string
  default     = "15.4"

  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+", var.rds_engine_version))
    error_message = "RDS engine version must be in format X.Y."
  }
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"

  validation {
    condition     = can(regex("^db\\.[a-z0-9]+\\.[a-z0-9]+$", var.rds_instance_class))
    error_message = "Invalid RDS instance class format."
  }
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100

  validation {
    condition     = var.rds_allocated_storage >= 20 && var.rds_allocated_storage <= 65536
    error_message = "RDS allocated storage must be between 20 and 65536 GB."
  }
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage for autoscaling (GB)"
  type        = number
  default     = 1000

  validation {
    condition     = var.rds_max_allocated_storage >= 20 && var.rds_max_allocated_storage <= 65536
    error_message = "RDS max allocated storage must be between 20 and 65536 GB."
  }
}

variable "rds_storage_type" {
  description = "RDS storage type"
  type        = string
  default     = "gp3"

  validation {
    condition     = contains(["gp2", "gp3", "io1", "io2"], var.rds_storage_type)
    error_message = "RDS storage type must be gp2, gp3, io1, or io2."
  }
}

variable "rds_database_name" {
  description = "RDS database name"
  type        = string
  default     = "vulcanami"

  validation {
    condition     = can(regex("^[a-z][a-z0-9_]{0,63}$", var.rds_database_name))
    error_message = "Database name must start with a letter, contain only lowercase letters, numbers, and underscores, and be up to 64 characters."
  }
}

variable "rds_username" {
  description = "RDS master username"
  type        = string
  default     = "dbadmin"

  validation {
    condition     = can(regex("^[a-z][a-z0-9_]{0,15}$", var.rds_username))
    error_message = "Username must start with a letter, contain only lowercase letters, numbers, and underscores, and be up to 16 characters."
  }
}

variable "rds_port" {
  description = "RDS database port"
  type        = number
  default     = 5432

  validation {
    condition     = var.rds_port >= 1150 && var.rds_port <= 65535
    error_message = "RDS port must be between 1150 and 65535."
  }
}

variable "rds_multi_az" {
  description = "Enable RDS Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period (days)"
  type        = number
  default     = 30

  validation {
    condition     = var.rds_backup_retention_period >= 1 && var.rds_backup_retention_period <= 35
    error_message = "RDS backup retention period must be between 1 and 35 days."
  }
}

variable "rds_backup_window" {
  description = "RDS backup window"
  type        = string
  default     = "03:00-04:00"

  validation {
    condition     = can(regex("^([01][0-9]|2[0-3]):[0-5][0-9]-([01][0-9]|2[0-3]):[0-5][0-9]$", var.rds_backup_window))
    error_message = "RDS backup window must be in format HH:MM-HH:MM."
  }
}

variable "rds_maintenance_window" {
  description = "RDS maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"

  validation {
    condition     = can(regex("^(mon|tue|wed|thu|fri|sat|sun):[0-2][0-9]:[0-5][0-9]-(mon|tue|wed|thu|fri|sat|sun):[0-2][0-9]:[0-5][0-9]$", var.rds_maintenance_window))
    error_message = "RDS maintenance window must be in format ddd:HH:MM-ddd:HH:MM."
  }
}

variable "rds_auto_minor_version_upgrade" {
  description = "Enable automatic minor version upgrades for RDS"
  type        = bool
  default     = true
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection for RDS"
  type        = bool
  default     = true
}

variable "rds_cloudwatch_logs_exports" {
  description = "RDS log types to export to CloudWatch"
  type        = list(string)
  default     = ["postgresql"]

  validation {
    condition = alltrue([
      for log in var.rds_cloudwatch_logs_exports :
      contains(["postgresql", "upgrade", "error", "general", "slowquery", "audit"], log)
    ])
    error_message = "Invalid log type for CloudWatch export."
  }
}

variable "rds_create_read_replica" {
  description = "Create a read replica for RDS"
  type        = bool
  default     = false
}

################################################################################
# ElastiCache Redis Configuration
################################################################################

variable "enable_redis" {
  description = "Enable Redis ElastiCache"
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"

  validation {
    condition     = can(regex("^cache\\.[a-z0-9]+\\.[a-z0-9]+$", var.redis_node_type))
    error_message = "Invalid Redis node type format."
  }
}

variable "redis_port" {
  description = "Redis port"
  type        = number
  default     = 6379

  validation {
    condition     = var.redis_port >= 1024 && var.redis_port <= 65535
    error_message = "Redis port must be between 1024 and 65535."
  }
}

variable "redis_family" {
  description = "Redis parameter group family"
  type        = string
  default     = "redis7"

  validation {
    condition     = can(regex("^redis[0-9]+(\\.[0-9]+)?$", var.redis_family))
    error_message = "Invalid Redis family format."
  }
}

variable "redis_parameters" {
  description = "Redis parameter overrides"
  type        = map(string)
  default     = {}
}

variable "redis_automatic_failover" {
  description = "Enable automatic failover for Redis"
  type        = bool
  default     = true
}

variable "redis_multi_az" {
  description = "Enable Multi-AZ for Redis"
  type        = bool
  default     = true
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters (nodes) for Redis"
  type        = number
  default     = 2

  validation {
    condition     = var.redis_num_cache_clusters >= 1 && var.redis_num_cache_clusters <= 6
    error_message = "Number of Redis cache clusters must be between 1 and 6."
  }
}

variable "redis_snapshot_retention_limit" {
  description = "Redis snapshot retention limit (days)"
  type        = number
  default     = 7

  validation {
    condition     = var.redis_snapshot_retention_limit >= 0 && var.redis_snapshot_retention_limit <= 35
    error_message = "Redis snapshot retention must be between 0 and 35 days."
  }
}

variable "redis_snapshot_window" {
  description = "Redis snapshot window"
  type        = string
  default     = "03:00-05:00"

  validation {
    condition     = can(regex("^([01][0-9]|2[0-3]):[0-5][0-9]-([01][0-9]|2[0-3]):[0-5][0-9]$", var.redis_snapshot_window))
    error_message = "Redis snapshot window must be in format HH:MM-HH:MM."
  }
}

variable "redis_maintenance_window" {
  description = "Redis maintenance window"
  type        = string
  default     = "sun:05:00-sun:07:00"

  validation {
    condition     = can(regex("^(mon|tue|wed|thu|fri|sat|sun):[0-2][0-9]:[0-5][0-9]-(mon|tue|wed|thu|fri|sat|sun):[0-2][0-9]:[0-5][0-9]$", var.redis_maintenance_window))
    error_message = "Redis maintenance window must be in format ddd:HH:MM-ddd:HH:MM."
  }
}

variable "redis_auto_minor_version_upgrade" {
  description = "Enable automatic minor version upgrades for Redis"
  type        = bool
  default     = true
}

################################################################################
# ECS Configuration
################################################################################

variable "enable_ecs" {
  description = "Enable ECS cluster"
  type        = bool
  default     = true
}

variable "ecs_container_insights" {
  description = "Enable Container Insights for ECS"
  type        = bool
  default     = true
}

variable "ecs_capacity_providers" {
  description = "ECS capacity providers"
  type        = list(string)
  default     = ["FARGATE", "FARGATE_SPOT"]

  validation {
    condition = alltrue([
      for provider in var.ecs_capacity_providers :
      contains(["FARGATE", "FARGATE_SPOT"], provider)
    ])
    error_message = "ECS capacity providers must be FARGATE or FARGATE_SPOT."
  }
}

variable "ecs_launch_type" {
  description = "ECS launch type"
  type        = string
  default     = "FARGATE"

  validation {
    condition     = contains(["EC2", "FARGATE"], var.ecs_launch_type)
    error_message = "ECS launch type must be EC2 or FARGATE."
  }
}

variable "ecs_task_cpu" {
  description = "ECS task CPU units"
  type        = string
  default     = "256"

  validation {
    condition     = contains(["256", "512", "1024", "2048", "4096"], var.ecs_task_cpu)
    error_message = "ECS task CPU must be 256, 512, 1024, 2048, or 4096."
  }
}

variable "ecs_task_memory" {
  description = "ECS task memory (MiB)"
  type        = string
  default     = "512"

  validation {
    condition     = contains(["512", "1024", "2048", "3072", "4096", "8192", "16384", "30720"], var.ecs_task_memory)
    error_message = "Invalid ECS task memory configuration."
  }
}

variable "ecs_service_desired_count" {
  description = "Desired number of ECS service tasks"
  type        = number
  default     = 2

  validation {
    condition     = var.ecs_service_desired_count >= 1 && var.ecs_service_desired_count <= 100
    error_message = "ECS service desired count must be between 1 and 100."
  }
}

variable "dqs_image" {
  description = "Docker image for DQS service"
  type        = string
  default     = "public.ecr.aws/amazonlinux/amazonlinux:latest"

  validation {
    condition     = length(var.dqs_image) > 0
    error_message = "DQS image must not be empty."
  }
}

variable "dqs_container_port" {
  description = "Container port for DQS service"
  type        = number
  default     = 8080

  validation {
    condition     = var.dqs_container_port >= 1 && var.dqs_container_port <= 65535
    error_message = "DQS container port must be between 1 and 65535."
  }
}

variable "dqs_environment_variables" {
  description = "Environment variables for DQS container"
  type        = list(map(string))
  default     = []
}

variable "dqs_health_check_path" {
  description = "Health check path for DQS service"
  type        = string
  default     = "/health"

  validation {
    condition     = can(regex("^/", var.dqs_health_check_path))
    error_message = "Health check path must start with /."
  }
}

variable "opa_image" {
  description = "Docker image for OPA service"
  type        = string
  default     = "openpolicyagent/opa:latest"

  validation {
    condition     = length(var.opa_image) > 0
    error_message = "OPA image must not be empty."
  }
}

variable "opa_container_port" {
  description = "Container port for OPA service"
  type        = number
  default     = 8181

  validation {
    condition     = var.opa_container_port >= 1 && var.opa_container_port <= 65535
    error_message = "OPA container port must be between 1 and 65535."
  }
}

variable "opa_environment_variables" {
  description = "Environment variables for OPA container"
  type        = list(map(string))
  default     = []
}

variable "opa_health_check_path" {
  description = "Health check path for OPA service"
  type        = string
  default     = "/health"

  validation {
    condition     = can(regex("^/", var.opa_health_check_path))
    error_message = "Health check path must start with /."
  }
}

################################################################################
# ALB Configuration
################################################################################

variable "alb_deletion_protection" {
  description = "Enable deletion protection for ALB"
  type        = bool
  default     = true
}

variable "enable_alb_logs" {
  description = "Enable ALB access logs"
  type        = bool
  default     = true
}

variable "alb_ssl_policy" {
  description = "SSL policy for ALB HTTPS listeners"
  type        = string
  default     = "ELBSecurityPolicy-TLS-1-2-2017-01"

  validation {
    condition     = can(regex("^ELBSecurityPolicy-", var.alb_ssl_policy))
    error_message = "ALB SSL policy must be a valid AWS SSL policy."
  }
}

variable "alb_health_check_interval" {
  description = "ALB health check interval (seconds)"
  type        = number
  default     = 30

  validation {
    condition     = var.alb_health_check_interval >= 5 && var.alb_health_check_interval <= 300
    error_message = "ALB health check interval must be between 5 and 300 seconds."
  }
}

variable "alb_healthy_threshold" {
  description = "ALB healthy threshold"
  type        = number
  default     = 2

  validation {
    condition     = var.alb_healthy_threshold >= 2 && var.alb_healthy_threshold <= 10
    error_message = "ALB healthy threshold must be between 2 and 10."
  }
}

variable "alb_unhealthy_threshold" {
  description = "ALB unhealthy threshold"
  type        = number
  default     = 2

  validation {
    condition     = var.alb_unhealthy_threshold >= 2 && var.alb_unhealthy_threshold <= 10
    error_message = "ALB unhealthy threshold must be between 2 and 10."
  }
}

variable "alb_health_check_timeout" {
  description = "ALB health check timeout (seconds)"
  type        = number
  default     = 5

  validation {
    condition     = var.alb_health_check_timeout >= 2 && var.alb_health_check_timeout <= 120
    error_message = "ALB health check timeout must be between 2 and 120 seconds."
  }
}

variable "alb_health_check_matcher" {
  description = "ALB health check HTTP status code matcher"
  type        = string
  default     = "200"

  validation {
    condition     = can(regex("^[0-9]{3}(,[0-9]{3})*$|^[0-9]{3}-[0-9]{3}$", var.alb_health_check_matcher))
    error_message = "ALB health check matcher must be HTTP status codes (e.g., 200 or 200-299 or 200,201,202)."
  }
}

variable "alb_deregistration_delay" {
  description = "ALB target deregistration delay (seconds)"
  type        = number
  default     = 30

  validation {
    condition     = var.alb_deregistration_delay >= 0 && var.alb_deregistration_delay <= 3600
    error_message = "ALB deregistration delay must be between 0 and 3600 seconds."
  }
}

################################################################################
# Monitoring and Logging
################################################################################

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "alarm_email_endpoints" {
  description = "Email addresses for alarm notifications"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for email in var.alarm_email_endpoints :
      can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", email))
    ])
    error_message = "All alarm email endpoints must be valid email addresses."
  }
}

variable "cloudwatch_retention_days" {
  description = "CloudWatch Logs retention in days. Note: Production deployments enforce minimum 365 days regardless of this value."
  type        = number
  default     = 365 # Changed default to 365 to match enforced minimum

  validation {
    condition = contains([
      0, 1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1096, 1827, 2192, 2557, 2922, 3288, 3653
    ], var.cloudwatch_retention_days)
    error_message = "CloudWatch retention days must be a valid AWS retention period."
  }
}

################################################################################
# Security and Compliance
################################################################################

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty"
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

  validation {
    condition     = var.secrets_rotation_days >= 1 && var.secrets_rotation_days <= 365
    error_message = "Secrets rotation period must be between 1 and 365 days."
  }
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

  validation {
    condition     = var.waf_rate_limit >= 100 && var.waf_rate_limit <= 2000000000
    error_message = "WAF rate limit must be between 100 and 2,000,000,000."
  }
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

  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 36500
    error_message = "Backup retention must be between 1 and 36500 days."
  }
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

  validation {
    condition     = var.dr_rto_minutes >= 1 && var.dr_rto_minutes <= 1440
    error_message = "RTO must be between 1 and 1440 minutes (24 hours)."
  }
}

variable "dr_rpo_minutes" {
  description = "Recovery Point Objective (minutes)"
  type        = number
  default     = 15

  validation {
    condition     = var.dr_rpo_minutes >= 1 && var.dr_rpo_minutes <= 1440
    error_message = "RPO must be between 1 and 1440 minutes (24 hours)."
  }
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

  validation {
    condition     = var.auto_scaling_min_capacity >= 1 && var.auto_scaling_min_capacity <= 100
    error_message = "Auto-scaling minimum capacity must be between 1 and 100."
  }
}

variable "auto_scaling_max_capacity" {
  description = "Maximum capacity for auto-scaling"
  type        = number
  default     = 10

  validation {
    condition     = var.auto_scaling_max_capacity >= 1 && var.auto_scaling_max_capacity <= 1000
    error_message = "Auto-scaling maximum capacity must be between 1 and 1000."
  }
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

  validation {
    condition = var.data_residency_region == "" || contains([
      "us-east-1", "us-east-2", "us-west-1", "us-west-2",
      "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-north-1",
      "ap-southeast-1", "ap-southeast-2", "ap-northeast-1", "ap-northeast-2",
      "ap-south-1", "sa-east-1", "ca-central-1"
    ], var.data_residency_region)
    error_message = "Data residency region must be empty or a valid AWS region."
  }
}

################################################################################
# Resource Tagging
################################################################################

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}

  validation {
    condition     = length(var.additional_tags) <= 50
    error_message = "Maximum of 50 additional tags allowed."
  }
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

  validation {
    condition     = var.custom_kms_key_policy == "" || can(jsondecode(var.custom_kms_key_policy))
    error_message = "Custom KMS key policy must be empty or valid JSON."
  }
}

variable "custom_iam_policies" {
  description = "Custom IAM policies to attach"
  type        = list(string)
  default     = []

  validation {
    condition     = length(var.custom_iam_policies) <= 10
    error_message = "Maximum of 10 custom IAM policies allowed."
  }
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
