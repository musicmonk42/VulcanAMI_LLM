################################################################################
# VulcanAMI Infrastructure - Terraform Main Configuration
# Version: 4.6.0
# Description: Complete infrastructure as code for VulcanAMI platform
################################################################################

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }

  # Backend configuration (uncomment and configure for remote state)
  # backend "s3" {
  #   bucket         = "vulcanami-terraform-state"
  #   key            = "infrastructure/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "vulcanami-terraform-locks"
  #   kms_key_id     = "alias/terraform-state"
  # }
}

################################################################################
# Provider Configuration
################################################################################

provider "aws" {
  region = var.primary_region

  default_tags {
    tags = merge(local.common_tags, {
      "Terraform"   = "true"
      "Environment" = var.environment
    })
  }
}

provider "aws" {
  alias  = "secondary"
  region = var.secondary_region

  default_tags {
    tags = merge(local.common_tags, {
      "Terraform"   = "true"
      "Environment" = var.environment
      "Region"      = "secondary"
    })
  }
}

################################################################################
# Data Sources
################################################################################

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}

################################################################################
# Local Variables
################################################################################

locals {
  # Resource naming
  name_prefix = "${var.project}-${var.environment}"

  # Bucket names
  bucket_primary         = "${var.bucket_prefix}-${var.primary_region}-${var.environment}"
  bucket_secondary       = "${var.bucket_prefix}-${var.secondary_region}-${var.environment}"
  bucket_logs            = "${var.bucket_prefix}-logs-${var.primary_region}-${var.environment}"
  bucket_cloudfront_logs = "${var.bucket_prefix}-cf-logs-${var.primary_region}-${var.environment}"
  bucket_backups         = "${var.bucket_prefix}-backups-${var.primary_region}-${var.environment}"

  # Network configuration
  azs              = slice(data.aws_availability_zones.available.names, 0, var.availability_zones)
  private_subnets  = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k)]
  public_subnets   = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k + 10)]
  database_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k + 20)]

  # Common tags
  common_tags = merge(
    {
      "Project"     = var.project
      "Environment" = var.environment
      "Version"     = var.vulcanami_version
      "Owner"       = var.owner
      "CostCenter"  = var.cost_center
      "ManagedBy"   = "Terraform"
      "Contact"     = var.contact_email
      "LastUpdated" = timestamp()
    },
    var.tag_compliance,
    var.additional_tags
  )
}

################################################################################
# VPC and Networking
################################################################################

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = var.enable_dns_hostnames
  enable_dns_support   = var.enable_dns_support

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc"
  })
}

# Restrict default security group to prevent accidental usage
resource "aws_default_security_group" "main" {
  vpc_id = aws_vpc.main.id

  # Restrict ingress to self only
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "Restrict default SG ingress to self"
  }

  # Restrict egress to self only
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "Restrict default SG egress to self"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-default-sg-restricted"
  })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-igw"
  })
}

resource "aws_subnet" "public" {
  count                   = length(local.public_subnets)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnets[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-${local.azs[count.index]}"
    Type = "public"
  })
}

resource "aws_subnet" "private" {
  count             = length(local.private_subnets)
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_subnets[count.index]
  availability_zone = local.azs[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-private-${local.azs[count.index]}"
    Type = "private"
  })
}

resource "aws_subnet" "database" {
  count             = length(local.database_subnets)
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.database_subnets[count.index]
  availability_zone = local.azs[count.index]

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-database-${local.azs[count.index]}"
    Type = "database"
  })
}

resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(local.azs)) : 0
  domain = "vpc"

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "main" {
  count         = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(local.azs)) : 0
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-nat-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-public-rt"
  })
}

resource "aws_route" "public_internet_gateway" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.main.id
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  count  = var.single_nat_gateway ? 1 : length(local.azs)
  vpc_id = aws_vpc.main.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-private-rt-${count.index + 1}"
  })
}

resource "aws_route" "private_nat_gateway" {
  count                  = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(local.azs)) : 0
  route_table_id         = aws_route_table.private[count.index].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.main[count.index].id
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[var.single_nat_gateway ? 0 : count.index].id
}

# VPC Flow Logs
resource "aws_flow_log" "main" {
  count                    = var.enable_vpc_flow_logs ? 1 : 0
  iam_role_arn             = aws_iam_role.vpc_flow_logs[0].arn
  log_destination          = aws_cloudwatch_log_group.vpc_flow_logs[0].arn
  traffic_type             = "ALL"
  vpc_id                   = aws_vpc.main.id
  max_aggregation_interval = 600

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc-flow-logs"
  })
}

resource "aws_cloudwatch_log_group" "vpc_flow_logs" {
  count             = var.enable_vpc_flow_logs ? 1 : 0
  name              = "/aws/vpc/${local.name_prefix}"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = local.common_tags
}

################################################################################
# Security Groups
################################################################################

resource "aws_security_group" "alb" {
  name_description = "${local.name_prefix}-alb-sg"
  vpc_id           = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb-sg"
  })
}

resource "aws_security_group" "ecs_tasks" {
  name_description = "${local.name_prefix}-ecs-tasks-sg"
  vpc_id           = aws_vpc.main.id

  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecs-tasks-sg"
  })
}

resource "aws_security_group" "rds" {
  count            = var.enable_rds ? 1 : 0
  name_description = "${local.name_prefix}-rds-sg"
  vpc_id           = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-sg"
  })
}

resource "aws_security_group" "redis" {
  count            = var.enable_redis ? 1 : 0
  name_description = "${local.name_prefix}-redis-sg"
  vpc_id           = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-sg"
  })
}

################################################################################
# KMS Keys
################################################################################

resource "aws_kms_key" "main" {
  description             = "${local.name_prefix} main encryption key"
  deletion_window_in_days = 10
  enable_key_rotation     = var.kms_key_rotation
  multi_region            = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-main-kms"
  })
}

resource "aws_kms_alias" "main" {
  name          = "alias/${local.name_prefix}-main"
  target_key_id = aws_kms_key.main.key_id
}

resource "aws_kms_key" "s3" {
  description             = "${local.name_prefix} S3 encryption key"
  deletion_window_in_days = 10
  enable_key_rotation     = var.kms_key_rotation

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-s3-kms"
  })
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${local.name_prefix}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

resource "aws_kms_key" "rds" {
  count                   = var.enable_rds ? 1 : 0
  description             = "${local.name_prefix} RDS encryption key"
  deletion_window_in_days = 10
  enable_key_rotation     = var.kms_key_rotation

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-kms"
  })
}

resource "aws_kms_alias" "rds" {
  count         = var.enable_rds ? 1 : 0
  name          = "alias/${local.name_prefix}-rds"
  target_key_id = aws_kms_key.rds[0].key_id
}

################################################################################
# S3 Buckets - Primary
################################################################################

resource "aws_s3_bucket" "primary" {
  bucket              = local.bucket_primary
  force_destroy       = false
  object_lock_enabled = var.enable_object_lock

  tags = merge(local.common_tags, {
    Name   = local.bucket_primary
    Tier   = "primary"
    Region = var.primary_region
  })
}

resource "aws_s3_bucket_versioning" "primary" {
  bucket = aws_s3_bucket.primary.id

  versioning_configuration {
    status = var.enable_bucket_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "primary" {
  count  = var.enable_bucket_encryption ? 1 : 0
  bucket = aws_s3_bucket.primary.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "primary" {
  bucket = aws_s3_bucket.primary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "primary" {
  count  = var.bucket_lifecycle_enabled ? 1 : 0
  bucket = aws_s3_bucket.primary.id

  rule {
    id     = "glacier_transition"
    status = "Enabled"

    filter {}

    transition {
      days          = var.lifecycle_transition_glacier_days
      storage_class = "GLACIER"
    }

    transition {
      days          = var.lifecycle_transition_deep_archive_days
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = var.lifecycle_expiration_days > 0 ? var.lifecycle_expiration_days : null
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

resource "aws_s3_bucket_object_lock_configuration" "primary" {
  count  = var.enable_object_lock ? 1 : 0
  bucket = aws_s3_bucket.primary.id

  rule {
    default_retention {
      mode = "COMPLIANCE"
      days = var.object_lock_retention_days
    }
  }
}

################################################################################
# S3 Buckets - Secondary
################################################################################

resource "aws_s3_bucket" "secondary" {
  provider            = aws.secondary
  bucket              = local.bucket_secondary
  force_destroy       = false
  object_lock_enabled = var.enable_object_lock

  tags = merge(local.common_tags, {
    Name   = local.bucket_secondary
    Tier   = "secondary"
    Region = var.secondary_region
  })
}

resource "aws_s3_bucket_versioning" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  versioning_configuration {
    status = var.enable_bucket_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secondary" {
  count    = var.enable_bucket_encryption ? 1 : 0
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/secondary/"
}

################################################################################
# S3 Bucket - Logs
################################################################################

resource "aws_s3_bucket" "logs" {
  bucket        = local.bucket_logs
  force_destroy = false

  tags = merge(local.common_tags, {
    Name = local.bucket_logs
    Type = "logs"
  })
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "log_retention"
    status = "Enabled"

    filter {}

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = var.lifecycle_expiration_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_logging" "primary" {
  count  = var.enable_bucket_logging ? 1 : 0
  bucket = aws_s3_bucket.primary.id

  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/primary/"
}

################################################################################
# S3 Replication
################################################################################

data "aws_iam_policy_document" "replication_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "replication" {
  statement {
    actions = [
      "s3:GetReplicationConfiguration",
      "s3:ListBucket"
    ]

    resources = [aws_s3_bucket.primary.arn]
  }

  statement {
    actions = [
      "s3:GetObjectVersionForReplication",
      "s3:GetObjectVersionAcl",
      "s3:GetObjectVersionTagging"
    ]

    resources = ["${aws_s3_bucket.primary.arn}/*"]
  }

  statement {
    actions = [
      "s3:ReplicateObject",
      "s3:ReplicateDelete",
      "s3:ReplicateTags"
    ]

    resources = ["${aws_s3_bucket.secondary.arn}/*"]
  }
}

resource "aws_iam_role" "replication" {
  count              = var.enable_bucket_replication ? 1 : 0
  name               = "${local.name_prefix}-s3-replication"
  assume_role_policy = data.aws_iam_policy_document.replication_assume.json

  tags = local.common_tags
}

resource "aws_iam_policy" "replication" {
  count  = var.enable_bucket_replication ? 1 : 0
  name   = "${local.name_prefix}-s3-replication"
  policy = data.aws_iam_policy_document.replication.json
}

resource "aws_iam_role_policy_attachment" "replication" {
  count      = var.enable_bucket_replication ? 1 : 0
  role       = aws_iam_role.replication[0].name
  policy_arn = aws_iam_policy.replication[0].arn
}

resource "aws_s3_bucket_replication_configuration" "primary_to_secondary" {
  count = var.enable_bucket_replication ? 1 : 0

  depends_on = [
    aws_s3_bucket_versioning.primary,
    aws_s3_bucket_versioning.secondary
  ]

  bucket = aws_s3_bucket.primary.id
  role   = aws_iam_role.replication[0].arn

  rule {
    id     = "replicate-all"
    status = "Enabled"

    filter {}

    destination {
      bucket        = aws_s3_bucket.secondary.arn
      storage_class = var.replication_storage_class

      replication_time {
        status = "Enabled"
        time {
          minutes = 15
        }
      }

      metrics {
        status = "Enabled"
        event_threshold {
          minutes = 15
        }
      }
    }

    delete_marker_replication {
      status = "Enabled"
    }
  }
}

################################################################################
# CloudFront Distribution
################################################################################

resource "aws_s3_bucket" "cloudfront_logs" {
  count         = var.enable_cloudfront_logging ? 1 : 0
  bucket        = local.bucket_cloudfront_logs
  force_destroy = false

  tags = merge(local.common_tags, {
    Name = local.bucket_cloudfront_logs
    Type = "cloudfront-logs"
  })
}

resource "aws_s3_bucket_public_access_block" "cloudfront_logs" {
  count  = var.enable_cloudfront_logging ? 1 : 0
  bucket = aws_s3_bucket.cloudfront_logs[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "cloudfront_logs" {
  count  = var.enable_cloudfront_logging ? 1 : 0
  bucket = aws_s3_bucket.cloudfront_logs[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudfront_logs" {
  count  = var.enable_cloudfront_logging ? 1 : 0
  bucket = aws_s3_bucket.cloudfront_logs[0].id

  rule {
    id     = "expire-old-cloudfront-logs"
    status = "Enabled"

    filter {}

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_logging" "cloudfront_logs" {
  count  = var.enable_cloudfront_logging ? 1 : 0
  bucket = aws_s3_bucket.cloudfront_logs[0].id

  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/cloudfront-logs/"
}

resource "aws_cloudfront_origin_access_control" "oac" {
  count                             = var.enable_cloudfront ? 1 : 0
  name                              = "${local.name_prefix}-oac"
  description                       = "OAC for ${local.name_prefix}"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "cdn" {
  count               = var.enable_cloudfront ? 1 : 0
  enabled             = true
  is_ipv6_enabled     = var.enable_ipv6
  http_version        = var.enable_http3 ? "http3" : (var.enable_http2 ? "http2" : "http1.1")
  comment             = "${local.name_prefix} v${var.vulcanami_version} CDN"
  aliases             = concat([var.domain_name], var.additional_domains)
  default_root_object = ""
  price_class         = var.cloudfront_price_class

  origin {
    domain_name              = aws_s3_bucket.primary.bucket_regional_domain_name
    origin_id                = "s3-origin-primary"
    origin_access_control_id = aws_cloudfront_origin_access_control.oac[0].id

    custom_header {
      name  = "X-Pack-Version"
      value = var.vulcanami_version
    }
  }

  default_cache_behavior {
    target_origin_id       = "s3-origin-primary"
    viewer_protocol_policy = "redirect-to-https"
    compress               = var.enable_cloudfront_compression

    allowed_methods = ["GET", "HEAD", "OPTIONS"]
    cached_methods  = ["GET", "HEAD"]

    forwarded_values {
      query_string = true
      headers = [
        "Range",
        "If-Range",
        "ETag",
        "X-Pack-Merkle",
        "X-Bloom",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
      ]

      cookies {
        forward = "none"
      }
    }

    min_ttl     = var.cloudfront_min_ttl
    default_ttl = var.cloudfront_default_ttl
    max_ttl     = var.cloudfront_max_ttl

    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = aws_lambda_function.edge_auth[0].qualified_arn
      include_body = false
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = var.cloudfront_geo_restriction_type
      locations        = var.cloudfront_geo_restriction_locations
    }
  }

  viewer_certificate {
    acm_certificate_arn            = var.ssl_certificate_arn != "" ? var.ssl_certificate_arn : null
    ssl_support_method             = var.ssl_certificate_arn != "" ? "sni-only" : null
    minimum_protocol_version       = var.minimum_tls_version
    cloudfront_default_certificate = var.ssl_certificate_arn == "" ? true : false
  }

  dynamic "logging_config" {
    for_each = var.enable_cloudfront_logging ? [1] : []
    content {
      include_cookies = false
      bucket          = aws_s3_bucket.cloudfront_logs[0].bucket_domain_name
      prefix          = "cloudfront/"
    }
  }

  custom_error_response {
    error_code            = 403
    error_caching_min_ttl = 60
  }

  custom_error_response {
    error_code            = 404
    error_caching_min_ttl = 60
  }

  custom_error_response {
    error_code            = 410
    error_caching_min_ttl = 180
  }

  web_acl_id = var.enable_cloudfront_waf ? aws_wafv2_web_acl.cloudfront[0].arn : null

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-cdn"
  })
}

################################################################################
# RDS PostgreSQL Database
################################################################################

resource "aws_db_subnet_group" "main" {
  count      = var.enable_rds ? 1 : 0
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.database[*].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "random_password" "rds_password" {
  count   = var.enable_rds ? 1 : 0
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "rds_password" {
  count                   = var.enable_rds ? 1 : 0
  name                    = "${local.name_prefix}-rds-password"
  description             = "RDS master password"
  recovery_window_in_days = 7

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  count         = var.enable_rds ? 1 : 0
  secret_id     = aws_secretsmanager_secret.rds_password[0].id
  secret_string = random_password.rds_password[0].result
}

resource "aws_db_instance" "main" {
  count                 = var.enable_rds ? 1 : 0
  identifier            = "${local.name_prefix}-db"
  engine                = "postgres"
  engine_version        = var.rds_engine_version
  instance_class        = var.rds_instance_class
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds[0].arn

  db_name  = "vulcanami"
  username = "admin"
  password = random_password.rds_password[0].result
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.main[0].name
  vpc_security_group_ids = [aws_security_group.rds[0].id]
  publicly_accessible    = false
  multi_az               = var.rds_multi_az

  backup_retention_period    = var.rds_backup_retention_days
  backup_window              = var.rds_backup_window
  maintenance_window         = var.rds_maintenance_window
  auto_minor_version_upgrade = true
  deletion_protection        = var.rds_deletion_protection
  skip_final_snapshot        = false
  final_snapshot_identifier  = "${local.name_prefix}-db-final-snapshot"
  copy_tags_to_snapshot      = true

  performance_insights_enabled    = var.rds_performance_insights_enabled
  performance_insights_kms_key_id = aws_kms_key.rds[0].arn
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  monitoring_interval             = var.rds_monitoring_interval
  monitoring_role_arn             = var.rds_monitoring_interval > 0 ? aws_iam_role.rds_monitoring[0].arn : null

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db"
  })
}

resource "aws_iam_role" "rds_monitoring" {
  count = var.enable_rds && var.rds_monitoring_interval > 0 ? 1 : 0
  name  = "${local.name_prefix}-rds-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "monitoring.rds.amazonaws.com"
      }
    }]
  })

  managed_policy_arns = ["arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"]

  tags = local.common_tags
}

################################################################################
# ElastiCache Redis Cluster
################################################################################

resource "aws_elasticache_subnet_group" "main" {
  count      = var.enable_redis ? 1 : 0
  name       = "${local.name_prefix}-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = local.common_tags
}

resource "aws_elasticache_parameter_group" "main" {
  count  = var.enable_redis ? 1 : 0
  name   = "${local.name_prefix}-redis-params"
  family = var.redis_parameter_group_family

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }

  tags = local.common_tags
}

resource "aws_elasticache_replication_group" "main" {
  count                      = var.enable_redis ? 1 : 0
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for ${local.name_prefix}"
  engine                     = "redis"
  engine_version             = var.redis_engine_version
  node_type                  = var.redis_node_type
  num_cache_clusters         = var.redis_num_cache_nodes
  parameter_group_name       = aws_elasticache_parameter_group.main[0].name
  subnet_group_name          = aws_elasticache_subnet_group.main[0].name
  security_group_ids         = [aws_security_group.redis[0].id]
  port                       = 6379
  multi_az_enabled           = var.redis_num_cache_nodes > 1
  automatic_failover_enabled = var.redis_num_cache_nodes > 1
  at_rest_encryption_enabled = var.redis_at_rest_encryption_enabled
  transit_encryption_enabled = var.redis_transit_encryption_enabled
  auth_token_enabled         = var.redis_transit_encryption_enabled
  snapshot_retention_limit   = var.redis_snapshot_retention_limit
  snapshot_window            = var.redis_snapshot_window
  maintenance_window         = var.redis_maintenance_window
  apply_immediately          = false
  auto_minor_version_upgrade = true

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis[0].name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis"
  })
}

resource "aws_cloudwatch_log_group" "redis" {
  count             = var.enable_redis ? 1 : 0
  name              = "/aws/elasticache/${local.name_prefix}-redis"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = local.common_tags
}

################################################################################
# ECS Cluster and Services
################################################################################

resource "aws_ecs_cluster" "main" {
  count = var.enable_ecs ? 1 : 0
  name  = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = var.enable_container_insights ? "enabled" : "disabled"
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  count              = var.enable_ecs ? 1 : 0
  cluster_name       = aws_ecs_cluster.main[0].name
  capacity_providers = var.ecs_launch_type == "FARGATE" ? ["FARGATE", "FARGATE_SPOT"] : []

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
}

# DQS Service
resource "aws_ecs_task_definition" "dqs" {
  count                    = var.enable_ecs ? 1 : 0
  family                   = "${local.name_prefix}-dqs"
  network_mode             = "awsvpc"
  requires_compatibilities = [var.ecs_launch_type]
  cpu                      = var.ecs_dqs_task_cpu
  memory                   = var.ecs_dqs_task_memory
  execution_role_arn       = aws_iam_role.ecs_execution[0].arn
  task_role_arn            = aws_iam_role.ecs_task[0].arn

  container_definitions = jsonencode([{
    name  = "dqs"
    image = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.primary_region}.amazonaws.com/${local.name_prefix}-dqs:${var.vulcanami_version}"

    essential = true

    portMappings = [{
      containerPort = 8080
      protocol      = "tcp"
    }]

    environment = [
      { name = "POSTGRES_HOST", value = var.enable_rds ? aws_db_instance.main[0].address : "" },
      { name = "REDIS_HOST", value = var.enable_redis ? aws_elasticache_replication_group.main[0].primary_endpoint_address : "" },
      { name = "ENVIRONMENT", value = var.environment },
      { name = "VERSION", value = var.vulcanami_version }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_dqs[0].name
        "awslogs-region"        = var.primary_region
        "awslogs-stream-prefix" = "dqs"
      }
    }
  }])

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "ecs_dqs" {
  count             = var.enable_ecs ? 1 : 0
  name              = "/ecs/${local.name_prefix}/dqs"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = local.common_tags
}

resource "aws_ecs_service" "dqs" {
  count           = var.enable_ecs ? 1 : 0
  name            = "${local.name_prefix}-dqs"
  cluster         = aws_ecs_cluster.main[0].id
  task_definition = aws_ecs_task_definition.dqs[0].arn
  desired_count   = var.ecs_dqs_desired_count
  launch_type     = var.ecs_launch_type

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.dqs[0].arn
    container_name   = "dqs"
    container_port   = 8080
  }

  enable_execute_command = var.enable_ecs_exec

  tags = local.common_tags
}

# OPA Service
resource "aws_ecs_task_definition" "opa" {
  count                    = var.enable_ecs ? 1 : 0
  family                   = "${local.name_prefix}-opa"
  network_mode             = "awsvpc"
  requires_compatibilities = [var.ecs_launch_type]
  cpu                      = var.ecs_opa_task_cpu
  memory                   = var.ecs_opa_task_memory
  execution_role_arn       = aws_iam_role.ecs_execution[0].arn
  task_role_arn            = aws_iam_role.ecs_task[0].arn

  container_definitions = jsonencode([{
    name  = "opa"
    image = "openpolicyagent/opa:0.65.0"

    essential = true

    command = ["run", "--server", "--addr=0.0.0.0:8181"]

    portMappings = [{
      containerPort = 8181
      protocol      = "tcp"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_opa[0].name
        "awslogs-region"        = var.primary_region
        "awslogs-stream-prefix" = "opa"
      }
    }
  }])

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "ecs_opa" {
  count             = var.enable_ecs ? 1 : 0
  name              = "/ecs/${local.name_prefix}/opa"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = local.common_tags
}

resource "aws_ecs_service" "opa" {
  count           = var.enable_ecs ? 1 : 0
  name            = "${local.name_prefix}-opa"
  cluster         = aws_ecs_cluster.main[0].id
  task_definition = aws_ecs_task_definition.opa[0].arn
  desired_count   = var.ecs_opa_desired_count
  launch_type     = var.ecs_launch_type

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.opa[0].arn
    container_name   = "opa"
    container_port   = 8181
  }

  enable_execute_command = var.enable_ecs_exec

  tags = local.common_tags
}

################################################################################
# Application Load Balancer
################################################################################

resource "aws_lb" "main" {
  count              = var.enable_ecs ? 1 : 0
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.environment == "production" ? true : false
  enable_http2               = true

  access_logs {
    bucket  = aws_s3_bucket.logs.id
    prefix  = "alb-access-logs"
    enabled = true
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb"
  })
}

resource "aws_lb_target_group" "dqs" {
  count       = var.enable_ecs ? 1 : 0
  name        = "${local.name_prefix}-dqs-tg"
  port        = 8080
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  tags = local.common_tags
}

resource "aws_lb_target_group" "opa" {
  count       = var.enable_ecs ? 1 : 0
  name        = "${local.name_prefix}-opa-tg"
  port        = 8181
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 10
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "https" {
  count             = var.enable_ecs ? 1 : 0
  load_balancer_arn = aws_lb.main[0].arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.ssl_certificate_arn

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "Not Found"
      status_code  = "404"
    }
  }
}

resource "aws_lb_listener_rule" "dqs" {
  count        = var.enable_ecs ? 1 : 0
  listener_arn = aws_lb_listener.https[0].arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dqs[0].arn
  }

  condition {
    path_pattern {
      values = ["/dqs/*"]
    }
  }
}

resource "aws_lb_listener_rule" "opa" {
  count        = var.enable_ecs ? 1 : 0
  listener_arn = aws_lb_listener.https[0].arn
  priority     = 200

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.opa[0].arn
  }

  condition {
    path_pattern {
      values = ["/opa/*"]
    }
  }
}

################################################################################
# IAM Roles for ECS
################################################################################

resource "aws_iam_role" "ecs_execution" {
  count = var.enable_ecs ? 1 : 0
  name  = "${local.name_prefix}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
  ]

  tags = local.common_tags
}

resource "aws_iam_role" "ecs_task" {
  count = var.enable_ecs ? 1 : 0
  name  = "${local.name_prefix}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "ecs_task_s3" {
  count = var.enable_ecs ? 1 : 0
  name  = "s3-access"
  role  = aws_iam_role.ecs_task[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.primary.arn,
        "${aws_s3_bucket.primary.arn}/*"
      ]
    }]
  })
}

################################################################################
# IAM Role for VPC Flow Logs
################################################################################

data "aws_iam_policy_document" "vpc_flow_logs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["vpc-flow-logs.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "vpc_flow_logs" {
  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role" "vpc_flow_logs" {
  count              = var.enable_vpc_flow_logs ? 1 : 0
  name               = "${local.name_prefix}-vpc-flow-logs"
  assume_role_policy = data.aws_iam_policy_document.vpc_flow_logs_assume.json

  tags = local.common_tags
}

resource "aws_iam_role_policy" "vpc_flow_logs" {
  count  = var.enable_vpc_flow_logs ? 1 : 0
  name   = "vpc-flow-logs-policy"
  role   = aws_iam_role.vpc_flow_logs[0].id
  policy = data.aws_iam_policy_document.vpc_flow_logs.json
}

################################################################################
# Lambda Functions
################################################################################

# Lambda@Edge for CloudFront authentication
resource "aws_iam_role" "lambda_edge" {
  count = var.enable_cloudfront ? 1 : 0
  name  = "${local.name_prefix}-lambda-edge"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = [
          "lambda.amazonaws.com",
          "edgelambda.amazonaws.com"
        ]
      }
    }]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  ]

  tags = local.common_tags
}

resource "aws_lambda_function" "edge_auth" {
  count            = var.enable_cloudfront ? 1 : 0
  filename         = "${path.module}/lambda/edge-auth.zip"
  function_name    = "${local.name_prefix}-edge-auth"
  role             = aws_iam_role.lambda_edge[0].arn
  handler          = "index.handler"
  source_code_hash = filebase64sha256("${path.module}/lambda/edge-auth.zip")
  runtime          = "nodejs18.x"
  publish          = true

  tags = local.common_tags
}

################################################################################
# WAF for CloudFront
################################################################################

resource "aws_wafv2_web_acl" "cloudfront" {
  count       = var.enable_cloudfront_waf ? 1 : 0
  name        = "${local.name_prefix}-cloudfront-waf"
  description = "WAF for CloudFront distribution"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name_prefix}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.name_prefix}-waf"
    sampled_requests_enabled   = true
  }

  tags = local.common_tags
}

################################################################################
# CloudWatch Alarms
################################################################################

resource "aws_sns_topic" "alarms" {
  count = var.enable_cloudwatch_alarms && length(var.alarm_email_endpoints) > 0 ? 1 : 0
  name  = "${local.name_prefix}-alarms"

  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "alarm_email" {
  count     = var.enable_cloudwatch_alarms ? length(var.alarm_email_endpoints) : 0
  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email_endpoints[count.index]
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  count               = var.enable_rds && var.enable_cloudwatch_alarms ? 1 : 0
  alarm_name          = "${local.name_prefix}-rds-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "RDS CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main[0].id
  }
}

resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  count               = var.enable_redis && var.enable_cloudwatch_alarms ? 1 : 0
  alarm_name          = "${local.name_prefix}-redis-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = "300"
  statistic           = "Average"
  threshold           = "75"
  alarm_description   = "Redis CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alarms[0].arn]

  dimensions = {
    CacheClusterId = aws_elasticache_replication_group.main[0].id
  }
}

################################################################################
# CloudTrail for Audit Logging
################################################################################

resource "aws_cloudtrail" "main" {
  count                         = var.enable_cloudtrail ? 1 : 0
  name                          = "${local.name_prefix}-trail"
  s3_bucket_name                = aws_s3_bucket.logs.id
  include_global_service_events = true
  is_multi_region_trail         = var.cloudtrail_multi_region
  enable_logging                = true
  enable_log_file_validation    = true

  event_selector {
    read_write_type           = "All"
    include_management_events = true
  }

  tags = local.common_tags
}

################################################################################
# AWS Backup
################################################################################

resource "aws_backup_vault" "main" {
  count = var.enable_aws_backup ? 1 : 0
  name  = "${local.name_prefix}-backup-vault"

  tags = local.common_tags
}

resource "aws_backup_plan" "main" {
  count = var.enable_aws_backup ? 1 : 0
  name  = "${local.name_prefix}-backup-plan"

  rule {
    rule_name         = "daily-backup"
    target_vault_name = aws_backup_vault.main[0].name
    schedule          = var.backup_plan_frequency == "daily" ? "cron(0 2 * * ? *)" : (var.backup_plan_frequency == "weekly" ? "cron(0 2 ? * SUN *)" : "cron(0 2 1 * ? *)")

    lifecycle {
      delete_after = var.backup_retention_days
    }
  }

  tags = local.common_tags
}

resource "aws_iam_role" "backup" {
  count = var.enable_aws_backup ? 1 : 0
  name  = "${local.name_prefix}-backup"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "backup.amazonaws.com"
      }
    }]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup",
    "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForRestores"
  ]

  tags = local.common_tags
}

resource "aws_backup_selection" "main" {
  count        = var.enable_aws_backup ? 1 : 0
  name         = "${local.name_prefix}-backup-selection"
  plan_id      = aws_backup_plan.main[0].id
  iam_role_arn = aws_iam_role.backup[0].arn

  resources = compact([
    var.enable_rds ? aws_db_instance.main[0].arn : ""
  ])
}

################################################################################
# End of Main Configuration
################################################################################
