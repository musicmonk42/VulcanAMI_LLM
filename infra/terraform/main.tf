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
  azs             = slice(data.aws_availability_zones.available.names, 0, var.availability_zones)
  private_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k + 10)]
  database_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 4, k + 20)]
  
  # Common tags
  common_tags = merge(
    {
      "Project"        = var.project
      "Environment"    = var.environment
      "Version"        = var.version
      "Owner"          = var.owner
      "CostCenter"     = var.cost_center
      "ManagedBy"      = "Terraform"
      "Contact"        = var.contact_email
      "LastUpdated"    = timestamp()
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
  map_public_ip_on_launch = false  # Fixed: Don't auto-assign public IPs

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
  count                = var.enable_vpc_flow_logs ? 1 : 0
  iam_role_arn         = aws_iam_role.vpc_flow_logs[0].arn
  log_destination      = aws_cloudwatch_log_group.vpc_flow_logs[0].arn
  traffic_type         = "ALL"
  vpc_id               = aws_vpc.main.id
  max_aggregation_interval = 600

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc-flow-logs"
  })
}

# KMS Key for CloudWatch Logs
resource "aws_kms_key" "logs" {
  description             = "KMS key for CloudWatch Logs encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = var.kms_key_rotation

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-logs-kms"
    Purpose = "CloudWatch Logs Encryption"
  })
}

resource "aws_kms_alias" "logs" {
  name          = "alias/${local.name_prefix}-logs"
  target_key_id = aws_kms_key.logs.key_id
}

resource "aws_cloudwatch_log_group" "vpc_flow_logs" {
  count             = var.enable_vpc_flow_logs ? 1 : 0
  name              = "${local.name_prefix}-vpc-flow-logs"
  retention_in_days = max(var.cloudwatch_retention_days, 365)  # Fixed: Minimum 1 year retention
  kms_key_id        = aws_kms_key.logs.arn  # Fixed: Added KMS encryption

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-vpc-flow-logs"
  })
}

################################################################################
# Security Groups
################################################################################

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "Security group for Application Load Balancer"  # Fixed: Added description
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS from allowed IPs"  # Fixed: Added description
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_ip_ranges  # Fixed: Using configured IP ranges instead of 0.0.0.0/0
  }

  ingress {
    description = "HTTP from allowed IPs for redirect"  # Fixed: Added description
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_ip_ranges  # Fixed: Using configured IP ranges
  }

  egress {
    description = "Allow all outbound traffic"  # Fixed: Added description
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
  name        = "${local.name_prefix}-ecs-tasks-sg"
  description = "Security group for ECS tasks"  # Fixed: Added description
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Traffic from ALB"  # Fixed: Added description
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]  # Fixed: Only from ALB, not from anywhere
  }

  egress {
    description = "Allow all outbound traffic"  # Fixed: Added description
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
  count       = var.enable_rds ? 1 : 0
  name        = "${local.name_prefix}-rds-sg"
  description = "Security group for RDS database"  # Fixed: Added description
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from ECS tasks"  # Fixed: Added description
    from_port       = var.rds_port
    to_port         = var.rds_port
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-sg"
  })
}

resource "aws_security_group" "redis" {
  count       = var.enable_redis ? 1 : 0
  name        = "${local.name_prefix}-redis-sg"
  description = "Security group for Redis ElastiCache"  # Fixed: Added description
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis from ECS tasks"  # Fixed: Added description
    from_port       = var.redis_port
    to_port         = var.redis_port
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
  description             = "Main KMS key for ${local.name_prefix}"
  deletion_window_in_days = 30
  enable_key_rotation     = var.kms_key_rotation

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-main-kms"
  })
}

resource "aws_kms_alias" "main" {
  name          = "alias/${local.name_prefix}-main"
  target_key_id = aws_kms_key.main.key_id
}

resource "aws_kms_key" "s3" {
  description             = "KMS key for S3 encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = var.kms_key_rotation

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-s3-kms"
  })
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${local.name_prefix}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

################################################################################
# S3 Buckets
################################################################################

resource "aws_s3_bucket" "primary" {
  bucket              = local.bucket_primary
  force_destroy       = var.environment == "dev" ? true : false
  object_lock_enabled = var.enable_object_lock

  tags = merge(local.common_tags, {
    Name = local.bucket_primary
  })
}

# Fixed: Added public access block for primary bucket
resource "aws_s3_bucket_public_access_block" "primary" {
  bucket = aws_s3_bucket.primary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "primary" {
  bucket = aws_s3_bucket.primary.id

  versioning_configuration {
    status = var.enable_bucket_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "primary" {
  bucket = aws_s3_bucket.primary.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn  # Fixed: Using KMS instead of AES256
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_logging" "primary" {
  count = var.enable_bucket_logging ? 1 : 0

  bucket = aws_s3_bucket.primary.id

  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/${local.bucket_primary}/"
}

resource "aws_s3_bucket_lifecycle_configuration" "primary" {
  count = var.bucket_lifecycle_enabled ? 1 : 0

  bucket = aws_s3_bucket.primary.id

  rule {
    id     = "transition-to-ia"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    transition {
      days          = var.lifecycle_transition_glacier_days
      storage_class = "GLACIER"
    }
  }

  rule {
    id     = "transition-to-deep-archive"
    status = "Enabled"

    transition {
      days          = var.lifecycle_transition_deep_archive_days
      storage_class = "DEEP_ARCHIVE"
    }
  }

  # Fixed: Added abort incomplete multipart upload
  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  dynamic "rule" {
    for_each = var.lifecycle_expiration_days > 0 ? [1] : []
    content {
      id     = "expiration"
      status = "Enabled"

      expiration {
        days = var.lifecycle_expiration_days
      }
    }
  }
}

resource "aws_s3_bucket" "secondary" {
  provider            = aws.secondary
  bucket              = local.bucket_secondary
  force_destroy       = var.environment == "dev" ? true : false
  object_lock_enabled = var.enable_object_lock

  tags = merge(local.common_tags, {
    Name = local.bucket_secondary
  })
}

# Fixed: Added public access block for secondary bucket
resource "aws_s3_bucket_public_access_block" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  versioning_configuration {
    status = var.enable_bucket_versioning ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn  # Fixed: Using KMS
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_logging" "secondary" {
  provider = aws.secondary
  bucket   = aws_s3_bucket.secondary.id

  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/${local.bucket_secondary}/"
}

resource "aws_s3_bucket" "logs" {
  bucket        = local.bucket_logs
  force_destroy = var.environment == "dev" ? true : false

  tags = merge(local.common_tags, {
    Name = local.bucket_logs
  })
}

# Fixed: Added public access block for logs bucket
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

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn  # Fixed: Using KMS
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

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
  }

  # Fixed: Added abort incomplete multipart upload
  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

################################################################################
# S3 Bucket Policies
################################################################################

data "aws_iam_policy_document" "primary_bucket" {
  statement {
    sid = "EnforceSecureTransport"
    effect = "Deny"
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.primary.arn,
      "${aws_s3_bucket.primary.arn}/*"
    ]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "primary" {
  bucket = aws_s3_bucket.primary.id
  policy = data.aws_iam_policy_document.primary_bucket.json
}

################################################################################
# S3 Bucket Replication
################################################################################

resource "aws_iam_role" "replication" {
  count = var.enable_bucket_replication ? 1 : 0
  name  = "${local.name_prefix}-s3-replication"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "s3.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "replication" {
  count = var.enable_bucket_replication ? 1 : 0
  name  = "${local.name_prefix}-s3-replication"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.primary.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = "${aws_s3_bucket.primary.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = "${aws_s3_bucket.secondary.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "replication" {
  count      = var.enable_bucket_replication ? 1 : 0
  role       = aws_iam_role.replication[0].name
  policy_arn = aws_iam_policy.replication[0].arn
}

resource "aws_s3_bucket_replication_configuration" "primary" {
  count = var.enable_bucket_replication ? 1 : 0

  role   = aws_iam_role.replication[0].arn
  bucket = aws_s3_bucket.primary.id

  rule {
    id     = "replicate-all"
    status = "Enabled"

    filter {}

    destination {
      bucket        = aws_s3_bucket.secondary.arn
      storage_class = var.replication_storage_class

      encryption_configuration {
        replica_kms_key_id = aws_kms_key.s3.arn
      }
    }

    delete_marker_replication {
      status = "Enabled"
    }
  }

  depends_on = [aws_s3_bucket_versioning.primary]
}

################################################################################
# CloudFront Distribution
################################################################################

resource "aws_s3_bucket" "cloudfront_logs" {
  count         = var.enable_cloudfront_logging ? 1 : 0
  bucket        = local.bucket_cloudfront_logs
  force_destroy = var.environment == "dev" ? true : false

  tags = merge(local.common_tags, {
    Name = local.bucket_cloudfront_logs
  })
}

# Fixed: Added public access block for CloudFront logs bucket
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

resource "aws_s3_bucket_logging" "cloudfront_logs" {
  count  = var.enable_cloudfront_logging ? 1 : 0
  bucket = aws_s3_bucket.cloudfront_logs[0].id

  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/${local.bucket_cloudfront_logs}/"
}

resource "aws_cloudfront_origin_access_identity" "main" {
  count   = var.enable_cloudfront ? 1 : 0
  comment = "OAI for ${local.name_prefix}"
}

resource "aws_cloudfront_distribution" "cdn" {
  count = var.enable_cloudfront ? 1 : 0

  enabled             = true
  is_ipv6_enabled     = var.enable_ipv6
  comment             = "CloudFront distribution for ${local.name_prefix}"
  default_root_object = "index.html"  # Fixed: Added default root object
  price_class         = var.cloudfront_price_class
  aliases             = concat([var.domain_name], var.additional_domains)
  web_acl_id          = var.enable_cloudfront_waf ? aws_wafv2_web_acl.cloudfront[0].arn : null

  origin {
    domain_name = aws_s3_bucket.primary.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.primary.id}"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main[0].cloudfront_access_identity_path
    }
  }

  # Fixed: Added origin failover group
  origin_group {
    origin_id = "S3-origin-group"

    failover_criteria {
      status_codes = [403, 404, 500, 502, 503, 504]
    }

    member {
      origin_id = "S3-${aws_s3_bucket.primary.id}"
    }

    member {
      origin_id = "S3-${aws_s3_bucket.secondary.id}"
    }
  }

  origin {
    domain_name = aws_s3_bucket.secondary.bucket_regional_domain_name
    origin_id   = "S3-${aws_s3_bucket.secondary.id}"

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.main[0].cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    allowed_methods  = var.cloudfront_allowed_methods
    cached_methods   = var.cloudfront_cached_methods
    target_origin_id = "S3-origin-group"  # Fixed: Using origin group for failover

    forwarded_values {
      query_string = var.cloudfront_forward_query_string

      cookies {
        forward = var.cloudfront_forward_cookies
      }

      headers = var.cloudfront_forward_headers
    }

    viewer_protocol_policy = "redirect-to-https"  # Fixed: Always redirect to HTTPS
    min_ttl                = var.cloudfront_min_ttl
    default_ttl            = var.cloudfront_default_ttl
    max_ttl                = var.cloudfront_max_ttl
    compress               = var.cloudfront_compress

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
    cloudfront_default_certificate = var.acm_certificate_arn == "" ? true : false
    acm_certificate_arn            = var.acm_certificate_arn != "" ? var.acm_certificate_arn : null
    ssl_support_method              = var.acm_certificate_arn != "" ? "sni-only" : null
    minimum_protocol_version        = "TLSv1.2_2021"  # Fixed: Using TLS 1.2 minimum
  }

  logging_config {
    include_cookies = var.cloudfront_log_cookies
    bucket          = var.enable_cloudfront_logging ? aws_s3_bucket.cloudfront_logs[0].bucket_domain_name : ""
    prefix          = "cloudfront-logs/"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-cdn"
  })

  depends_on = [aws_lambda_function.edge_auth]
}

################################################################################
# RDS Database
################################################################################

resource "aws_kms_key" "rds" {
  count                   = var.enable_rds ? 1 : 0
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 30
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

resource "aws_secretsmanager_secret" "rds_password" {
  count                   = var.enable_rds ? 1 : 0
  name                    = "${local.name_prefix}-rds-password"
  description             = "Password for RDS database"
  recovery_window_in_days = 30
  kms_key_id              = aws_kms_key.rds[0].arn  # Fixed: Added KMS encryption for secrets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-rds-password"
  })
}

resource "random_password" "rds" {
  count   = var.enable_rds ? 1 : 0
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  count         = var.enable_rds ? 1 : 0
  secret_id     = aws_secretsmanager_secret.rds_password[0].id
  secret_string = random_password.rds[0].result
}

resource "aws_db_subnet_group" "main" {
  count      = var.enable_rds ? 1 : 0
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.database[*].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-subnet-group"
  })
}

resource "aws_db_instance" "main" {
  count                     = var.enable_rds ? 1 : 0
  identifier                = "${local.name_prefix}-db"
  engine                    = var.rds_engine
  engine_version            = var.rds_engine_version
  instance_class            = var.rds_instance_class
  allocated_storage         = var.rds_allocated_storage
  max_allocated_storage     = var.rds_max_allocated_storage
  storage_type              = var.rds_storage_type
  storage_encrypted         = true
  kms_key_id                = aws_kms_key.rds[0].arn
  db_name                   = var.rds_database_name
  username                  = var.rds_username
  password                  = random_password.rds[0].result
  port                      = var.rds_port
  multi_az                  = var.rds_multi_az
  publicly_accessible       = false
  backup_retention_period   = var.rds_backup_retention_period
  backup_window             = var.rds_backup_window
  maintenance_window        = var.rds_maintenance_window
  auto_minor_version_upgrade = var.rds_auto_minor_version_upgrade
  deletion_protection       = var.rds_deletion_protection
  skip_final_snapshot       = var.environment == "dev" ? true : false
  final_snapshot_identifier = var.environment == "dev" ? null : "${local.name_prefix}-db-final-snapshot-${replace(timestamp(), ":", "-")}"
  copy_tags_to_snapshot     = true
  db_subnet_group_name      = aws_db_subnet_group.main[0].name
  vpc_security_group_ids    = [aws_security_group.rds[0].id]
  enabled_cloudwatch_logs_exports = var.rds_cloudwatch_logs_exports
  iam_database_authentication_enabled = true  # Fixed: Enabled IAM authentication

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db"
  })
}

resource "aws_db_instance" "read_replica" {
  count                     = var.enable_rds && var.rds_create_read_replica ? 1 : 0
  identifier                = "${local.name_prefix}-db-read-replica"
  replicate_source_db       = aws_db_instance.main[0].identifier
  instance_class            = var.rds_instance_class
  publicly_accessible       = false
  auto_minor_version_upgrade = var.rds_auto_minor_version_upgrade
  copy_tags_to_snapshot     = true

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-db-read-replica"
  })
}

################################################################################
# ElastiCache Redis
################################################################################

resource "aws_elasticache_subnet_group" "main" {
  count      = var.enable_redis ? 1 : 0
  name       = "${local.name_prefix}-redis-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-subnet-group"
  })
}

resource "aws_elasticache_parameter_group" "main" {
  count  = var.enable_redis ? 1 : 0
  name   = "${local.name_prefix}-redis-params"
  family = var.redis_family

  dynamic "parameter" {
    for_each = var.redis_parameters
    content {
      name  = parameter.key
      value = parameter.value
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-params"
  })
}

# Fixed: Added KMS key for ElastiCache
resource "aws_kms_key" "redis" {
  count                   = var.enable_redis ? 1 : 0
  description             = "KMS key for Redis encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = var.kms_key_rotation

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-kms"
  })
}

resource "aws_kms_alias" "redis" {
  count         = var.enable_redis ? 1 : 0
  name          = "alias/${local.name_prefix}-redis"
  target_key_id = aws_kms_key.redis[0].key_id
}

# Fixed: Added auth token for Redis
resource "random_password" "redis_auth_token" {
  count   = var.enable_redis ? 1 : 0
  length  = 32
  special = false  # Redis auth tokens don't support special characters
}

resource "aws_secretsmanager_secret" "redis_auth_token" {
  count                   = var.enable_redis ? 1 : 0
  name                    = "${local.name_prefix}-redis-auth-token"
  description             = "Auth token for Redis"
  recovery_window_in_days = 30
  kms_key_id              = aws_kms_key.redis[0].arn

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-auth-token"
  })
}

resource "aws_secretsmanager_secret_version" "redis_auth_token" {
  count         = var.enable_redis ? 1 : 0
  secret_id     = aws_secretsmanager_secret.redis_auth_token[0].id
  secret_string = random_password.redis_auth_token[0].result
}

resource "aws_elasticache_replication_group" "main" {
  count                      = var.enable_redis ? 1 : 0
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for ${local.name_prefix}"
  engine                     = "redis"
  node_type                  = var.redis_node_type
  port                       = var.redis_port
  parameter_group_name       = aws_elasticache_parameter_group.main[0].name
  subnet_group_name          = aws_elasticache_subnet_group.main[0].name
  security_group_ids         = [aws_security_group.redis[0].id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true  # Fixed: Enabled transit encryption
  auth_token                 = random_password.redis_auth_token[0].result  # Fixed: Added auth token
  kms_key_id                 = aws_kms_key.redis[0].arn  # Fixed: Added KMS key
  automatic_failover_enabled = var.redis_automatic_failover
  multi_az_enabled           = var.redis_multi_az
  num_cache_clusters         = var.redis_num_cache_clusters
  snapshot_retention_limit   = var.redis_snapshot_retention_limit
  snapshot_window            = var.redis_snapshot_window
  maintenance_window         = var.redis_maintenance_window
  notification_topic_arn     = var.enable_cloudwatch_alarms && length(var.alarm_email_endpoints) > 0 ? aws_sns_topic.alarms[0].arn : null
  apply_immediately          = var.environment == "dev" ? true : false
  auto_minor_version_upgrade = var.redis_auto_minor_version_upgrade

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
  name              = "${local.name_prefix}-redis-logs"
  retention_in_days = max(var.cloudwatch_retention_days, 365)  # Fixed: Minimum 1 year retention
  kms_key_id        = aws_kms_key.logs.arn  # Fixed: Added KMS encryption

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-logs"
  })
}

################################################################################
# ECS Cluster
################################################################################

resource "aws_ecs_cluster" "main" {
  count = var.enable_ecs ? 1 : 0
  name  = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = var.ecs_container_insights ? "enabled" : "disabled"
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-cluster"
  })
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  count        = var.enable_ecs ? 1 : 0
  cluster_name = aws_ecs_cluster.main[0].name

  capacity_providers = var.ecs_capacity_providers

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = var.ecs_capacity_providers[0]
  }
}

################################################################################
# ECS Task Definitions
################################################################################

resource "aws_iam_role" "ecs_task_execution" {
  count = var.enable_ecs ? 1 : 0
  name  = "${local.name_prefix}-ecs-task-execution"

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

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  count      = var.enable_ecs ? 1 : 0
  role       = aws_iam_role.ecs_task_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
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

resource "aws_cloudwatch_log_group" "ecs_dqs" {
  count             = var.enable_ecs ? 1 : 0
  name              = "/ecs/${local.name_prefix}/dqs"
  retention_in_days = max(var.cloudwatch_retention_days, 365)  # Fixed: Minimum 1 year retention
  kms_key_id        = aws_kms_key.logs.arn  # Fixed: Added KMS encryption

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecs-dqs-logs"
  })
}

resource "aws_ecs_task_definition" "dqs" {
  count                    = var.enable_ecs ? 1 : 0
  family                   = "${local.name_prefix}-dqs"
  network_mode             = "awsvpc"
  requires_compatibilities = var.ecs_launch_type == "FARGATE" ? ["FARGATE"] : ["EC2"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution[0].arn
  task_role_arn            = aws_iam_role.ecs_task[0].arn

  container_definitions = jsonencode([{
    name      = "dqs"
    image     = var.dqs_image
    essential = true
    
    portMappings = [{
      containerPort = var.dqs_container_port
      protocol      = "tcp"
    }]

    environment = var.dqs_environment_variables

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_dqs[0].name
        "awslogs-region"        = var.primary_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    readonlyRootFilesystem = true  # Fixed: Read-only root filesystem for security
    
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:${var.dqs_container_port}/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-dqs-task"
  })
}

resource "aws_ecs_task_definition" "opa" {
  count                    = var.enable_ecs ? 1 : 0
  family                   = "${local.name_prefix}-opa"
  network_mode             = "awsvpc"
  requires_compatibilities = var.ecs_launch_type == "FARGATE" ? ["FARGATE"] : ["EC2"]
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution[0].arn
  task_role_arn            = aws_iam_role.ecs_task[0].arn

  container_definitions = jsonencode([{
    name      = "opa"
    image     = var.opa_image
    essential = true
    
    portMappings = [{
      containerPort = var.opa_container_port
      protocol      = "tcp"
    }]

    environment = var.opa_environment_variables

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_opa[0].name
        "awslogs-region"        = var.primary_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    readonlyRootFilesystem = true  # Fixed: Read-only root filesystem for security
    
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:${var.opa_container_port}/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-opa-task"
  })
}

resource "aws_cloudwatch_log_group" "ecs_opa" {
  count             = var.enable_ecs ? 1 : 0
  name              = "/ecs/${local.name_prefix}/opa"
  retention_in_days = max(var.cloudwatch_retention_days, 365)  # Fixed: Minimum 1 year retention
  kms_key_id        = aws_kms_key.logs.arn  # Fixed: Added KMS encryption

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-ecs-opa-logs"
  })
}

################################################################################
# ECS Services
################################################################################

resource "aws_ecs_service" "dqs" {
  count           = var.enable_ecs ? 1 : 0
  name            = "${local.name_prefix}-dqs"
  cluster         = aws_ecs_cluster.main[0].id
  task_definition = aws_ecs_task_definition.dqs[0].arn
  desired_count   = var.ecs_service_desired_count
  launch_type     = var.ecs_launch_type

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.dqs[0].arn
    container_name   = "dqs"
    container_port   = var.dqs_container_port
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-dqs-service"
  })
}

resource "aws_ecs_service" "opa" {
  count           = var.enable_ecs ? 1 : 0
  name            = "${local.name_prefix}-opa"
  cluster         = aws_ecs_cluster.main[0].id
  task_definition = aws_ecs_task_definition.opa[0].arn
  desired_count   = var.ecs_service_desired_count
  launch_type     = var.ecs_launch_type

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.opa[0].arn
    container_name   = "opa"
    container_port   = var.opa_container_port
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-opa-service"
  })
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

  enable_deletion_protection = var.alb_deletion_protection
  enable_http2               = true
  enable_cross_zone_load_balancing = true
  drop_invalid_header_fields = true  # Fixed: Added dropping invalid headers

  access_logs {
    bucket  = aws_s3_bucket.logs.bucket
    prefix  = "alb-logs"
    enabled = var.enable_alb_logs
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-alb"
  })
}

resource "aws_lb_target_group" "dqs" {
  count       = var.enable_ecs ? 1 : 0
  name        = "${local.name_prefix}-dqs-tg"
  port        = var.dqs_container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = var.ecs_launch_type == "FARGATE" ? "ip" : "instance"

  health_check {
    enabled             = true
    interval            = var.alb_health_check_interval
    path                = var.dqs_health_check_path
    protocol            = "HTTP"
    healthy_threshold   = var.alb_healthy_threshold
    unhealthy_threshold = var.alb_unhealthy_threshold
    timeout             = var.alb_health_check_timeout
    matcher             = var.alb_health_check_matcher
  }

  deregistration_delay = var.alb_deregistration_delay

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-dqs-tg"
  })
}

resource "aws_lb_target_group" "opa" {
  count       = var.enable_ecs ? 1 : 0
  name        = "${local.name_prefix}-opa-tg"
  port        = var.opa_container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = var.ecs_launch_type == "FARGATE" ? "ip" : "instance"

  health_check {
    enabled             = true
    interval            = var.alb_health_check_interval
    path                = var.opa_health_check_path
    protocol            = "HTTP"
    healthy_threshold   = var.alb_healthy_threshold
    unhealthy_threshold = var.alb_unhealthy_threshold
    timeout             = var.alb_health_check_timeout
    matcher             = var.alb_health_check_matcher
  }

  deregistration_delay = var.alb_deregistration_delay

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-opa-tg"
  })
}

resource "aws_lb_listener" "https" {
  count             = var.enable_ecs ? 1 : 0
  load_balancer_arn = aws_lb.main[0].arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = var.alb_ssl_policy
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type = "fixed-response"
    
    fixed_response {
      content_type = "text/plain"
      message_body = "404: Not Found"
      status_code  = "404"
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-https-listener"
  })
}

resource "aws_lb_listener" "http" {
  count             = var.enable_ecs ? 1 : 0
  load_balancer_arn = aws_lb.main[0].arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-http-listener"
  })
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
      values = ["/dqs*"]
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-dqs-rule"
  })
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
      values = ["/opa*"]
    }
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-opa-rule"
  })
}

################################################################################
# Auto Scaling
################################################################################

resource "aws_appautoscaling_target" "ecs_dqs" {
  count              = var.enable_ecs && var.enable_auto_scaling ? 1 : 0
  max_capacity       = var.auto_scaling_max_capacity
  min_capacity       = var.auto_scaling_min_capacity
  resource_id        = "service/${aws_ecs_cluster.main[0].name}/${aws_ecs_service.dqs[0].name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_dqs_cpu" {
  count              = var.enable_ecs && var.enable_auto_scaling ? 1 : 0
  name               = "${local.name_prefix}-dqs-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_dqs[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_dqs[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_dqs[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    target_value = var.auto_scaling_target_cpu
  }
}

resource "aws_appautoscaling_target" "ecs_opa" {
  count              = var.enable_ecs && var.enable_auto_scaling ? 1 : 0
  max_capacity       = var.auto_scaling_max_capacity
  min_capacity       = var.auto_scaling_min_capacity
  resource_id        = "service/${aws_ecs_cluster.main[0].name}/${aws_ecs_service.opa[0].name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_opa_cpu" {
  count              = var.enable_ecs && var.enable_auto_scaling ? 1 : 0
  name               = "${local.name_prefix}-opa-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_opa[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_opa[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_opa[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    target_value = var.auto_scaling_target_cpu
  }
}

################################################################################
# IAM Policies and Roles for VPC Flow Logs
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

# Fixed: Added Lambda DLQ for error handling
resource "aws_sqs_queue" "lambda_dlq" {
  count                     = var.enable_cloudfront ? 1 : 0
  name                      = "${local.name_prefix}-lambda-dlq"
  message_retention_seconds = 1209600  # 14 days
  kms_master_key_id         = aws_kms_key.main.arn

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-lambda-dlq"
  })
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
  timeout          = 5
  reserved_concurrent_executions = 100  # Fixed: Added concurrent execution limit

  dead_letter_config {
    target_arn = aws_sqs_queue.lambda_dlq[0].arn  # Fixed: Added DLQ
  }

  environment {
    variables = {
      ENVIRONMENT = var.environment
    }
  }

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
  count             = var.enable_cloudwatch_alarms && length(var.alarm_email_endpoints) > 0 ? 1 : 0
  name              = "${local.name_prefix}-alarms"
  kms_master_key_id = aws_kms_key.main.arn  # Fixed: Added KMS encryption for SNS

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

  tags = local.common_tags
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

  tags = local.common_tags
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
  kms_key_id                    = aws_kms_key.main.arn  # Fixed: Added KMS encryption

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
  count        = var.enable_aws_backup ? 1 : 0
  name         = "${local.name_prefix}-backup-vault"
  kms_key_id   = aws_kms_key.main.arn  # Fixed: Added KMS encryption

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
