################################################################################
# VulcanAMI Infrastructure - Terraform Outputs
# Version: 4.6.0
# Description: Comprehensive outputs for all infrastructure resources
################################################################################

################################################################################
# Project Information
################################################################################

output "project_name" {
  description = "Project name"
  value       = var.project
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "version" {
  description = "VulcanAMI version"
  value       = var.version
}

output "aws_account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "primary_region" {
  description = "Primary AWS region"
  value       = var.primary_region
}

output "secondary_region" {
  description = "Secondary AWS region"
  value       = var.secondary_region
}

################################################################################
# VPC and Networking Outputs
################################################################################

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "vpc_arn" {
  description = "ARN of the VPC"
  value       = aws_vpc.main.arn
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "List of database subnet IDs"
  value       = aws_subnet.database[*].id
}

output "public_subnet_cidrs" {
  description = "CIDR blocks of public subnets"
  value       = aws_subnet.public[*].cidr_block
}

output "private_subnet_cidrs" {
  description = "CIDR blocks of private subnets"
  value       = aws_subnet.private[*].cidr_block
}

output "availability_zones" {
  description = "Availability zones used"
  value       = local.azs
}

output "nat_gateway_ips" {
  description = "Elastic IPs of NAT gateways"
  value       = var.enable_nat_gateway ? aws_eip.nat[*].public_ip : []
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

################################################################################
# Security Group Outputs
################################################################################

output "alb_security_group_id" {
  description = "Security group ID for ALB"
  value       = aws_security_group.alb.id
}

output "ecs_tasks_security_group_id" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "rds_security_group_id" {
  description = "Security group ID for RDS"
  value       = var.enable_rds ? aws_security_group.rds[0].id : null
}

output "redis_security_group_id" {
  description = "Security group ID for Redis"
  value       = var.enable_redis ? aws_security_group.redis[0].id : null
}

################################################################################
# S3 Bucket Outputs
################################################################################

output "primary_bucket_id" {
  description = "Primary S3 bucket ID"
  value       = aws_s3_bucket.primary.id
}

output "primary_bucket_arn" {
  description = "Primary S3 bucket ARN"
  value       = aws_s3_bucket.primary.arn
}

output "primary_bucket_domain_name" {
  description = "Primary S3 bucket domain name"
  value       = aws_s3_bucket.primary.bucket_domain_name
}

output "primary_bucket_regional_domain_name" {
  description = "Primary S3 bucket regional domain name"
  value       = aws_s3_bucket.primary.bucket_regional_domain_name
}

output "secondary_bucket_id" {
  description = "Secondary S3 bucket ID"
  value       = aws_s3_bucket.secondary.id
}

output "secondary_bucket_arn" {
  description = "Secondary S3 bucket ARN"
  value       = aws_s3_bucket.secondary.arn
}

output "secondary_bucket_domain_name" {
  description = "Secondary S3 bucket domain name"
  value       = aws_s3_bucket.secondary.bucket_domain_name
}

output "logs_bucket_id" {
  description = "Logs S3 bucket ID"
  value       = aws_s3_bucket.logs.id
}

output "logs_bucket_arn" {
  description = "Logs S3 bucket ARN"
  value       = aws_s3_bucket.logs.arn
}

output "cloudfront_logs_bucket_id" {
  description = "CloudFront logs S3 bucket ID"
  value       = var.enable_cloudfront_logging ? aws_s3_bucket.cloudfront_logs[0].id : null
}

################################################################################
# KMS Key Outputs
################################################################################

output "main_kms_key_id" {
  description = "Main KMS key ID"
  value       = aws_kms_key.main.id
}

output "main_kms_key_arn" {
  description = "Main KMS key ARN"
  value       = aws_kms_key.main.arn
}

output "s3_kms_key_id" {
  description = "S3 KMS key ID"
  value       = aws_kms_key.s3.id
}

output "s3_kms_key_arn" {
  description = "S3 KMS key ARN"
  value       = aws_kms_key.s3.arn
}

output "rds_kms_key_id" {
  description = "RDS KMS key ID"
  value       = var.enable_rds ? aws_kms_key.rds[0].id : null
}

output "rds_kms_key_arn" {
  description = "RDS KMS key ARN"
  value       = var.enable_rds ? aws_kms_key.rds[0].arn : null
}

################################################################################
# CloudFront Outputs
################################################################################

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.cdn[0].id : null
}

output "cloudfront_distribution_arn" {
  description = "CloudFront distribution ARN"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.cdn[0].arn : null
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.cdn[0].domain_name : null
}

output "cloudfront_hosted_zone_id" {
  description = "CloudFront distribution hosted zone ID"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.cdn[0].hosted_zone_id : null
}

output "cloudfront_status" {
  description = "CloudFront distribution status"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.cdn[0].status : null
}

################################################################################
# RDS Database Outputs
################################################################################

output "rds_instance_id" {
  description = "RDS instance ID"
  value       = var.enable_rds ? aws_db_instance.main[0].id : null
}

output "rds_instance_arn" {
  description = "RDS instance ARN"
  value       = var.enable_rds ? aws_db_instance.main[0].arn : null
}

output "rds_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.main[0].endpoint : null
}

output "rds_instance_address" {
  description = "RDS instance address"
  value       = var.enable_rds ? aws_db_instance.main[0].address : null
}

output "rds_instance_port" {
  description = "RDS instance port"
  value       = var.enable_rds ? aws_db_instance.main[0].port : null
}

output "rds_database_name" {
  description = "RDS database name"
  value       = var.enable_rds ? aws_db_instance.main[0].db_name : null
}

output "rds_username" {
  description = "RDS master username"
  value       = var.enable_rds ? aws_db_instance.main[0].username : null
}

output "rds_password_secret_arn" {
  description = "ARN of the secret containing RDS password"
  value       = var.enable_rds ? aws_secretsmanager_secret.rds_password[0].arn : null
  sensitive   = true
}

output "rds_read_replica_id" {
  description = "RDS read replica instance ID"
  value       = var.enable_rds && var.rds_create_read_replica ? aws_db_instance.read_replica[0].id : null
}

output "rds_read_replica_endpoint" {
  description = "RDS read replica endpoint"
  value       = var.enable_rds && var.rds_create_read_replica ? aws_db_instance.read_replica[0].endpoint : null
}

################################################################################
# ElastiCache Redis Outputs
################################################################################

output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].id : null
}

output "redis_cluster_arn" {
  description = "Redis cluster ARN"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].arn : null
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint address"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].primary_endpoint_address : null
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint address"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].reader_endpoint_address : null
}

output "redis_port" {
  description = "Redis port"
  value       = var.enable_redis ? aws_elasticache_replication_group.main[0].port : null
}

output "redis_auth_token_secret_arn" {
  description = "ARN of the secret containing Redis auth token"
  value       = var.enable_redis ? aws_secretsmanager_secret.redis_auth_token[0].arn : null
  sensitive   = true
}

################################################################################
# ECS Cluster Outputs
################################################################################

output "ecs_cluster_id" {
  description = "ECS cluster ID"
  value       = var.enable_ecs ? aws_ecs_cluster.main[0].id : null
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = var.enable_ecs ? aws_ecs_cluster.main[0].arn : null
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = var.enable_ecs ? aws_ecs_cluster.main[0].name : null
}

output "ecs_dqs_service_id" {
  description = "ECS DQS service ID"
  value       = var.enable_ecs ? aws_ecs_service.dqs[0].id : null
}

output "ecs_dqs_service_name" {
  description = "ECS DQS service name"
  value       = var.enable_ecs ? aws_ecs_service.dqs[0].name : null
}

output "ecs_opa_service_id" {
  description = "ECS OPA service ID"
  value       = var.enable_ecs ? aws_ecs_service.opa[0].id : null
}

output "ecs_opa_service_name" {
  description = "ECS OPA service name"
  value       = var.enable_ecs ? aws_ecs_service.opa[0].name : null
}

################################################################################
# Application Load Balancer Outputs
################################################################################

output "alb_id" {
  description = "ALB ID"
  value       = var.enable_ecs ? aws_lb.main[0].id : null
}

output "alb_arn" {
  description = "ALB ARN"
  value       = var.enable_ecs ? aws_lb.main[0].arn : null
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = var.enable_ecs ? aws_lb.main[0].dns_name : null
}

output "alb_zone_id" {
  description = "ALB zone ID"
  value       = var.enable_ecs ? aws_lb.main[0].zone_id : null
}

output "alb_dqs_target_group_arn" {
  description = "DQS target group ARN"
  value       = var.enable_ecs ? aws_lb_target_group.dqs[0].arn : null
}

output "alb_opa_target_group_arn" {
  description = "OPA target group ARN"
  value       = var.enable_ecs ? aws_lb_target_group.opa[0].arn : null
}

################################################################################
# Auto Scaling Outputs
################################################################################

output "dqs_autoscaling_target_id" {
  description = "DQS auto-scaling target ID"
  value       = var.enable_ecs && var.enable_auto_scaling ? aws_appautoscaling_target.ecs_dqs[0].id : null
}

output "opa_autoscaling_target_id" {
  description = "OPA auto-scaling target ID"
  value       = var.enable_ecs && var.enable_auto_scaling ? aws_appautoscaling_target.ecs_opa[0].id : null
}

################################################################################
# CloudWatch Outputs
################################################################################

output "vpc_flow_logs_log_group_name" {
  description = "VPC Flow Logs log group name"
  value       = var.enable_vpc_flow_logs ? aws_cloudwatch_log_group.vpc_flow_logs[0].name : null
}

output "ecs_dqs_log_group_name" {
  description = "ECS DQS service log group name"
  value       = var.enable_ecs ? aws_cloudwatch_log_group.ecs_dqs[0].name : null
}

output "ecs_opa_log_group_name" {
  description = "ECS OPA service log group name"
  value       = var.enable_ecs ? aws_cloudwatch_log_group.ecs_opa[0].name : null
}

output "redis_log_group_name" {
  description = "Redis log group name"
  value       = var.enable_redis ? aws_cloudwatch_log_group.redis[0].name : null
}

output "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarms"
  value       = var.enable_cloudwatch_alarms && length(var.alarm_email_endpoints) > 0 ? aws_sns_topic.alarms[0].arn : null
}

################################################################################
# Lambda Outputs
################################################################################

output "lambda_edge_auth_function_arn" {
  description = "Lambda@Edge auth function ARN"
  value       = var.enable_cloudfront ? aws_lambda_function.edge_auth[0].arn : null
}

output "lambda_edge_auth_qualified_arn" {
  description = "Lambda@Edge auth function qualified ARN"
  value       = var.enable_cloudfront ? aws_lambda_function.edge_auth[0].qualified_arn : null
}

output "lambda_dlq_url" {
  description = "Lambda Dead Letter Queue URL"
  value       = var.enable_cloudfront ? aws_sqs_queue.lambda_dlq[0].url : null
}

################################################################################
# WAF Outputs
################################################################################

output "waf_web_acl_id" {
  description = "WAF Web ACL ID"
  value       = var.enable_cloudfront_waf ? aws_wafv2_web_acl.cloudfront[0].id : null
}

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN"
  value       = var.enable_cloudfront_waf ? aws_wafv2_web_acl.cloudfront[0].arn : null
}

################################################################################
# Backup Outputs
################################################################################

output "backup_vault_id" {
  description = "AWS Backup vault ID"
  value       = var.enable_aws_backup ? aws_backup_vault.main[0].id : null
}

output "backup_vault_arn" {
  description = "AWS Backup vault ARN"
  value       = var.enable_aws_backup ? aws_backup_vault.main[0].arn : null
}

output "backup_plan_id" {
  description = "AWS Backup plan ID"
  value       = var.enable_aws_backup ? aws_backup_plan.main[0].id : null
}

output "backup_plan_arn" {
  description = "AWS Backup plan ARN"
  value       = var.enable_aws_backup ? aws_backup_plan.main[0].arn : null
}

################################################################################
# CloudTrail Outputs
################################################################################

output "cloudtrail_id" {
  description = "CloudTrail ID"
  value       = var.enable_cloudtrail ? aws_cloudtrail.main[0].id : null
}

output "cloudtrail_arn" {
  description = "CloudTrail ARN"
  value       = var.enable_cloudtrail ? aws_cloudtrail.main[0].arn : null
}

output "cloudtrail_home_region" {
  description = "CloudTrail home region"
  value       = var.enable_cloudtrail ? aws_cloudtrail.main[0].home_region : null
}

################################################################################
# Service Endpoints
################################################################################

output "dqs_service_endpoint" {
  description = "DQS service endpoint URL"
  value       = var.enable_ecs ? "https://${aws_lb.main[0].dns_name}/dqs" : null
}

output "opa_service_endpoint" {
  description = "OPA service endpoint URL"
  value       = var.enable_ecs ? "https://${aws_lb.main[0].dns_name}/opa" : null
}

output "cdn_endpoint" {
  description = "CDN endpoint URL"
  value       = var.enable_cloudfront ? "https://${var.domain_name}" : null
}

output "cloudfront_url" {
  description = "CloudFront distribution URL"
  value       = var.enable_cloudfront ? "https://${aws_cloudfront_distribution.cdn[0].domain_name}" : null
}

################################################################################
# Connection Information (Sensitive)
################################################################################

output "database_connection_info" {
  description = "Database connection information"
  value = var.enable_rds ? {
    endpoint   = aws_db_instance.main[0].endpoint
    port       = aws_db_instance.main[0].port
    database   = aws_db_instance.main[0].db_name
    username   = aws_db_instance.main[0].username
    secret_arn = aws_secretsmanager_secret.rds_password[0].arn
  } : null
  sensitive = true
}

output "redis_connection_info" {
  description = "Redis connection information"
  value = var.enable_redis ? {
    primary_endpoint = aws_elasticache_replication_group.main[0].primary_endpoint_address
    reader_endpoint  = aws_elasticache_replication_group.main[0].reader_endpoint_address
    port             = aws_elasticache_replication_group.main[0].port
    auth_token_secret_arn = aws_secretsmanager_secret.redis_auth_token[0].arn
  } : null
  sensitive = true
}

################################################################################
# Infrastructure Summary
################################################################################

output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    project            = var.project
    environment        = var.environment
    version            = var.version
    primary_region     = var.primary_region
    secondary_region   = var.secondary_region
    vpc_id             = aws_vpc.main.id
    vpc_cidr           = aws_vpc.main.cidr_block
    availability_zones = local.azs
    rds_enabled        = var.enable_rds
    redis_enabled      = var.enable_redis
    ecs_enabled        = var.enable_ecs
    cloudfront_enabled = var.enable_cloudfront
    backup_enabled     = var.enable_aws_backup
    cloudtrail_enabled = var.enable_cloudtrail
    waf_enabled        = var.enable_cloudfront_waf
  }
}

################################################################################
# Resource Tags
################################################################################

output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

################################################################################
# Deployment Information
################################################################################

output "deployment_timestamp" {
  description = "Timestamp of this deployment"
  value       = timestamp()
}

output "terraform_version" {
  description = "Terraform version used for deployment"
  value       = "1.7.0+"
}

################################################################################
# Quick Reference URLs
################################################################################

output "quick_reference" {
  description = "Quick reference URLs for common operations"
  value = {
    s3_console          = "https://s3.console.aws.amazon.com/s3/buckets/${local.bucket_primary}"
    cloudfront_console  = var.enable_cloudfront ? "https://console.aws.amazon.com/cloudfront/v3/home#/distributions/${aws_cloudfront_distribution.cdn[0].id}" : null
    rds_console         = var.enable_rds ? "https://console.aws.amazon.com/rds/home?region=${var.primary_region}#database:id=${aws_db_instance.main[0].id}" : null
    elasticache_console = var.enable_redis ? "https://console.aws.amazon.com/elasticache/home?region=${var.primary_region}#redis-group-nodes:id=${aws_elasticache_replication_group.main[0].id}" : null
    ecs_console         = var.enable_ecs ? "https://console.aws.amazon.com/ecs/home?region=${var.primary_region}#/clusters/${aws_ecs_cluster.main[0].name}" : null
    cloudwatch_console  = "https://console.aws.amazon.com/cloudwatch/home?region=${var.primary_region}"
    vpc_console         = "https://console.aws.amazon.com/vpc/home?region=${var.primary_region}#vpcs:VpcId=${aws_vpc.main.id}"
  }
}

################################################################################
# Next Steps
################################################################################

output "next_steps" {
  description = "Recommended next steps after deployment"
  value = <<-EOT
    VulcanAMI Infrastructure Deployment Complete!
    
    Next Steps:
    1. Update DNS records to point to CloudFront: ${var.enable_cloudfront ? aws_cloudfront_distribution.cdn[0].domain_name : "N/A"}
    2. Retrieve database password: aws secretsmanager get-secret-value --secret-id ${var.enable_rds ? aws_secretsmanager_secret.rds_password[0].id : "N/A"}
    3. Deploy applications to ECS: ${var.enable_ecs ? aws_ecs_cluster.main[0].name : "N/A"}
    4. Configure monitoring dashboards in CloudWatch
    5. Review and configure CloudWatch alarms
    6. Test disaster recovery procedures
    7. Configure backup retention policies
    8. Review security group rules
    9. Enable additional logging if needed
    10. Document connection strings and credentials
    
    For more information, see: https://docs.vulcanami.io
  EOT
}

################################################################################
# End of Outputs
################################################################################
