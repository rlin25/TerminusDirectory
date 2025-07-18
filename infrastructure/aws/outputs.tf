# AWS Infrastructure Outputs for Rental ML System

# VPC outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnets" {
  description = "List of IDs of database subnets"
  value       = module.vpc.database_subnets
}

output "database_subnet_group" {
  description = "ID of the database subnet group"
  value       = module.vpc.database_subnet_group
}

# EKS outputs
output "cluster_id" {
  description = "The ID of the EKS cluster"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = module.eks.cluster_version
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by the EKS cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

output "node_security_group_id" {
  description = "ID of the node shared security group"
  value       = module.eks.node_security_group_id
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

# RDS outputs
output "db_instance_id" {
  description = "The RDS instance ID"
  value       = module.rds.db_instance_id
}

output "db_instance_endpoint" {
  description = "The connection endpoint"
  value       = module.rds.db_instance_endpoint
}

output "db_instance_port" {
  description = "The database port"
  value       = module.rds.db_instance_port
}

output "db_instance_arn" {
  description = "The ARN of the RDS instance"
  value       = module.rds.db_instance_arn
}

output "db_instance_status" {
  description = "The RDS instance status"
  value       = module.rds.db_instance_status
}

output "db_instance_name" {
  description = "The database name"
  value       = module.rds.db_instance_name
}

output "db_instance_username" {
  description = "The master username for the database"
  value       = module.rds.db_instance_username
  sensitive   = true
}

output "db_instance_password" {
  description = "The database password (this password may be old, because Terraform doesn't track it after initial creation)"
  value       = module.rds.db_instance_password
  sensitive   = true
}

output "db_subnet_group_id" {
  description = "The db subnet group name"
  value       = module.rds.db_subnet_group_id
}

output "db_parameter_group_id" {
  description = "The db parameter group id"
  value       = module.rds.db_parameter_group_id
}

output "db_read_replica_id" {
  description = "The read replica instance ID"
  value       = var.environment == "production" ? module.rds.db_read_replica_id : null
}

output "db_read_replica_endpoint" {
  description = "The read replica connection endpoint"
  value       = var.environment == "production" ? module.rds.db_read_replica_endpoint : null
}

# Redis outputs
output "redis_cluster_id" {
  description = "The cache cluster identifier"
  value       = module.redis.cache_cluster_id
}

output "redis_cluster_address" {
  description = "The cache cluster address"
  value       = module.redis.cache_cluster_address
}

output "redis_cluster_port" {
  description = "The cache cluster port"
  value       = module.redis.cache_cluster_port
}

output "redis_subnet_group_name" {
  description = "The cache subnet group name"
  value       = module.redis.cache_subnet_group_name
}

output "redis_parameter_group_id" {
  description = "The cache parameter group id"
  value       = module.redis.cache_parameter_group_id
}

# Load Balancer outputs
output "alb_id" {
  description = "The ID and ARN of the load balancer"
  value       = module.alb.lb_id
}

output "alb_arn" {
  description = "The ARN of the load balancer"
  value       = module.alb.lb_arn
}

output "alb_dns_name" {
  description = "The DNS name of the load balancer"
  value       = module.alb.lb_dns_name
}

output "alb_hosted_zone_id" {
  description = "The canonical hosted zone ID of the load balancer"
  value       = module.alb.lb_zone_id
}

output "alb_target_group_arns" {
  description = "ARNs of the target groups"
  value       = module.alb.target_group_arns
}

output "alb_target_group_arn_suffixes" {
  description = "ARN suffixes of our target groups - can be used with CloudWatch"
  value       = module.alb.target_group_arn_suffixes
}

# S3 outputs
output "s3_bucket_ml_models" {
  description = "The name of the ML models S3 bucket"
  value       = module.s3.bucket_ids["ml_models"]
}

output "s3_bucket_app_data" {
  description = "The name of the app data S3 bucket"
  value       = module.s3.bucket_ids["app_data"]
}

output "s3_bucket_backups" {
  description = "The name of the backups S3 bucket"
  value       = module.s3.bucket_ids["backups"]
}

output "s3_bucket_alb_logs" {
  description = "The name of the ALB logs S3 bucket"
  value       = module.s3.bucket_ids["alb_logs"]
}

# Security Group outputs
output "alb_security_group_id" {
  description = "The ID of the ALB security group"
  value       = module.security_groups.alb_security_group_id
}

output "rds_security_group_id" {
  description = "The ID of the RDS security group"
  value       = module.security_groups.rds_security_group_id
}

output "redis_security_group_id" {
  description = "The ID of the Redis security group"
  value       = module.security_groups.redis_security_group_id
}

# ACM outputs
output "acm_certificate_arn" {
  description = "The ARN of the certificate"
  value       = module.acm.certificate_arn
}

output "acm_certificate_domain_validation_options" {
  description = "A list of attributes to feed into other resources to complete certificate validation"
  value       = module.acm.certificate_domain_validation_options
}

output "acm_certificate_status" {
  description = "Status of the certificate"
  value       = module.acm.certificate_status
}

# Route53 outputs
output "route53_record_main" {
  description = "Main domain Route53 record"
  value = {
    name = aws_route53_record.main.name
    type = aws_route53_record.main.type
    fqdn = aws_route53_record.main.fqdn
  }
}

output "route53_record_api" {
  description = "API subdomain Route53 record"
  value = {
    name = aws_route53_record.api.name
    type = aws_route53_record.api.type
    fqdn = aws_route53_record.api.fqdn
  }
}

# CloudWatch outputs
output "cloudwatch_log_group_eks_cluster" {
  description = "Name of the EKS cluster CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

# Parameter Store outputs
output "parameter_store_database_url" {
  description = "Parameter Store key for database URL"
  value       = aws_ssm_parameter.database_url.name
  sensitive   = true
}

output "parameter_store_redis_url" {
  description = "Parameter Store key for Redis URL"
  value       = aws_ssm_parameter.redis_url.name
  sensitive   = true
}

# Auto Scaling outputs
output "autoscaling_policy_scale_up_arn" {
  description = "ARN of the scale up auto scaling policy"
  value       = aws_autoscaling_policy.scale_up.arn
}

output "autoscaling_policy_scale_down_arn" {
  description = "ARN of the scale down auto scaling policy"
  value       = aws_autoscaling_policy.scale_down.arn
}

# CloudWatch Alarms outputs
output "cloudwatch_alarm_cpu_high_arn" {
  description = "ARN of the high CPU CloudWatch alarm"
  value       = aws_cloudwatch_metric_alarm.cpu_high.arn
}

output "cloudwatch_alarm_cpu_low_arn" {
  description = "ARN of the low CPU CloudWatch alarm"
  value       = aws_cloudwatch_metric_alarm.cpu_low.arn
}

# Connection strings for applications
output "application_database_connection_string" {
  description = "Database connection string for applications"
  value       = "postgresql://${var.database_username}:${module.rds.db_instance_password}@${module.rds.db_instance_endpoint}:${module.rds.db_instance_port}/${var.database_name}"
  sensitive   = true
}

output "application_redis_connection_string" {
  description = "Redis connection string for applications"
  value       = "redis://${module.redis.cache_cluster_address}:${module.redis.cache_cluster_port}"
  sensitive   = true
}

# Kubernetes configuration
output "kubectl_config" {
  description = "kubectl config as generated by the module"
  value = {
    cluster_name                      = module.eks.cluster_id
    endpoint                         = module.eks.cluster_endpoint
    cluster_ca_certificate           = module.eks.cluster_certificate_authority_data
    cluster_security_group_id        = module.eks.cluster_security_group_id
    node_security_group_id           = module.eks.node_security_group_id
    oidc_provider_arn               = module.eks.oidc_provider_arn
    cluster_iam_role_arn            = module.eks.cluster_iam_role_arn
  }
  sensitive = true
}

# Environment-specific outputs
output "environment_info" {
  description = "Environment-specific information"
  value = {
    environment           = var.environment
    region               = var.aws_region
    project_name         = var.project_name
    cluster_name         = var.cluster_name
    domain_name          = var.domain_name
    vpc_cidr             = var.vpc_cidr
    availability_zones   = slice(data.aws_availability_zones.available.names, 0, 3)
  }
}

# Cost tracking outputs
output "cost_allocation_tags" {
  description = "Tags used for cost allocation"
  value       = var.cost_allocation_tags
}

# Monitoring endpoints
output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    prometheus_url = "https://monitoring.${var.domain_name}/prometheus"
    grafana_url    = "https://monitoring.${var.domain_name}/grafana"
    application_url = "https://api.${var.domain_name}"
    main_url       = "https://${var.domain_name}"
  }
}

# Service discovery information
output "service_discovery" {
  description = "Service discovery information for applications"
  value = {
    database_endpoint = module.rds.db_instance_endpoint
    database_port     = module.rds.db_instance_port
    redis_endpoint    = module.redis.cache_cluster_address
    redis_port        = module.redis.cache_cluster_port
    load_balancer_dns = module.alb.lb_dns_name
    cluster_endpoint  = module.eks.cluster_endpoint
  }
}

# Security information
output "security_info" {
  description = "Security-related information"
  value = {
    cluster_security_group_id = module.eks.cluster_security_group_id
    node_security_group_id    = module.eks.node_security_group_id
    alb_security_group_id     = module.security_groups.alb_security_group_id
    rds_security_group_id     = module.security_groups.rds_security_group_id
    redis_security_group_id   = module.security_groups.redis_security_group_id
    certificate_arn           = module.acm.certificate_arn
  }
}