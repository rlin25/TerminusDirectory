# Outputs for AWS Infrastructure

# ================================
# VPC and Networking Outputs
# ================================
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.vpc.database_subnets
}

output "database_subnet_group_name" {
  description = "Name of the database subnet group"
  value       = aws_db_subnet_group.main.name
}

# ================================
# EKS Cluster Outputs
# ================================
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
  sensitive   = true
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
  sensitive   = true
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

# ================================
# RDS Database Outputs
# ================================
output "rds_primary_endpoint" {
  description = "RDS primary instance endpoint"
  value       = aws_db_instance.primary.endpoint
  sensitive   = true
}

output "rds_primary_id" {
  description = "RDS primary instance ID"
  value       = aws_db_instance.primary.id
}

output "rds_primary_arn" {
  description = "RDS primary instance ARN"
  value       = aws_db_instance.primary.arn
}

output "rds_replica_endpoint" {
  description = "RDS replica instance endpoint"
  value       = var.create_rds_replica ? aws_db_instance.replica[0].endpoint : null
  sensitive   = true
}

output "rds_replica_id" {
  description = "RDS replica instance ID"
  value       = var.create_rds_replica ? aws_db_instance.replica[0].id : null
}

output "database_name" {
  description = "Name of the database"
  value       = aws_db_instance.primary.db_name
}

output "database_username" {
  description = "Database master username"
  value       = aws_db_instance.primary.username
  sensitive   = true
}

output "database_password" {
  description = "Database master password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "database_port" {
  description = "Database port"
  value       = aws_db_instance.primary.port
}

# ================================
# ElastiCache Redis Outputs
# ================================
output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint"
  value       = aws_elasticache_replication_group.redis.reader_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

output "redis_auth_token" {
  description = "Redis auth token"
  value       = random_password.redis_password.result
  sensitive   = true
}

# ================================
# Load Balancer Outputs
# ================================
output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.main.arn
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

# ================================
# S3 Bucket Outputs
# ================================
output "alb_logs_bucket" {
  description = "S3 bucket for ALB access logs"
  value       = aws_s3_bucket.alb_logs.bucket
}

output "ml_artifacts_bucket" {
  description = "S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "backups_bucket" {
  description = "S3 bucket for backups"
  value       = aws_s3_bucket.backups.bucket
}

# ================================
# KMS Key Outputs
# ================================
output "rds_kms_key_id" {
  description = "KMS key ID for RDS encryption"
  value       = aws_kms_key.rds.key_id
}

output "rds_kms_key_arn" {
  description = "KMS key ARN for RDS encryption"
  value       = aws_kms_key.rds.arn
}

output "s3_kms_key_id" {
  description = "KMS key ID for S3 encryption"
  value       = aws_kms_key.s3.key_id
}

output "s3_kms_key_arn" {
  description = "KMS key ARN for S3 encryption"
  value       = aws_kms_key.s3.arn
}

# ================================
# DNS and SSL Outputs
# ================================
output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = var.create_route53_zone ? aws_route53_zone.main[0].zone_id : null
}

output "route53_zone_name" {
  description = "Route53 hosted zone name"
  value       = var.create_route53_zone ? aws_route53_zone.main[0].name : null
}

output "ssl_certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = var.create_ssl_certificate ? aws_acm_certificate.main[0].arn : null
}

# ================================
# Security Group Outputs
# ================================
output "eks_cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = aws_security_group.eks_cluster.id
}

output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}

output "elasticache_security_group_id" {
  description = "ElastiCache security group ID"
  value       = aws_security_group.elasticache.id
}

output "alb_security_group_id" {
  description = "ALB security group ID"
  value       = aws_security_group.alb.id
}

# ================================
# Kubernetes Configuration Outputs
# ================================
output "kubectl_config" {
  description = "kubectl config for the EKS cluster"
  value = {
    cluster_name     = module.eks.cluster_id
    endpoint        = module.eks.cluster_endpoint
    ca_certificate  = module.eks.cluster_certificate_authority_data
    region          = var.aws_region
  }
  sensitive = true
}

# ================================
# Connection Strings and URLs
# ================================
output "database_url" {
  description = "Database connection URL"
  value       = "postgresql://${aws_db_instance.primary.username}:${random_password.db_password.result}@${aws_db_instance.primary.endpoint}/${aws_db_instance.primary.db_name}"
  sensitive   = true
}

output "redis_url" {
  description = "Redis connection URL"
  value       = "redis://:${random_password.redis_password.result}@${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}/0"
  sensitive   = true
}

# ================================
# Environment Configuration
# ================================
output "environment_config" {
  description = "Environment configuration for applications"
  value = {
    environment     = var.environment
    cluster_name    = local.cluster_name
    region         = var.aws_region
    vpc_id         = module.vpc.vpc_id
    
    database = {
      host     = aws_db_instance.primary.endpoint
      port     = aws_db_instance.primary.port
      name     = aws_db_instance.primary.db_name
      username = aws_db_instance.primary.username
    }
    
    redis = {
      host = aws_elasticache_replication_group.redis.primary_endpoint_address
      port = aws_elasticache_replication_group.redis.port
    }
    
    storage = {
      ml_artifacts_bucket = aws_s3_bucket.ml_artifacts.bucket
      backups_bucket      = aws_s3_bucket.backups.bucket
    }
    
    monitoring = {
      cloudwatch_log_group = aws_cloudwatch_log_group.cluster.name
    }
  }
  sensitive = true
}

# ================================
# Cost and Resource Information
# ================================
output "estimated_monthly_cost" {
  description = "Estimated monthly cost (approximate)"
  value = {
    eks_cluster    = "~$73/month (cluster) + node costs"
    rds_primary    = "~$200-400/month depending on instance class"
    rds_replica    = var.create_rds_replica ? "~$200-400/month depending on instance class" : "Not created"
    redis_cluster  = "~$150-300/month depending on node type"
    load_balancer  = "~$20/month + data transfer"
    nat_gateways   = "~$135/month (3 NAT gateways)"
    total_estimate = "~$600-1200/month (excluding data transfer and node costs)"
  }
}

output "resource_inventory" {
  description = "Inventory of created resources"
  value = {
    vpc_resources = {
      vpc                = 1
      private_subnets    = length(module.vpc.private_subnets)
      public_subnets     = length(module.vpc.public_subnets)
      database_subnets   = length(module.vpc.database_subnets)
      elasticache_subnets = length(module.vpc.elasticache_subnets)
      nat_gateways       = length(module.vpc.natgw_ids)
    }
    
    compute_resources = {
      eks_cluster        = 1
      node_groups        = length(module.eks.eks_managed_node_groups)
    }
    
    database_resources = {
      rds_primary        = 1
      rds_replica        = var.create_rds_replica ? 1 : 0
      redis_cluster      = 1
    }
    
    storage_resources = {
      s3_buckets         = 3
      kms_keys          = 2
    }
    
    networking_resources = {
      load_balancers     = 1
      security_groups    = 4
      route53_zones      = var.create_route53_zone ? 1 : 0
      ssl_certificates   = var.create_ssl_certificate ? 1 : 0
    }
  }
}