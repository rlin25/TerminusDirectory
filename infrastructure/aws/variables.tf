# AWS Infrastructure Variables for Rental ML System

variable "aws_region" {
  description = "AWS region for infrastructure deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rental-ml-system"
}

variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
}

# Networking variables
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "database_subnets" {
  description = "Database subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
}

# EKS Cluster variables
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "rental-ml-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the EKS managed node group"
  type        = number
  default     = 2
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the EKS managed node group"
  type        = number
  default     = 20
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the EKS managed node group"
  type        = number
  default     = 3
}

variable "node_instance_types" {
  description = "Instance types for EKS managed node group"
  type        = list(string)
  default     = ["t3.large", "t3.xlarge"]
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# RDS Database variables
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.large"
}

variable "rds_replica_instance_class" {
  description = "RDS read replica instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage for RDS instance (GB)"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for RDS instance (GB)"
  type        = number
  default     = 1000
}

variable "database_name" {
  description = "Name of the database to create"
  type        = string
  default     = "rental_ml_db"
}

variable "database_username" {
  description = "Database admin username"
  type        = string
  default     = "postgres"
}

# Redis variables
variable "redis_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_nodes" {
  description = "Number of Redis nodes"
  type        = number
  default     = 2
}

variable "redis_parameter_group" {
  description = "Redis parameter group"
  type        = string
  default     = "default.redis7"
}

# Domain and SSL variables
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
}

# Monitoring variables
variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

# Cost optimization variables
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_pools" {
  description = "Number of spot instance pools to use"
  type        = number
  default     = 2
}

# Security variables
variable "enable_waf" {
  description = "Enable AWS WAF for application protection"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield Advanced for DDoS protection"
  type        = bool
  default     = false
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Backup and disaster recovery variables
variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

variable "backup_region" {
  description = "Region for cross-region backup replication"
  type        = string
  default     = "us-east-1"
}

# Performance and scaling variables
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period in days"
  type        = number
  default     = 7
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler for EKS"
  type        = bool
  default     = true
}

# ML workload specific variables
variable "enable_gpu_nodes" {
  description = "Enable GPU nodes for ML training workloads"
  type        = bool
  default     = true
}

variable "gpu_node_instance_types" {
  description = "Instance types for GPU nodes"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge"]
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 10
}

# Data lake and analytics variables
variable "enable_data_lake" {
  description = "Enable data lake with S3 and Glue"
  type        = bool
  default     = true
}

variable "enable_kinesis_analytics" {
  description = "Enable Kinesis Analytics for real-time processing"
  type        = bool
  default     = true
}

variable "kinesis_shard_count" {
  description = "Number of Kinesis shards"
  type        = number
  default     = 2
}

# Compliance and governance variables
variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable GuardDuty for threat detection"
  type        = bool
  default     = true
}

# Environment-specific sizing
variable "environment_config" {
  description = "Environment-specific configuration"
  type = map(object({
    node_group_min_size    = number
    node_group_max_size    = number
    node_group_desired_size = number
    rds_instance_class     = string
    redis_node_type        = string
    backup_retention_days  = number
    enable_multi_az        = bool
    enable_read_replicas   = bool
  }))
  default = {
    dev = {
      node_group_min_size    = 1
      node_group_max_size    = 5
      node_group_desired_size = 2
      rds_instance_class     = "db.t3.medium"
      redis_node_type        = "cache.t3.micro"
      backup_retention_days  = 7
      enable_multi_az        = false
      enable_read_replicas   = false
    }
    staging = {
      node_group_min_size    = 2
      node_group_max_size    = 10
      node_group_desired_size = 3
      rds_instance_class     = "db.t3.large"
      redis_node_type        = "cache.t3.small"
      backup_retention_days  = 14
      enable_multi_az        = true
      enable_read_replicas   = false
    }
    production = {
      node_group_min_size    = 3
      node_group_max_size    = 50
      node_group_desired_size = 5
      rds_instance_class     = "db.r6g.xlarge"
      redis_node_type        = "cache.r6g.large"
      backup_retention_days  = 30
      enable_multi_az        = true
      enable_read_replicas   = true
    }
  }
}

# Cost allocation tags
variable "cost_allocation_tags" {
  description = "Tags for cost allocation and tracking"
  type        = map(string)
  default = {
    Department = "Engineering"
    Team       = "ML-Platform"
    CostCenter = "ml-infrastructure"
    BillingOwner = "ml-team"
  }
}

# Notification and alerting variables
variable "notification_email" {
  description = "Email address for infrastructure notifications"
  type        = string
  default     = "ml-team@company.com"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

# Feature flags for infrastructure components
variable "feature_flags" {
  description = "Feature flags for enabling/disabling infrastructure components"
  type = object({
    enable_monitoring        = bool
    enable_logging          = bool
    enable_tracing          = bool
    enable_service_mesh     = bool
    enable_secrets_manager  = bool
    enable_parameter_store  = bool
    enable_vpc_flow_logs    = bool
    enable_network_acls     = bool
  })
  default = {
    enable_monitoring        = true
    enable_logging          = true
    enable_tracing          = true
    enable_service_mesh     = true
    enable_secrets_manager  = true
    enable_parameter_store  = true
    enable_vpc_flow_logs    = true
    enable_network_acls     = true
  }
}