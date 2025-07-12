# Variables for AWS Infrastructure Deployment

# ================================
# Project Configuration
# ================================
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rental-ml-system"
}

variable "environment" {
  description = "Environment (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "ml-team"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

# ================================
# Networking Configuration
# ================================
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
}

variable "elasticache_subnet_cidrs" {
  description = "CIDR blocks for ElastiCache subnets"
  type        = list(string)
  default     = ["10.0.301.0/24", "10.0.302.0/24", "10.0.303.0/24"]
}

# ================================
# EKS Configuration
# ================================
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.27"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ================================
# RDS Configuration
# ================================
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "rds_replica_instance_class" {
  description = "RDS replica instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

variable "create_rds_replica" {
  description = "Whether to create RDS read replica"
  type        = bool
  default     = true
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "rental_ml_prod"
}

variable "database_username" {
  description = "Database master username"
  type        = string
  default     = "postgres"
}

# ================================
# ElastiCache Configuration
# ================================
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the Redis replication group"
  type        = number
  default     = 3
}

# ================================
# DNS and SSL Configuration
# ================================
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "rental-ml.com"
}

variable "create_route53_zone" {
  description = "Whether to create Route53 hosted zone"
  type        = bool
  default     = true
}

variable "create_ssl_certificate" {
  description = "Whether to create SSL certificate"
  type        = bool
  default     = true
}

# ================================
# Monitoring Configuration
# ================================
variable "enable_cloudwatch_logs" {
  description = "Whether to enable CloudWatch logs"
  type        = bool
  default     = true
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# ================================
# Cost Optimization
# ================================
variable "enable_spot_instances" {
  description = "Whether to enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "enable_resource_scheduling" {
  description = "Whether to enable resource scheduling for cost optimization"
  type        = bool
  default     = false
}

# ================================
# Security Configuration
# ================================
variable "enable_network_policy" {
  description = "Whether to enable network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Whether to enable pod security policies"
  type        = bool
  default     = true
}

variable "enable_encryption_at_rest" {
  description = "Whether to enable encryption at rest"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Whether to enable encryption in transit"
  type        = bool
  default     = true
}

# ================================
# Backup Configuration
# ================================
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Whether to enable cross-region backup"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "Region for cross-region backup"
  type        = string
  default     = "us-east-1"
}

# ================================
# Auto Scaling Configuration
# ================================
variable "enable_cluster_autoscaler" {
  description = "Whether to enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Whether to enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

variable "enable_vertical_pod_autoscaler" {
  description = "Whether to enable vertical pod autoscaler"
  type        = bool
  default     = false
}

# ================================
# ML Specific Configuration
# ================================
variable "enable_gpu_nodes" {
  description = "Whether to enable GPU nodes for ML training"
  type        = bool
  default     = true
}

variable "gpu_instance_types" {
  description = "GPU instance types for ML training"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge", "p3.16xlarge"]
}

variable "ml_storage_size" {
  description = "Storage size for ML data and models in GB"
  type        = number
  default     = 1000
}

# ================================
# Environment Specific Overrides
# ================================
variable "environment_config" {
  description = "Environment-specific configuration overrides"
  type = object({
    rds_instance_class    = optional(string)
    redis_node_type      = optional(string)
    min_nodes           = optional(number)
    max_nodes           = optional(number)
    enable_monitoring   = optional(bool)
    backup_retention   = optional(number)
  })
  default = {}
}

# ================================
# Feature Flags
# ================================
variable "enable_service_mesh" {
  description = "Whether to enable Istio service mesh"
  type        = bool
  default     = true
}

variable "enable_observability_stack" {
  description = "Whether to enable full observability stack (Prometheus, Grafana, Jaeger)"
  type        = bool
  default     = true
}

variable "enable_security_scanning" {
  description = "Whether to enable security scanning tools"
  type        = bool
  default     = true
}

variable "enable_cost_monitoring" {
  description = "Whether to enable cost monitoring and optimization"
  type        = bool
  default     = true
}

# ================================
# Advanced Configuration
# ================================
variable "custom_security_groups" {
  description = "Custom security group rules"
  type = list(object({
    name        = string
    description = string
    ingress = list(object({
      from_port   = number
      to_port     = number
      protocol    = string
      cidr_blocks = list(string)
    }))
    egress = list(object({
      from_port   = number
      to_port     = number
      protocol    = string
      cidr_blocks = list(string)
    }))
  }))
  default = []
}

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# ================================
# Disaster Recovery Configuration
# ================================
variable "enable_disaster_recovery" {
  description = "Whether to enable disaster recovery setup"
  type        = bool
  default     = false
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-east-1"
}

variable "rpo_hours" {
  description = "Recovery Point Objective in hours"
  type        = number
  default     = 1
}

variable "rto_hours" {
  description = "Recovery Time Objective in hours"
  type        = number
  default     = 4
}