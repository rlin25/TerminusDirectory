# AWS Infrastructure for Rental ML System
# Production-ready infrastructure with high availability, security, and scalability

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "s3" {
    bucket         = "rental-ml-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "rental-ml-terraform-locks"
  }
}

# ================================
# Provider Configuration
# ================================
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "rental-ml-system"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "ml-team"
      CostCenter  = "engineering"
    }
  }
}

# ================================
# Data Sources
# ================================
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ================================
# Local Values
# ================================
locals {
  cluster_name = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner
  }

  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

# ================================
# Random Password Generation
# ================================
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
}

# ================================
# VPC and Networking
# ================================
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  # Database subnets
  database_subnets                   = var.database_subnet_cidrs
  create_database_subnet_group       = true
  create_database_subnet_route_table = true

  # ElastiCache subnets
  elasticache_subnets = var.elasticache_subnet_cidrs

  # Enable DNS
  enable_dns_hostnames = true
  enable_dns_support   = true

  # NAT Gateway
  enable_nat_gateway = true
  single_nat_gateway = false
  one_nat_gateway_per_az = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  # VPC Endpoints
  enable_s3_endpoint = true
  enable_dynamodb_endpoint = true

  tags = local.common_tags
}

# ================================
# Security Groups
# ================================
# EKS Cluster Security Group
resource "aws_security_group" "eks_cluster" {
  name_prefix = "${local.cluster_name}-cluster-"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-cluster-sg"
  })
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${local.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-sg"
  })
}

# ElastiCache Security Group
resource "aws_security_group" "elasticache" {
  name_prefix = "${local.cluster_name}-elasticache-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-elasticache-sg"
  })
}

# ================================
# EKS Cluster
# ================================
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name = "general"
      
      instance_types = ["m5.large", "m5.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 3
      max_size     = 10
      desired_size = 3
      
      update_config = {
        max_unavailable_percentage = 25
      }
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }
      
      tags = {
        ExtraTag = "general-nodes"
      }
    }

    # ML training nodes with GPU
    ml_training = {
      name = "ml-training"
      
      instance_types = ["p3.2xlarge", "p3.8xlarge"]
      capacity_type  = "SPOT"
      
      min_size     = 0
      max_size     = 5
      desired_size = 0
      
      taints = {
        dedicated = {
          key    = "ml-training"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "ml-training"
        WorkloadType = "gpu"
      }
    }

    # Monitoring nodes
    monitoring = {
      name = "monitoring"
      
      instance_types = ["c5.large", "c5.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      min_size     = 2
      max_size     = 4
      desired_size = 2
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "monitoring"
        WorkloadType = "monitoring"
      }
    }
  }

  # Cluster security group
  cluster_security_group_id = aws_security_group.eks_cluster.id

  tags = local.common_tags
}

# ================================
# RDS PostgreSQL Database
# ================================
resource "aws_db_subnet_group" "main" {
  name       = "${local.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.database_subnets

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-subnet-group"
  })
}

resource "aws_db_parameter_group" "postgres" {
  name   = "${local.cluster_name}-postgres-params"
  family = "postgres15"

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  tags = local.common_tags
}

# Primary RDS Instance
resource "aws_db_instance" "primary" {
  identifier = "${local.cluster_name}-postgres-primary"

  # Engine configuration
  engine         = "postgres"
  engine_version = var.postgres_version
  instance_class = var.rds_instance_class

  # Database configuration
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn

  # Database credentials
  db_name  = var.database_name
  username = var.database_username
  password = random_password.db_password.result

  # Network configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # Backup configuration
  backup_retention_period = var.rds_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  # Performance and monitoring
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn

  # Parameter group
  parameter_group_name = aws_db_parameter_group.postgres.name

  # Deletion protection
  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${local.cluster_name}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-postgres-primary"
  })
}

# Read Replica
resource "aws_db_instance" "replica" {
  count = var.create_rds_replica ? 1 : 0

  identifier = "${local.cluster_name}-postgres-replica"

  # Replica configuration
  replicate_source_db = aws_db_instance.primary.identifier
  instance_class      = var.rds_replica_instance_class

  # Network configuration
  publicly_accessible = false

  # Performance monitoring
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-postgres-replica"
  })
}

# ================================
# ElastiCache Redis Cluster
# ================================
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.cluster_name}-cache-subnet"
  subnet_ids = module.vpc.elasticache_subnets

  tags = local.common_tags
}

resource "aws_elasticache_parameter_group" "redis" {
  name   = "${local.cluster_name}-redis-params"
  family = "redis7.x"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${local.cluster_name}-redis"
  description                = "Redis cluster for ${local.cluster_name}"

  # Redis configuration
  node_type          = var.redis_node_type
  port               = 6379
  parameter_group_name = aws_elasticache_parameter_group.redis.name

  # Cluster configuration
  num_cache_clusters = var.redis_num_cache_nodes
  multi_az_enabled   = true
  automatic_failover_enabled = true

  # Security
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.elasticache.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token = random_password.redis_password.result

  # Backup
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"

  tags = local.common_tags
}

# ================================
# Application Load Balancer
# ================================
resource "aws_lb" "main" {
  name               = "${local.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"

  access_logs {
    bucket  = aws_s3_bucket.alb_logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }

  tags = local.common_tags
}

resource "aws_security_group" "alb" {
  name_prefix = "${local.cluster_name}-alb-"
  vpc_id      = module.vpc.vpc_id

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
    Name = "${local.cluster_name}-alb-sg"
  })
}

# ================================
# S3 Buckets
# ================================
# ALB Access Logs Bucket
resource "aws_s3_bucket" "alb_logs" {
  bucket = "${local.cluster_name}-alb-logs-${random_id.bucket_suffix.hex}"

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# ML Models and Artifacts Bucket
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${local.cluster_name}-ml-artifacts-${random_id.bucket_suffix.hex}"

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "ml_artifacts" {
  bucket = aws_s3_bucket.ml_artifacts.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.s3.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

# Application Data Backup Bucket
resource "aws_s3_bucket" "backups" {
  bucket = "${local.cluster_name}-backups-${random_id.bucket_suffix.hex}"

  tags = local.common_tags
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup_lifecycle"
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
      days = 2555  # 7 years
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ================================
# KMS Keys
# ================================
resource "aws_kms_key" "rds" {
  description             = "RDS encryption key for ${local.cluster_name}"
  deletion_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-key"
  })
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${local.cluster_name}-rds"
  target_key_id = aws_kms_key.rds.key_id
}

resource "aws_kms_key" "s3" {
  description             = "S3 encryption key for ${local.cluster_name}"
  deletion_window_in_days = 7

  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-s3-key"
  })
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${local.cluster_name}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# ================================
# IAM Roles and Policies
# ================================
# RDS Monitoring Role
resource "aws_iam_role" "rds_monitoring" {
  name = "${local.cluster_name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ================================
# CloudWatch and Monitoring
# ================================
resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${local.cluster_name}/cluster"
  retention_in_days = 30

  tags = local.common_tags
}

# ================================
# Route53 and DNS
# ================================
resource "aws_route53_zone" "main" {
  count = var.create_route53_zone ? 1 : 0
  name  = var.domain_name

  tags = local.common_tags
}

# ================================
# Certificate Manager
# ================================
resource "aws_acm_certificate" "main" {
  count           = var.create_ssl_certificate ? 1 : 0
  domain_name     = var.domain_name
  validation_method = "DNS"

  subject_alternative_names = [
    "*.${var.domain_name}",
  ]

  lifecycle {
    create_before_destroy = true
  }

  tags = local.common_tags
}