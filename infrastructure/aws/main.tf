# AWS Infrastructure for Rental ML System
# Production-ready multi-AZ deployment with auto-scaling and monitoring

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
  }
  
  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "rental-ml-system/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

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

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  name               = "${var.project_name}-${var.environment}"
  cidr               = var.vpc_cidr
  azs                = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets    = var.private_subnets
  public_subnets     = var.public_subnets
  database_subnets   = var.database_subnets
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_destination_type            = "cloud-watch-logs"
  
  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    "kubernetes.io/role/internal-elb"           = "1"
  }
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version
  
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      min_size       = var.node_group_min_size
      max_size       = var.node_group_max_size
      desired_size   = var.node_group_desired_size
      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
      
      # Taints for specific workloads
      taints = {
        dedicated = {
          key    = "dedicated"
          value  = "ml-workloads"
          effect = "NO_SCHEDULE"
        }
      }
    }
    
    ml_training = {
      min_size       = 0
      max_size       = 10
      desired_size   = 0
      instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]
      capacity_type  = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "ml-training"
        WorkloadType = "gpu"
      }
      
      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
  
  # EKS Addons
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
  
  # Enable IRSA
  enable_irsa = true
  
  # Cluster endpoint configuration
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  cluster_endpoint_public_access_cidrs = var.cluster_endpoint_public_access_cidrs
  
  # Enable cluster logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Security group rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Node groups to cluster API"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }
  
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 65535
      type        = "ingress"
      self        = true
    }
    
    egress_all = {
      description      = "Node all egress"
      protocol         = "-1"
      from_port        = 0
      to_port          = 65535
      type             = "egress"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = ["::/0"]
    }
  }
}

# RDS Database
module "rds" {
  source = "./modules/rds"
  
  identifier = "${var.project_name}-${var.environment}-postgres"
  
  engine         = "postgres"
  engine_version = var.postgres_version
  instance_class = var.rds_instance_class
  
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_encrypted     = true
  
  db_name  = var.database_name
  username = var.database_username
  manage_master_user_password = true
  
  vpc_security_group_ids = [module.security_groups.rds_security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group
  
  # Multi-AZ for production
  multi_az = var.environment == "production" ? true : false
  
  # Backup configuration
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Enable deletion protection for production
  deletion_protection = var.environment == "production" ? true : false
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = var.environment == "production" ? 7 : 7
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn
  
  # Enable automated minor version upgrade
  auto_minor_version_upgrade = true
  
  # Read replica for production
  create_read_replica = var.environment == "production" ? true : false
  replica_instance_class = var.environment == "production" ? var.rds_replica_instance_class : null
  
  tags = {
    Name = "${var.project_name}-${var.environment}-postgres"
  }
}

# ElastiCache Redis Cluster
module "redis" {
  source = "./modules/redis"
  
  cluster_id = "${var.project_name}-${var.environment}-redis"
  
  engine_version       = var.redis_version
  node_type           = var.redis_node_type
  num_cache_nodes     = var.redis_num_nodes
  parameter_group_name = var.redis_parameter_group
  port                = 6379
  
  subnet_group_name  = module.vpc.elasticache_subnet_group_name
  security_group_ids = [module.security_groups.redis_security_group_id]
  
  # Multi-AZ for production
  az_mode = var.environment == "production" ? "cross-az" : "single-az"
  
  # Enable automatic failover for production
  automatic_failover_enabled = var.environment == "production" ? true : false
  multi_az_enabled          = var.environment == "production" ? true : false
  
  # Backup configuration
  snapshot_retention_limit = var.environment == "production" ? 5 : 1
  snapshot_window         = "03:00-05:00"
  
  # Enable at-rest encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "${var.project_name}-${var.environment}-redis"
  }
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"
  
  name               = "${var.project_name}-${var.environment}-alb"
  load_balancer_type = "application"
  
  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets
  
  security_groups = [module.security_groups.alb_security_group_id]
  
  # Enable access logs
  access_logs = {
    bucket  = module.s3.alb_logs_bucket_id
    enabled = true
  }
  
  # Target groups
  target_groups = [
    {
      name             = "${var.project_name}-${var.environment}-app"
      backend_protocol = "HTTP"
      backend_port     = 8000
      target_type      = "ip"
      health_check = {
        enabled             = true
        healthy_threshold   = 3
        interval            = 30
        matcher             = "200"
        path                = "/health"
        port                = "traffic-port"
        protocol            = "HTTP"
        timeout             = 5
        unhealthy_threshold = 2
      }
    }
  ]
  
  # HTTPS listener
  https_listeners = [
    {
      port               = 443
      protocol           = "HTTPS"
      certificate_arn    = module.acm.certificate_arn
      target_group_index = 0
    }
  ]
  
  # HTTP to HTTPS redirect
  http_tcp_listeners = [
    {
      port        = 80
      protocol    = "HTTP"
      action_type = "redirect"
      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  ]
  
  tags = {
    Name = "${var.project_name}-${var.environment}-alb"
  }
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"
  
  project_name = var.project_name
  environment  = var.environment
  
  # Create buckets for different purposes
  buckets = {
    ml_models = {
      versioning_enabled = true
      lifecycle_rules = [
        {
          id     = "ml_models_lifecycle"
          status = "Enabled"
          
          transition = [
            {
              days          = 30
              storage_class = "STANDARD_IA"
            },
            {
              days          = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    
    app_data = {
      versioning_enabled = true
      lifecycle_rules = [
        {
          id     = "app_data_lifecycle"
          status = "Enabled"
          
          transition = [
            {
              days          = 30
              storage_class = "STANDARD_IA"
            }
          ]
        }
      ]
    }
    
    backups = {
      versioning_enabled = true
      lifecycle_rules = [
        {
          id     = "backup_lifecycle"
          status = "Enabled"
          
          transition = [
            {
              days          = 7
              storage_class = "STANDARD_IA"
            },
            {
              days          = 30
              storage_class = "GLACIER"
            },
            {
              days          = 365
              storage_class = "DEEP_ARCHIVE"
            }
          ]
        }
      ]
    }
    
    alb_logs = {
      versioning_enabled = false
      lifecycle_rules = [
        {
          id     = "alb_logs_lifecycle"
          status = "Enabled"
          
          expiration = {
            days = 90
          }
        }
      ]
    }
  }
}

# Security Groups
module "security_groups" {
  source = "./modules/security-groups"
  
  name_prefix = "${var.project_name}-${var.environment}"
  vpc_id      = module.vpc.vpc_id
  
  # ALB security group rules
  alb_ingress_cidr_blocks = ["0.0.0.0/0"]
  alb_egress_cidr_blocks  = ["0.0.0.0/0"]
  
  # RDS security group rules
  rds_ingress_security_groups = [module.eks.node_security_group_id]
  
  # Redis security group rules
  redis_ingress_security_groups = [module.eks.node_security_group_id]
}

# ACM Certificate
module "acm" {
  source = "./modules/acm"
  
  domain_name = var.domain_name
  zone_id     = var.route53_zone_id
  
  subject_alternative_names = [
    "*.${var.domain_name}",
    "api.${var.domain_name}",
    "monitoring.${var.domain_name}"
  ]
  
  wait_for_validation = true
  
  tags = {
    Name = "${var.project_name}-${var.environment}"
  }
}

# Route53 Records
resource "aws_route53_record" "main" {
  zone_id = var.route53_zone_id
  name    = var.domain_name
  type    = "A"
  
  alias {
    name                   = module.alb.lb_dns_name
    zone_id                = module.alb.lb_zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "api" {
  zone_id = var.route53_zone_id
  name    = "api.${var.domain_name}"
  type    = "A"
  
  alias {
    name                   = module.alb.lb_dns_name
    zone_id                = module.alb.lb_zone_id
    evaluate_target_health = true
  }
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name_prefix = "${var.project_name}-rds-monitoring-"
  
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
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.cloudwatch_log_retention_days
  
  tags = {
    Name = "${var.cluster_name}-logs"
  }
}

# Parameter Store values for application configuration
resource "aws_ssm_parameter" "database_url" {
  name  = "/${var.project_name}/${var.environment}/database/url"
  type  = "SecureString"
  value = "postgresql://${var.database_username}:${module.rds.db_instance_password}@${module.rds.db_instance_endpoint}/${var.database_name}"
  
  tags = {
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "redis_url" {
  name  = "/${var.project_name}/${var.environment}/redis/url"
  type  = "SecureString"
  value = "redis://${module.redis.cache_cluster_address}:${module.redis.cache_cluster_port}"
  
  tags = {
    Environment = var.environment
  }
}

# Auto Scaling for EKS nodes based on custom metrics
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "${var.cluster_name}-scale-up"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = module.eks.eks_managed_node_groups["main"].asg_name
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "${var.cluster_name}-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = module.eks.eks_managed_node_groups["main"].asg_name
}

# CloudWatch Alarms for Auto Scaling
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.cluster_name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "75"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]
  
  dimensions = {
    AutoScalingGroupName = module.eks.eks_managed_node_groups["main"].asg_name
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name          = "${var.cluster_name}-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "25"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]
  
  dimensions = {
    AutoScalingGroupName = module.eks.eks_managed_node_groups["main"].asg_name
  }
}