# Azure Infrastructure Variables

variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rental-ml"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "East US 2"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["1", "2", "3"]
}

# Terraform State
variable "terraform_state_resource_group" {
  description = "Resource group for Terraform state"
  type        = string
}

variable "terraform_state_storage_account" {
  description = "Storage account for Terraform state"
  type        = string
}

variable "terraform_state_container" {
  description = "Container for Terraform state"
  type        = string
  default     = "tfstate"
}

# Networking
variable "vnet_cidr" {
  description = "CIDR block for VNet"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "CIDR block for AKS subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "private_endpoints_subnet_cidr" {
  description = "CIDR block for private endpoints subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "database_subnet_cidr" {
  description = "CIDR block for database subnet"
  type        = string
  default     = "10.0.3.0/24"
}

variable "application_gateway_subnet_cidr" {
  description = "CIDR block for Application Gateway subnet"
  type        = string
  default     = "10.0.4.0/24"
}

variable "dns_service_ip" {
  description = "DNS service IP for AKS"
  type        = string
  default     = "10.1.0.10"
}

variable "service_cidr" {
  description = "Service CIDR for AKS"
  type        = string
  default     = "10.1.0.0/16"
}

# AKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "private_cluster_enabled" {
  description = "Enable private cluster"
  type        = bool
  default     = true
}

variable "aks_admin_groups" {
  description = "Azure AD group object IDs for AKS admins"
  type        = list(string)
  default     = []
}

# System Node Pool
variable "system_node_pool" {
  description = "System node pool configuration"
  type = object({
    vm_size         = string
    node_count      = number
    min_count       = number
    max_count       = number
    max_pods        = number
    os_disk_size_gb = number
  })
  default = {
    vm_size         = "Standard_D4s_v3"
    node_count      = 3
    min_count       = 3
    max_count       = 10
    max_pods        = 30
    os_disk_size_gb = 100
  }
}

# User Node Pool
variable "user_node_pool" {
  description = "User node pool configuration"
  type = object({
    vm_size         = string
    node_count      = number
    min_count       = number
    max_count       = number
    max_pods        = number
    os_disk_size_gb = number
  })
  default = {
    vm_size         = "Standard_D8s_v3"
    node_count      = 2
    min_count       = 2
    max_count       = 20
    max_pods        = 50
    os_disk_size_gb = 200
  }
}

# GPU Node Pool
variable "enable_gpu_nodes" {
  description = "Enable GPU node pool"
  type        = bool
  default     = false
}

variable "gpu_node_pool" {
  description = "GPU node pool configuration"
  type = object({
    vm_size         = string
    max_count       = number
    max_pods        = number
    os_disk_size_gb = number
  })
  default = {
    vm_size         = "Standard_NC6s_v3"
    max_count       = 10
    max_pods        = 15
    os_disk_size_gb = 500
  }
}

# PostgreSQL Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "postgres_sku_name" {
  description = "PostgreSQL SKU name"
  type        = string
  default     = "GP_Standard_D4s_v3"
}

variable "postgres_storage_mb" {
  description = "PostgreSQL storage in MB"
  type        = number
  default     = 131072 # 128 GB
}

variable "postgres_admin_username" {
  description = "PostgreSQL admin username"
  type        = string
  default     = "pgadmin"
}

variable "postgres_admin_password" {
  description = "PostgreSQL admin password"
  type        = string
  sensitive   = true
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "rental_ml"
}

# Redis Configuration
variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "Redis cache SKU name"
  type        = string
  default     = "Premium"
}

variable "redis_private_ip" {
  description = "Redis private IP address"
  type        = string
  default     = "10.0.2.10"
}

variable "redis_memory_reserved" {
  description = "Redis memory reserved"
  type        = number
  default     = 30
}

variable "redis_memory_delta" {
  description = "Redis memory delta"
  type        = number
  default     = 30
}

# Application Gateway
variable "application_gateway_capacity" {
  description = "Application Gateway capacity"
  type        = number
  default     = 2
}

# Monitoring
variable "admin_email" {
  description = "Admin email for alerts"
  type        = string
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alerts"
  type        = string
  default     = ""
}

# Domain
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "rental-ml.com"
}