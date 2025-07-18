# Azure Infrastructure for Rental ML System
# Production-ready multi-region deployment with auto-scaling and monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.40"
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
      version = "~> 3.4"
    }
  }

  backend "azurerm" {
    resource_group_name  = var.terraform_state_resource_group
    storage_account_name = var.terraform_state_storage_account
    container_name       = var.terraform_state_container
    key                  = "rental-ml-system/terraform.tfstate"
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = var.environment == "production"
    }
    
    key_vault {
      purge_soft_delete_on_destroy    = var.environment != "production"
      recover_soft_deleted_key_vaults = true
    }
    
    cognitive_account {
      purge_soft_delete_on_destroy = var.environment != "production"
    }
  }

  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}

provider "azuread" {
  tenant_id = var.tenant_id
}

# Random ID for resource naming
resource "random_id" "main" {
  byte_length = 4
}

# Data sources
data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}-rg"
  location = var.location

  tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
    CostCenter  = "ml-platform"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-${var.environment}-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = azurerm_resource_group.main.tags
}

# Subnets
resource "azurerm_subnet" "aks" {
  name                 = "${var.project_name}-${var.environment}-aks-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.aks_subnet_cidr]

  service_endpoints = [
    "Microsoft.Storage",
    "Microsoft.Sql",
    "Microsoft.KeyVault",
    "Microsoft.ContainerRegistry"
  ]
}

resource "azurerm_subnet" "private_endpoints" {
  name                 = "${var.project_name}-${var.environment}-pe-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.private_endpoints_subnet_cidr]

  private_endpoint_network_policies_enabled = false
}

resource "azurerm_subnet" "database" {
  name                 = "${var.project_name}-${var.environment}-db-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.database_subnet_cidr]

  service_endpoints = ["Microsoft.Sql"]

  delegation {
    name = "fs"
    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action",
      ]
    }
  }
}

resource "azurerm_subnet" "application_gateway" {
  name                 = "${var.project_name}-${var.environment}-agw-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.application_gateway_subnet_cidr]
}

# Network Security Groups
resource "azurerm_network_security_group" "aks" {
  name                = "${var.project_name}-${var.environment}-aks-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.aks.id
}

# Azure Key Vault
resource "azurerm_key_vault" "main" {
  name                = "${var.project_name}-${var.environment}-kv-${random_id.main.hex}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"

  enable_rbac_authorization       = true
  enabled_for_disk_encryption     = true
  enabled_for_template_deployment = true
  purge_protection_enabled        = var.environment == "production"
  soft_delete_retention_days      = var.environment == "production" ? 90 : 7

  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    virtual_network_subnet_ids = [
      azurerm_subnet.aks.id,
      azurerm_subnet.private_endpoints.id
    ]
  }

  tags = azurerm_resource_group.main.tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-${var.environment}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "production" ? 90 : 30

  tags = azurerm_resource_group.main.tags
}

# Application Insights
resource "azurerm_application_insights" "main" {
  name                = "${var.project_name}-${var.environment}-ai"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"

  tags = azurerm_resource_group.main.tags
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "${var.project_name}${var.environment}acr${random_id.main.hex}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Premium"
  admin_enabled       = false

  identity {
    type = "SystemAssigned"
  }

  encryption {
    enabled            = true
    key_vault_key_id   = azurerm_key_vault_key.acr_encryption.id
    identity_client_id = azurerm_user_assigned_identity.acr.client_id
  }

  public_network_access_enabled = false
  quarantine_policy_enabled     = true
  trust_policy_enabled          = true

  retention_policy {
    days    = var.environment == "production" ? 30 : 7
    enabled = true
  }

  tags = azurerm_resource_group.main.tags

  depends_on = [
    azurerm_key_vault_access_policy.acr
  ]
}

# User Assigned Identity for ACR encryption
resource "azurerm_user_assigned_identity" "acr" {
  name                = "${var.project_name}-${var.environment}-acr-identity"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  tags = azurerm_resource_group.main.tags
}

# Key Vault Key for ACR encryption
resource "azurerm_key_vault_key" "acr_encryption" {
  name         = "acr-encryption-key"
  key_vault_id = azurerm_key_vault.main.id
  key_type     = "RSA"
  key_size     = 2048

  key_opts = [
    "decrypt",
    "encrypt",
    "sign",
    "unwrapKey",
    "verify",
    "wrapKey",
  ]

  depends_on = [
    azurerm_key_vault_access_policy.terraform
  ]
}

# Key Vault Access Policies
resource "azurerm_key_vault_access_policy" "terraform" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  key_permissions = [
    "Create",
    "Delete",
    "Get",
    "Purge",
    "Recover",
    "Update",
    "List",
    "Decrypt",
    "Encrypt",
    "Sign",
    "UnwrapKey",
    "Verify",
    "WrapKey",
  ]

  secret_permissions = [
    "Get",
    "List",
    "Set",
    "Delete",
    "Purge",
    "Recover"
  ]
}

resource "azurerm_key_vault_access_policy" "acr" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.acr.principal_id

  key_permissions = [
    "Get",
    "UnwrapKey",
    "WrapKey"
  ]
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-${var.environment}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}-${var.environment}"

  kubernetes_version        = var.kubernetes_version
  automatic_channel_upgrade = var.environment == "production" ? "stable" : "patch"
  sku_tier                 = var.environment == "production" ? "Paid" : "Free"

  default_node_pool {
    name                         = "system"
    vm_size                      = var.system_node_pool.vm_size
    node_count                   = var.system_node_pool.node_count
    min_count                    = var.system_node_pool.min_count
    max_count                    = var.system_node_pool.max_count
    enable_auto_scaling          = true
    enable_host_encryption       = true
    enable_node_public_ip        = false
    max_pods                     = var.system_node_pool.max_pods
    orchestrator_version         = var.kubernetes_version
    os_disk_size_gb             = var.system_node_pool.os_disk_size_gb
    os_disk_type                = "Ephemeral"
    vnet_subnet_id              = azurerm_subnet.aks.id
    zones                       = var.availability_zones

    only_critical_addons_enabled = true

    upgrade_settings {
      max_surge = "33%"
    }

    node_labels = {
      "nodepool-type" = "system"
      "environment"   = var.environment
    }

    tags = azurerm_resource_group.main.tags
  }

  identity {
    type = "SystemAssigned"
  }

  # Network configuration
  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    dns_service_ip    = var.dns_service_ip
    service_cidr      = var.service_cidr
    load_balancer_sku = "standard"

    load_balancer_profile {
      outbound_ip_count = 1
    }
  }

  # Azure Active Directory integration
  azure_active_directory_role_based_access_control {
    managed                = true
    admin_group_object_ids = var.aks_admin_groups
    azure_rbac_enabled     = true
  }

  # Add-ons
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  azure_policy_enabled = true

  http_application_routing_enabled = false

  key_vault_secrets_provider {
    secret_rotation_enabled  = true
    secret_rotation_interval = "2m"
  }

  # Security configurations
  private_cluster_enabled             = var.private_cluster_enabled
  private_dns_zone_id                = var.private_cluster_enabled ? "System" : null
  private_cluster_public_fqdn_enabled = false

  # Monitoring
  monitor_metrics {
    annotations_allowed = null
    labels_allowed      = null
  }

  # Auto-scaler profile
  auto_scaler_profile {
    balance_similar_node_groups      = false
    expander                        = "random"
    max_graceful_termination_sec    = "600"
    max_node_provisioning_time      = "15m"
    max_unready_nodes              = 3
    max_unready_percentage         = 45
    new_pod_scale_up_delay         = "10s"
    scale_down_delay_after_add     = "10m"
    scale_down_delay_after_delete  = "10s"
    scale_down_delay_after_failure = "3m"
    scan_interval                  = "10s"
    scale_down_unneeded            = "10m"
    scale_down_unready             = "20m"
    scale_down_utilization_threshold = "0.5"
    empty_bulk_delete_max          = "10"
    skip_nodes_with_local_storage  = false
    skip_nodes_with_system_pods    = true
  }

  tags = azurerm_resource_group.main.tags

  depends_on = [
    azurerm_log_analytics_workspace.main
  ]
}

# User Node Pool for applications
resource "azurerm_kubernetes_cluster_node_pool" "user" {
  name                  = "user"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = var.user_node_pool.vm_size
  node_count           = var.user_node_pool.node_count
  min_count            = var.user_node_pool.min_count
  max_count            = var.user_node_pool.max_count
  enable_auto_scaling   = true
  enable_host_encryption = true
  enable_node_public_ip = false
  max_pods             = var.user_node_pool.max_pods
  orchestrator_version = var.kubernetes_version
  os_disk_size_gb      = var.user_node_pool.os_disk_size_gb
  os_disk_type         = "Ephemeral"
  vnet_subnet_id       = azurerm_subnet.aks.id
  zones                = var.availability_zones

  upgrade_settings {
    max_surge = "33%"
  }

  node_labels = {
    "nodepool-type" = "user"
    "environment"   = var.environment
  }

  node_taints = [
    "workload=user:NoSchedule"
  ]

  tags = azurerm_resource_group.main.tags
}

# GPU Node Pool for ML workloads
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  count = var.enable_gpu_nodes ? 1 : 0

  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size              = var.gpu_node_pool.vm_size
  node_count           = 0
  min_count            = 0
  max_count            = var.gpu_node_pool.max_count
  enable_auto_scaling   = true
  enable_host_encryption = true
  enable_node_public_ip = false
  max_pods             = var.gpu_node_pool.max_pods
  orchestrator_version = var.kubernetes_version
  os_disk_size_gb      = var.gpu_node_pool.os_disk_size_gb
  os_disk_type         = "Ephemeral"
  vnet_subnet_id       = azurerm_subnet.aks.id
  zones                = var.availability_zones

  upgrade_settings {
    max_surge = "33%"
  }

  node_labels = {
    "nodepool-type"    = "gpu"
    "environment"      = var.environment
    "accelerator"      = "nvidia-tesla-v100"
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = azurerm_resource_group.main.tags
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${var.project_name}-${var.environment}-postgres"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = var.postgres_version
  delegated_subnet_id    = azurerm_subnet.database.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = var.postgres_admin_username
  administrator_password = var.postgres_admin_password
  zone                   = "1"

  storage_mb   = var.postgres_storage_mb
  sku_name     = var.postgres_sku_name
  backup_retention_days = var.environment == "production" ? 35 : 7

  high_availability {
    mode                      = var.environment == "production" ? "ZoneRedundant" : "Disabled"
    standby_availability_zone = var.environment == "production" ? "2" : null
  }

  maintenance_window {
    day_of_week  = 0  # Sunday
    start_hour   = 3
    start_minute = 0
  }

  tags = azurerm_resource_group.main.tags

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = var.database_name
  server_id = azurerm_postgresql_flexible_server.main.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Private DNS Zone for PostgreSQL
resource "azurerm_private_dns_zone" "postgres" {
  name                = "${var.project_name}-${var.environment}-postgres.private.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${var.project_name}-${var.environment}-postgres-vnet-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  resource_group_name   = azurerm_resource_group.main.name

  tags = azurerm_resource_group.main.tags
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  name                = "${var.project_name}-${var.environment}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  subnet_id                     = azurerm_subnet.private_endpoints.id
  private_static_ip_address     = var.redis_private_ip
  public_network_access_enabled = false

  redis_configuration {
    enable_authentication           = true
    maxfragmentationmemory_reserved = var.redis_memory_reserved
    maxmemory_reserved             = var.redis_memory_reserved
    maxmemory_delta                = var.redis_memory_delta
    maxmemory_policy               = "allkeys-lru"
  }

  patch_schedule {
    day_of_week    = "Sunday"
    start_hour_utc = 3
  }

  tags = azurerm_resource_group.main.tags
}

# Storage Accounts
resource "azurerm_storage_account" "ml_models" {
  name                     = "${var.project_name}${var.environment}mlmodels${random_id.main.hex}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "production" ? "GRS" : "LRS"
  account_kind             = "StorageV2"

  blob_properties {
    versioning_enabled       = true
    change_feed_enabled      = true
    change_feed_retention_in_days = 7
    last_access_time_enabled = true

    delete_retention_policy {
      days = var.environment == "production" ? 30 : 7
    }

    container_delete_retention_policy {
      days = var.environment == "production" ? 30 : 7
    }
  }

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.aks.id]
    bypass                     = ["AzureServices"]
  }

  identity {
    type = "SystemAssigned"
  }

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_storage_account" "app_data" {
  name                     = "${var.project_name}${var.environment}appdata${random_id.main.hex}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "production" ? "GRS" : "LRS"
  account_kind             = "StorageV2"

  blob_properties {
    versioning_enabled       = true
    change_feed_enabled      = true
    change_feed_retention_in_days = 7

    delete_retention_policy {
      days = var.environment == "production" ? 30 : 7
    }
  }

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.aks.id]
    bypass                     = ["AzureServices"]
  }

  identity {
    type = "SystemAssigned"
  }

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_storage_account" "backups" {
  name                     = "${var.project_name}${var.environment}backups${random_id.main.hex}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "production" ? "GRS" : "LRS"
  account_kind             = "StorageV2"
  access_tier              = "Cool"

  blob_properties {
    versioning_enabled       = true
    
    delete_retention_policy {
      days = var.environment == "production" ? 90 : 30
    }
  }

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.aks.id]
    bypass                     = ["AzureServices"]
  }

  identity {
    type = "SystemAssigned"
  }

  tags = azurerm_resource_group.main.tags
}

# Application Gateway
resource "azurerm_public_ip" "application_gateway" {
  name                = "${var.project_name}-${var.environment}-agw-pip"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = var.availability_zones

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_application_gateway" "main" {
  name                = "${var.project_name}-${var.environment}-agw"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = var.application_gateway_capacity
  }

  zones = var.availability_zones

  gateway_ip_configuration {
    name      = "gateway-ip-configuration"
    subnet_id = azurerm_subnet.application_gateway.id
  }

  frontend_port {
    name = "port-80"
    port = 80
  }

  frontend_port {
    name = "port-443"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "public-frontend-ip"
    public_ip_address_id = azurerm_public_ip.application_gateway.id
  }

  backend_address_pool {
    name = "aks-backend-pool"
  }

  backend_http_settings {
    name                  = "http-settings"
    cookie_based_affinity = "Disabled"
    path                  = "/"
    port                  = 80
    protocol              = "Http"
    request_timeout       = 60
  }

  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "public-frontend-ip"
    frontend_port_name             = "port-80"
    protocol                       = "Http"
  }

  request_routing_rule {
    name                       = "http-routing-rule"
    rule_type                  = "Basic"
    http_listener_name         = "http-listener"
    backend_address_pool_name  = "aks-backend-pool"
    backend_http_settings_name = "http-settings"
    priority                   = 100
  }

  waf_configuration {
    enabled          = true
    firewall_mode    = "Prevention"
    rule_set_type    = "OWASP"
    rule_set_version = "3.2"
  }

  tags = azurerm_resource_group.main.tags
}

# Role assignments
resource "azurerm_role_assignment" "aks_acr" {
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                           = azurerm_container_registry.main.id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "aks_network_contributor" {
  principal_id                     = azurerm_kubernetes_cluster.main.identity[0].principal_id
  role_definition_name             = "Network Contributor"
  scope                           = azurerm_virtual_network.main.id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "aks_monitoring" {
  principal_id                     = azurerm_kubernetes_cluster.main.identity[0].principal_id
  role_definition_name             = "Monitoring Metrics Publisher"
  scope                           = azurerm_resource_group.main.id
  skip_service_principal_aad_check = true
}

# Key Vault Secrets
resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-password"
  value        = var.postgres_admin_password
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.terraform]

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_key_vault_secret" "redis_connection_string" {
  name         = "redis-connection-string"
  value        = azurerm_redis_cache.main.primary_connection_string
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.terraform]

  tags = azurerm_resource_group.main.tags
}

# Monitor Action Groups
resource "azurerm_monitor_action_group" "main" {
  name                = "${var.project_name}-${var.environment}-alerts"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "alerts"

  email_receiver {
    name          = "admin"
    email_address = var.admin_email
  }

  webhook_receiver {
    name                    = "slack"
    service_uri             = var.slack_webhook_url
    use_common_alert_schema = true
  }

  tags = azurerm_resource_group.main.tags
}

# Monitor Metric Alerts
resource "azurerm_monitor_metric_alert" "aks_cpu" {
  name                = "${var.project_name}-${var.environment}-aks-cpu-alert"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_kubernetes_cluster.main.id]
  description         = "Alert when AKS CPU usage is high"
  severity            = 2

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "node_cpu_usage_percentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80

    dimension {
      name     = "node"
      operator = "Include"
      values   = ["*"]
    }
  }

  action {
    action_group_id = azurerm_monitor_action_group.main.id
  }

  frequency   = "PT1M"
  window_size = "PT5M"

  tags = azurerm_resource_group.main.tags
}

resource "azurerm_monitor_metric_alert" "postgres_cpu" {
  name                = "${var.project_name}-${var.environment}-postgres-cpu-alert"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_postgresql_flexible_server.main.id]
  description         = "Alert when PostgreSQL CPU usage is high"
  severity            = 2

  criteria {
    metric_namespace = "Microsoft.DBforPostgreSQL/flexibleServers"
    metric_name      = "cpu_percent"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.main.id
  }

  frequency   = "PT1M"
  window_size = "PT5M"

  tags = azurerm_resource_group.main.tags
}

# Backup vault for AKS
resource "azurerm_data_protection_backup_vault" "main" {
  name                = "${var.project_name}-${var.environment}-backup-vault"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  datastore_type      = "VaultStore"
  redundancy          = var.environment == "production" ? "GeoRedundant" : "LocallyRedundant"

  identity {
    type = "SystemAssigned"
  }

  tags = azurerm_resource_group.main.tags
}