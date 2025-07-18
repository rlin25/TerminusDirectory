# GCP Infrastructure for Rental ML System
# Production-ready multi-region deployment with auto-scaling and monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.84"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 4.84"
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

  backend "gcs" {
    bucket = var.terraform_state_bucket
    prefix = "rental-ml-system/terraform.tfstate"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone

  default_labels = {
    project     = var.project_name
    environment = var.environment
    managed-by  = "terraform"
    team        = "ml-platform"
  }
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Data sources
data "google_client_config" "default" {}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "containerregistry.googleapis.com",
    "cloudsql.googleapis.com",
    "redis.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudkms.googleapis.com",
    "dns.googleapis.com",
    "certificatemanager.googleapis.com",
    "networkservices.googleapis.com",
    "servicemesh.googleapis.com",
    "bigquery.googleapis.com",
    "dataflow.googleapis.com",
    "pubsub.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# VPC Network
resource "google_compute_network" "main" {
  name                    = "${var.project_name}-${var.environment}-vpc"
  auto_create_subnetworks = false
  mtu                     = 1460

  depends_on = [google_project_service.apis]
}

# Subnets
resource "google_compute_subnetwork" "private" {
  count = length(var.private_subnet_cidrs)

  name                     = "${var.project_name}-${var.environment}-private-${count.index + 1}"
  ip_cidr_range           = var.private_subnet_cidrs[count.index]
  region                  = var.region
  network                 = google_compute_network.main.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pod_subnet_cidrs[count.index]
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.service_subnet_cidrs[count.index]
  }

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_subnetwork" "public" {
  count = length(var.public_subnet_cidrs)

  name          = "${var.project_name}-${var.environment}-public-${count.index + 1}"
  ip_cidr_range = var.public_subnet_cidrs[count.index]
  region        = var.region
  network       = google_compute_network.main.id

  log_config {
    aggregation_interval = "INTERVAL_10_MIN"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

# Cloud Router for NAT
resource "google_compute_router" "main" {
  name    = "${var.project_name}-${var.environment}-router"
  region  = var.region
  network = google_compute_network.main.id
}

# NAT Gateway
resource "google_compute_router_nat" "main" {
  name                               = "${var.project_name}-${var.environment}-nat"
  router                            = google_compute_router.main.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = "${var.project_name}-${var.environment}-cluster"
  location = var.region

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.private[0].name

  # Enable network policy
  network_policy {
    enabled = true
  }

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable IP aliasing
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block = var.master_ipv4_cidr_block
  }

  # Master authorized networks
  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.authorized_networks
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  # Enable logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Enable cluster autoscaling
  cluster_autoscaling {
    enabled = true
    
    auto_provisioning_defaults {
      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
      ]
      
      management {
        auto_repair  = true
        auto_upgrade = true
      }
    }

    resource_limits {
      resource_type = "cpu"
      minimum       = var.cluster_autoscaling.min_cpu
      maximum       = var.cluster_autoscaling.max_cpu
    }

    resource_limits {
      resource_type = "memory"
      minimum       = var.cluster_autoscaling.min_memory
      maximum       = var.cluster_autoscaling.max_memory
    }
  }

  # Enable binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Database encryption
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.gke.id
  }

  # Release channel
  release_channel {
    channel = var.gke_release_channel
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
    
    istio_config {
      disabled = false
      auth     = "AUTH_MUTUAL_TLS"
    }
    
    cloudrun_config {
      disabled = false
    }
    
    gcp_filestore_csi_driver_config {
      enabled = true
    }
    
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  # Maintenance policy
  maintenance_policy {
    recurring_window {
      start_time = var.maintenance_start_time
      end_time   = var.maintenance_end_time
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }

  depends_on = [
    google_project_service.apis,
    google_compute_subnetwork.private
  ]
}

# KMS key for GKE encryption
resource "google_kms_key_ring" "gke" {
  name     = "${var.project_name}-${var.environment}-gke"
  location = var.region

  depends_on = [google_project_service.apis]
}

resource "google_kms_crypto_key" "gke" {
  name     = "gke-encryption-key"
  key_ring = google_kms_key_ring.gke.id

  lifecycle {
    prevent_destroy = true
  }
}

# GKE Node Pools
resource "google_container_node_pool" "main" {
  name       = "main-pool"
  location   = var.region
  cluster    = google_container_cluster.main.name
  node_count = var.node_pool_config.initial_node_count

  autoscaling {
    min_node_count = var.node_pool_config.min_node_count
    max_node_count = var.node_pool_config.max_node_count
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = var.node_pool_config.max_surge
    max_unavailable = var.node_pool_config.max_unavailable
  }

  node_config {
    preemptible  = var.node_pool_config.preemptible
    machine_type = var.node_pool_config.machine_type
    disk_size_gb = var.node_pool_config.disk_size_gb
    disk_type    = var.node_pool_config.disk_type
    image_type   = "COS_CONTAINERD"

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node-pool   = "main"
    }

    # Taints for workload isolation
    dynamic "taint" {
      for_each = var.node_pool_config.taints
      content {
        key    = taint.value.key
        value  = taint.value.value
        effect = taint.value.effect
      }
    }

    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Shielded instance config
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    # Boot disk encryption
    gcfs_config {
      enabled = true
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

# GPU Node Pool for ML workloads
resource "google_container_node_pool" "gpu" {
  count = var.enable_gpu_nodes ? 1 : 0

  name       = "gpu-pool"
  location   = var.region
  cluster    = google_container_cluster.main.name
  node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = var.gpu_node_pool_config.max_node_count
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    preemptible  = var.gpu_node_pool_config.preemptible
    machine_type = var.gpu_node_pool_config.machine_type
    disk_size_gb = var.gpu_node_pool_config.disk_size_gb
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = var.gpu_node_pool_config.gpu_type
      count = var.gpu_node_pool_config.gpu_count
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node-pool   = "gpu"
      accelerator = var.gpu_node_pool_config.gpu_type
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.project_name}-${var.environment}-gke-nodes"
  display_name = "GKE Nodes Service Account"
  description  = "Service account for GKE nodes"
}

# IAM bindings for GKE nodes
resource "google_project_iam_member" "gke_nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/artifactregistry.reader"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Cloud SQL instance
resource "google_sql_database_instance" "main" {
  name             = "${var.project_name}-${var.environment}-postgres"
  database_version = "POSTGRES_15"
  region          = var.region

  settings {
    tier                        = var.cloudsql_tier
    disk_autoresize            = true
    disk_autoresize_limit      = var.cloudsql_max_disk_size
    disk_size                  = var.cloudsql_disk_size
    disk_type                  = "PD_SSD"
    deletion_protection_enabled = var.environment == "production"

    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = var.environment == "production" ? 30 : 7
      }
      transaction_log_retention_days = var.environment == "production" ? 7 : 3
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
      require_ssl     = true

      dynamic "authorized_networks" {
        for_each = var.authorized_networks
        content {
          name  = authorized_networks.value.display_name
          value = authorized_networks.value.cidr_block
        }
      }
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }

    database_flags {
      name  = "log_connections"
      value = "on"
    }

    database_flags {
      name  = "log_disconnections"
      value = "on"
    }

    database_flags {
      name  = "log_lock_waits"
      value = "on"
    }

    database_flags {
      name  = "log_temp_files"
      value = "0"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }

    maintenance_window {
      day  = 7  # Sunday
      hour = 3
    }
  }

  depends_on = [
    google_project_service.apis,
    google_service_networking_connection.private_vpc_connection
  ]
}

# Private VPC connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.project_name}-${var.environment}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]

  depends_on = [google_project_service.apis]
}

# Cloud SQL Database
resource "google_sql_database" "main" {
  name     = var.database_name
  instance = google_sql_database_instance.main.name
}

# Cloud SQL User
resource "google_sql_user" "main" {
  name     = var.database_username
  instance = google_sql_database_instance.main.name
  password = var.database_password
}

# Redis instance
resource "google_redis_instance" "main" {
  name           = "${var.project_name}-${var.environment}-redis"
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region

  authorized_network = google_compute_network.main.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version     = var.redis_version
  display_name      = "${var.project_name} ${var.environment} Redis"
  reserved_ip_range = var.redis_reserved_ip_range

  # High availability for production
  replica_count             = var.environment == "production" ? 1 : 0
  read_replicas_mode       = var.environment == "production" ? "READ_REPLICAS_ENABLED" : "READ_REPLICAS_DISABLED"
  
  auth_enabled = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
      }
    }
  }

  depends_on = [google_project_service.apis]
}

# Cloud Storage buckets
resource "google_storage_bucket" "ml_models" {
  name          = "${var.project_name}-${var.environment}-ml-models-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
}

resource "google_storage_bucket" "app_data" {
  name          = "${var.project_name}-${var.environment}-app-data-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
}

resource "google_storage_bucket" "backups" {
  name          = "${var.project_name}-${var.environment}-backups-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = var.environment != "production"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "ARCHIVE"
    }
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
}

# Random ID for bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# KMS key for storage encryption
resource "google_kms_crypto_key" "storage" {
  name     = "storage-encryption-key"
  key_ring = google_kms_key_ring.gke.id

  lifecycle {
    prevent_destroy = true
  }
}

# Load Balancer
resource "google_compute_global_address" "default" {
  name = "${var.project_name}-${var.environment}-lb-ip"
}

# Health check
resource "google_compute_health_check" "default" {
  name               = "${var.project_name}-${var.environment}-health-check"
  check_interval_sec = 30
  timeout_sec        = 5

  http_health_check {
    port         = 8000
    request_path = "/health"
  }
}

# Backend service
resource "google_compute_backend_service" "default" {
  name                  = "${var.project_name}-${var.environment}-backend"
  protocol              = "HTTP"
  timeout_sec           = 30
  health_checks         = [google_compute_health_check.default.id]
  load_balancing_scheme = "EXTERNAL"

  backend {
    group           = google_container_cluster.main.node_pool[0].instance_group_urls[0]
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# URL map
resource "google_compute_url_map" "default" {
  name            = "${var.project_name}-${var.environment}-url-map"
  default_service = google_compute_backend_service.default.id

  host_rule {
    hosts        = ["api.${var.domain_name}"]
    path_matcher = "api"
  }

  path_matcher {
    name            = "api"
    default_service = google_compute_backend_service.default.id

    path_rule {
      paths   = ["/api/*"]
      service = google_compute_backend_service.default.id
    }
  }
}

# SSL certificate
resource "google_compute_managed_ssl_certificate" "default" {
  name = "${var.project_name}-${var.environment}-ssl-cert"

  managed {
    domains = [
      var.domain_name,
      "api.${var.domain_name}",
      "monitoring.${var.domain_name}"
    ]
  }
}

# HTTPS proxy
resource "google_compute_target_https_proxy" "default" {
  name             = "${var.project_name}-${var.environment}-https-proxy"
  url_map          = google_compute_url_map.default.id
  ssl_certificates = [google_compute_managed_ssl_certificate.default.id]
}

# Global forwarding rule
resource "google_compute_global_forwarding_rule" "default" {
  name       = "${var.project_name}-${var.environment}-forwarding-rule"
  target     = google_compute_target_https_proxy.default.id
  port_range = "443"
  ip_address = google_compute_global_address.default.address
}

# DNS records
resource "google_dns_record_set" "main" {
  count = var.enable_dns ? 1 : 0

  name = "${var.domain_name}."
  type = "A"
  ttl  = 300

  managed_zone = var.dns_zone_name

  rrdatas = [google_compute_global_address.default.address]
}

resource "google_dns_record_set" "api" {
  count = var.enable_dns ? 1 : 0

  name = "api.${var.domain_name}."
  type = "A"
  ttl  = 300

  managed_zone = var.dns_zone_name

  rrdatas = [google_compute_global_address.default.address]
}

# Secret Manager secrets
resource "google_secret_manager_secret" "database_password" {
  secret_id = "${var.project_name}-${var.environment}-database-password"

  replication {
    automatic = true
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "database_password" {
  secret      = google_secret_manager_secret.database_password.id
  secret_data = var.database_password
}

resource "google_secret_manager_secret" "redis_auth" {
  secret_id = "${var.project_name}-${var.environment}-redis-auth"

  replication {
    automatic = true
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "redis_auth" {
  secret      = google_secret_manager_secret.redis_auth.id
  secret_data = google_redis_instance.main.auth_string
}

# Monitoring workspace
resource "google_monitoring_notification_channel" "email" {
  display_name = "${var.project_name} ${var.environment} Email Alerts"
  type         = "email"

  labels = {
    email_address = var.notification_email
  }

  depends_on = [google_project_service.apis]
}

# Alerting policies
resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "${var.project_name} ${var.environment} High CPU"
  combiner     = "OR"

  conditions {
    display_name = "High CPU usage"

    condition_threshold {
      filter          = "resource.type=\"gke_container\""
      duration        = "300s"
      comparison      = "GREATER_THAN"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }

  depends_on = [google_project_service.apis]
}

# Firewall rules
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.project_name}-${var.environment}-allow-internal"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.vpc_cidr]
}

resource "google_compute_firewall" "allow_lb_health_check" {
  name    = "${var.project_name}-${var.environment}-allow-lb-health-check"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "8080", "80", "443"]
  }

  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = ["gke-node"]
}

# IAM for Workload Identity
resource "google_service_account" "workload_identity" {
  for_each = var.workload_identity_accounts

  account_id   = each.key
  display_name = each.value.display_name
  description  = each.value.description
}

resource "google_service_account_iam_binding" "workload_identity" {
  for_each = var.workload_identity_accounts

  service_account_id = google_service_account.workload_identity[each.key].name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${each.value.namespace}/${each.value.ksa_name}]"
  ]
}

# Binary Authorization policy
resource "google_binary_authorization_policy" "policy" {
  admission_whitelist_patterns {
    name_pattern = "gcr.io/${var.project_id}/*"
  }

  default_admission_rule {
    evaluation_mode  = "REQUIRE_ATTESTATION"
    enforcement_mode = "ENFORCED_BLOCK_AND_AUDIT_LOG"

    require_attestations_by = [
      google_binary_authorization_attestor.attestor.name
    ]
  }

  depends_on = [google_project_service.apis]
}

resource "google_binary_authorization_attestor" "attestor" {
  name = "${var.project_name}-${var.environment}-attestor"

  attestation_authority_note {
    note_reference = google_container_analysis_note.note.name

    public_keys {
      ascii_armored_pgp_public_key = var.binary_authorization_public_key
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_container_analysis_note" "note" {
  name = "${var.project_name}-${var.environment}-attestor-note"

  attestation_authority {
    hint {
      human_readable_name = "${var.project_name} ${var.environment} Attestor"
    }
  }

  depends_on = [google_project_service.apis]
}