# EKS Module Outputs

output "cluster_id" {
  description = "The ID of the EKS cluster. Note: currently a value is returned only for local EKS clusters created on Outposts"
  value       = aws_eks_cluster.main.cluster_id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = aws_eks_cluster.main.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_endpoint" {
  description = "Endpoint for your Kubernetes API server"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = aws_eks_cluster.main.version
}

output "cluster_platform_version" {
  description = "Platform version for the cluster"
  value       = aws_eks_cluster.main.platform_version
}

output "cluster_status" {
  description = "Status of the EKS cluster. One of `CREATING`, `ACTIVE`, `DELETING`, `FAILED`"
  value       = aws_eks_cluster.main.status
}

output "cluster_security_group_id" {
  description = "ID of the cluster security group"
  value       = aws_security_group.cluster.id
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.cluster.name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "cluster_iam_role_unique_id" {
  description = "Stable and unique string identifying the IAM role"
  value       = aws_iam_role.cluster.unique_id
}

output "cluster_identity_providers" {
  description = "Map of attribute maps for all EKS identity providers enabled"
  value       = aws_eks_cluster.main.identity
}

output "cluster_issuer" {
  description = "The OpenID Connect identity provider for the cluster"
  value       = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

output "cluster_tls_certificate_sha1_fingerprint" {
  description = "The SHA1 fingerprint of the public key of the cluster's certificate"
  value       = data.tls_certificate.cluster.certificates[0].sha1_fingerprint
}

output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value = {
    for k, v in aws_eks_node_group.main : k => {
      arn           = v.arn
      node_group_id = v.id
      status        = v.status
      capacity_type = v.capacity_type
      scaling_config = v.scaling_config
      instance_types = v.instance_types
      ami_type      = v.ami_type
      release_version = v.release_version
      version       = v.version
      asg_name      = v.resources[0].autoscaling_groups[0].name
      asg_arn       = "arn:aws:autoscaling:${data.aws_partition.current.dns_suffix}:${data.aws_caller_identity.current.account_id}:autoScalingGroup:*:autoScalingGroupName/${v.resources[0].autoscaling_groups[0].name}"
    }
  }
}

output "node_security_group_id" {
  description = "ID of the node shared security group"
  value       = aws_security_group.node_group_one.id
}

output "node_security_group_arn" {
  description = "Amazon Resource Name (ARN) of the node shared security group"
  value       = aws_security_group.node_group_one.arn
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = var.enable_irsa ? aws_iam_openid_connect_provider.cluster[0].arn : null
}

output "cluster_tls_certificate" {
  description = "The TLS certificate of the cluster"
  value       = data.tls_certificate.cluster.certificates[0].cert_pem
}

output "cloudwatch_log_group_name" {
  description = "Name of cloudwatch log group created"
  value       = aws_cloudwatch_log_group.cluster.name
}

output "cloudwatch_log_group_arn" {
  description = "Arn of cloudwatch log group created"
  value       = aws_cloudwatch_log_group.cluster.arn
}

output "aws_load_balancer_controller_role_arn" {
  description = "The ARN of the AWS Load Balancer Controller IAM role"
  value       = var.enable_irsa ? aws_iam_role.aws_load_balancer_controller[0].arn : null
}

output "cluster_autoscaler_role_arn" {
  description = "The ARN of the Cluster Autoscaler IAM role"
  value       = var.enable_irsa ? aws_iam_role.cluster_autoscaler[0].arn : null
}

output "cluster_encryption_config" {
  description = "Cluster encryption configuration"
  value       = aws_eks_cluster.main.encryption_config
}

output "cluster_addons" {
  description = "Map of attribute maps for all EKS cluster addons enabled"
  value = {
    for k, v in aws_eks_addon.addons : k => {
      arn               = v.arn
      addon_name        = v.addon_name
      addon_version     = v.addon_version
      status            = v.status
      created_at        = v.created_at
      modified_at       = v.modified_at
      service_account_role_arn = v.service_account_role_arn
    }
  }
}

# Useful outputs for kubectl configuration
output "kubectl_config" {
  description = "kubectl config for EKS cluster"
  value = {
    apiVersion = "v1"
    kind       = "Config"
    current-context = aws_eks_cluster.main.name
    contexts = [{
      name = aws_eks_cluster.main.name
      context = {
        cluster = aws_eks_cluster.main.name
        user    = aws_eks_cluster.main.name
      }
    }]
    clusters = [{
      name = aws_eks_cluster.main.name
      cluster = {
        server                     = aws_eks_cluster.main.endpoint
        certificate-authority-data = aws_eks_cluster.main.certificate_authority[0].data
      }
    }]
    users = [{
      name = aws_eks_cluster.main.name
      user = {
        exec = {
          apiVersion = "client.authentication.k8s.io/v1beta1"
          command    = "aws"
          args = [
            "eks",
            "get-token",
            "--cluster-name",
            aws_eks_cluster.main.name
          ]
        }
      }
    }]
  }
}