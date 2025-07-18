#!/bin/bash

# EKS Node User Data Script
# This script configures the EKS worker nodes

set -o xtrace

# Bootstrap the node to join the EKS cluster
/etc/eks/bootstrap.sh ${cluster_name} ${bootstrap_extra_args}

# Install additional packages
yum update -y
yum install -y amazon-cloudwatch-agent

# Configure CloudWatch agent
cat <<EOF > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "cwagent"
  },
  "metrics": {
    "namespace": "CWAgent",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          "cpu_usage_idle",
          "cpu_usage_iowait",
          "cpu_usage_user",
          "cpu_usage_system"
        ],
        "metrics_collection_interval": 60,
        "totalcpu": false
      },
      "disk": {
        "measurement": [
          "used_percent"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "diskio": {
        "measurement": [
          "io_time"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "mem": {
        "measurement": [
          "mem_used_percent"
        ],
        "metrics_collection_interval": 60
      },
      "netstat": {
        "measurement": [
          "tcp_established",
          "tcp_time_wait"
        ],
        "metrics_collection_interval": 60
      },
      "swap": {
        "measurement": [
          "swap_used_percent"
        ],
        "metrics_collection_interval": 60
      }
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/messages",
            "log_group_name": "/aws/eks/${cluster_name}/node",
            "log_stream_name": "{instance_id}/messages"
          },
          {
            "file_path": "/var/log/dmesg",
            "log_group_name": "/aws/eks/${cluster_name}/node",
            "log_stream_name": "{instance_id}/dmesg"
          }
        ]
      }
    }
  }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s

# Install SSM agent for session manager
yum install -y amazon-ssm-agent
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent

# Configure node labels
kubectl label node $(hostname) node.kubernetes.io/instance-type=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)

# Configure disk space monitoring
cat <<EOF > /opt/disk-cleanup.sh
#!/bin/bash
# Clean up disk space when usage exceeds 80%
DISK_USAGE=\$(df / | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{ print \$5 }' | sed 's/%//g')
if [ \$DISK_USAGE -gt 80 ]; then
    # Clean docker images
    docker system prune -f
    # Clean package cache
    yum clean all
    # Clean logs older than 7 days
    find /var/log -name "*.log" -type f -mtime +7 -exec rm -f {} \;
fi
EOF

chmod +x /opt/disk-cleanup.sh

# Add cron job for disk cleanup
echo "0 */6 * * * /opt/disk-cleanup.sh" | crontab -

# Configure log rotation
cat <<EOF > /etc/logrotate.d/eks-node
/var/log/kubernetes/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 root root
}
EOF

# Signal that the instance is ready
/opt/aws/bin/cfn-signal -e \$? --stack \${AWS::StackName} --resource NodeGroup --region \${AWS::Region}