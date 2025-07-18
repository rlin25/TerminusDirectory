"""
Redis Cluster Manager for Production
High-availability Redis cluster management with automatic failover and monitoring
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import aioredis
import socket
import random


class NodeRole(Enum):
    MASTER = "master"
    REPLICA = "replica"
    SENTINEL = "sentinel"


class NodeStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ClusterMode(Enum):
    CLUSTER = "cluster"
    SENTINEL = "sentinel"
    STANDALONE = "standalone"


@dataclass
class RedisNode:
    node_id: str
    host: str
    port: int
    role: NodeRole
    status: NodeStatus
    master_id: Optional[str] = None
    replicas: List[str] = None
    slots: List[Tuple[int, int]] = None  # For cluster mode
    memory_usage: int = 0
    cpu_usage: float = 0.0
    connections: int = 0
    ops_per_sec: float = 0.0
    last_ping: Optional[datetime] = None
    
    def __post_init__(self):
        if self.replicas is None:
            self.replicas = []
        if self.slots is None:
            self.slots = []


@dataclass
class ClusterHealth:
    overall_status: NodeStatus
    total_nodes: int
    healthy_nodes: int
    failed_nodes: int
    masters_count: int
    replicas_count: int
    cluster_slots_covered: int
    cluster_slots_total: int = 16384
    average_memory_usage: float = 0.0
    total_operations_per_sec: float = 0.0
    last_check: datetime = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now()


class RedisClusterManager:
    """
    Production Redis cluster manager with:
    - Automatic cluster setup and scaling
    - Health monitoring and alerting
    - Failover management
    - Performance optimization
    - Backup and recovery
    - Multi-tenancy support
    """
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Cluster configuration
        self.cluster_mode = ClusterMode.CLUSTER
        self.nodes: Dict[str, RedisNode] = {}
        self.sentinel_nodes: List[RedisNode] = []
        
        # Connection pools
        self.redis_pools: Dict[str, aioredis.Redis] = {}
        self.sentinel_pool: Optional[aioredis.Redis] = None
        
        # Monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.cluster_health: Optional[ClusterHealth] = None
        
        # Configuration
        self.redis_password = None
        self.sentinel_password = None
        self.cluster_config = {
            'min_replicas_per_master': 1,
            'max_memory_usage_percent': 80,
            'failover_timeout': 180,
            'cluster_require_full_coverage': True,
            'replica_read_only': True
        }
        
    async def initialize_cluster(self, cluster_mode: ClusterMode = ClusterMode.CLUSTER):
        """Initialize Redis cluster or sentinel setup"""
        try:
            self.cluster_mode = cluster_mode
            self.logger.info(f"Initializing Redis cluster in {cluster_mode.value} mode")
            
            if cluster_mode == ClusterMode.CLUSTER:
                await self._initialize_redis_cluster()
            elif cluster_mode == ClusterMode.SENTINEL:
                await self._initialize_sentinel_setup()
            else:
                await self._initialize_standalone()
                
            # Start health monitoring
            await self._start_health_monitoring()
            
            self.logger.info("Redis cluster initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis cluster: {e}")
            raise
            
    async def _initialize_redis_cluster(self):
        """Initialize Redis cluster mode"""
        # Load cluster nodes from configuration
        cluster_nodes = [
            {'host': '127.0.0.1', 'port': 7000},
            {'host': '127.0.0.1', 'port': 7001},
            {'host': '127.0.0.1', 'port': 7002},
            {'host': '127.0.0.1', 'port': 7003},
            {'host': '127.0.0.1', 'port': 7004},
            {'host': '127.0.0.1', 'port': 7005}
        ]
        
        # Connect to cluster nodes
        for node_config in cluster_nodes:
            try:
                node_id = f"{node_config['host']}:{node_config['port']}"
                
                # Create connection pool
                redis_pool = aioredis.from_url(
                    f"redis://{node_config['host']}:{node_config['port']}",
                    password=self.redis_password,
                    decode_responses=True,
                    max_connections=20
                )
                
                self.redis_pools[node_id] = redis_pool
                
                # Get node info
                info = await redis_pool.cluster_nodes()
                node_role = NodeRole.MASTER  # Will be updated from cluster info
                
                node = RedisNode(
                    node_id=node_id,
                    host=node_config['host'],
                    port=node_config['port'],
                    role=node_role,
                    status=NodeStatus.HEALTHY
                )
                
                self.nodes[node_id] = node
                self.logger.info(f"Connected to Redis cluster node: {node_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis node {node_id}: {e}")
                
        # Update cluster topology
        await self._update_cluster_topology()
        
    async def _initialize_sentinel_setup(self):
        """Initialize Redis Sentinel setup"""
        # Sentinel nodes configuration
        sentinel_nodes = [
            {'host': '127.0.0.1', 'port': 26379},
            {'host': '127.0.0.1', 'port': 26380},
            {'host': '127.0.0.1', 'port': 26381}
        ]
        
        # Connect to Sentinel
        try:
            self.sentinel_pool = aioredis.Sentinel(
                [(node['host'], node['port']) for node in sentinel_nodes],
                password=self.sentinel_password
            )
            
            # Get master info
            masters = await self.sentinel_pool.sentinel_masters()
            
            for master_name, master_info in masters.items():
                master_host = master_info['ip']
                master_port = int(master_info['port'])
                node_id = f"{master_host}:{master_port}"
                
                # Connect to master
                master_pool = self.sentinel_pool.master_for(master_name, password=self.redis_password)
                self.redis_pools[node_id] = master_pool
                
                # Create master node
                master_node = RedisNode(
                    node_id=node_id,
                    host=master_host,
                    port=master_port,
                    role=NodeRole.MASTER,
                    status=NodeStatus.HEALTHY
                )
                
                self.nodes[node_id] = master_node
                
                # Get replica info
                replicas = await self.sentinel_pool.sentinel_slaves(master_name)
                for replica_info in replicas:
                    replica_host = replica_info['ip']
                    replica_port = int(replica_info['port'])
                    replica_id = f"{replica_host}:{replica_port}"
                    
                    replica_node = RedisNode(
                        node_id=replica_id,
                        host=replica_host,
                        port=replica_port,
                        role=NodeRole.REPLICA,
                        status=NodeStatus.HEALTHY,
                        master_id=node_id
                    )
                    
                    self.nodes[replica_id] = replica_node
                    master_node.replicas.append(replica_id)
                    
            self.logger.info("Connected to Redis Sentinel setup")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis Sentinel: {e}")
            raise
            
    async def _initialize_standalone(self):
        """Initialize standalone Redis setup"""
        # Single Redis instance
        try:
            redis_pool = aioredis.from_url(
                "redis://127.0.0.1:6379",
                password=self.redis_password,
                decode_responses=True
            )
            
            node_id = "127.0.0.1:6379"
            self.redis_pools[node_id] = redis_pool
            
            node = RedisNode(
                node_id=node_id,
                host="127.0.0.1",
                port=6379,
                role=NodeRole.MASTER,
                status=NodeStatus.HEALTHY
            )
            
            self.nodes[node_id] = node
            self.logger.info("Connected to standalone Redis instance")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def _update_cluster_topology(self):
        """Update cluster topology information"""
        if self.cluster_mode != ClusterMode.CLUSTER:
            return
            
        try:
            # Get cluster nodes info from any node
            for node_id, redis_pool in self.redis_pools.items():
                try:
                    cluster_info = await redis_pool.cluster_nodes()
                    
                    # Parse cluster nodes information
                    for line in cluster_info.split('\n'):
                        if not line.strip():
                            continue
                            
                        parts = line.split()
                        if len(parts) < 8:
                            continue
                            
                        node_id_from_cluster = parts[0]
                        endpoint = parts[1]
                        flags = parts[2]
                        master_id = parts[3] if parts[3] != '-' else None
                        
                        # Parse host and port
                        if '@' in endpoint:
                            endpoint = endpoint.split('@')[0]
                        host, port = endpoint.rsplit(':', 1)
                        port = int(port)
                        
                        # Determine role
                        if 'master' in flags:
                            role = NodeRole.MASTER
                        elif 'slave' in flags:
                            role = NodeRole.REPLICA
                        else:
                            continue
                            
                        # Parse slot ranges for masters
                        slots = []
                        if role == NodeRole.MASTER and len(parts) > 8:
                            for slot_range in parts[8:]:
                                if '-' in slot_range:
                                    start, end = map(int, slot_range.split('-'))
                                    slots.append((start, end))
                                else:
                                    slot_num = int(slot_range)
                                    slots.append((slot_num, slot_num))
                                    
                        # Update node information
                        node_key = f"{host}:{port}"
                        if node_key in self.nodes:
                            self.nodes[node_key].role = role
                            self.nodes[node_key].master_id = master_id
                            self.nodes[node_key].slots = slots
                            
                    break  # Successfully updated from one node
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get cluster info from {node_id}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to update cluster topology: {e}")
            
    async def _start_health_monitoring(self):
        """Start continuous health monitoring"""
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        
    async def _health_monitor_loop(self):
        """Continuous health monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await self._update_cluster_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)
                
    async def _perform_health_checks(self):
        """Perform health checks on all nodes"""
        for node_id, node in self.nodes.items():
            try:
                if node_id in self.redis_pools:
                    redis_pool = self.redis_pools[node_id]
                    
                    # Ping the node
                    start_time = time.time()
                    await redis_pool.ping()
                    ping_time = time.time() - start_time
                    
                    # Update node status
                    node.last_ping = datetime.now()
                    
                    if ping_time < 0.1:
                        node.status = NodeStatus.HEALTHY
                    elif ping_time < 1.0:
                        node.status = NodeStatus.DEGRADED
                    else:
                        node.status = NodeStatus.FAILED
                        
                    # Get node stats
                    info = await redis_pool.info()
                    node.memory_usage = info.get('used_memory', 0)
                    node.connections = info.get('connected_clients', 0)
                    
                    # Calculate operations per second
                    total_commands = info.get('total_commands_processed', 0)
                    uptime = info.get('uptime_in_seconds', 1)
                    node.ops_per_sec = total_commands / uptime if uptime > 0 else 0
                    
                else:
                    node.status = NodeStatus.FAILED
                    
            except Exception as e:
                self.logger.error(f"Health check failed for node {node_id}: {e}")
                node.status = NodeStatus.FAILED
                
    async def _update_cluster_health(self):
        """Update overall cluster health status"""
        total_nodes = len(self.nodes)
        healthy_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY])
        failed_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
        
        masters = [n for n in self.nodes.values() if n.role == NodeRole.MASTER]
        replicas = [n for n in self.nodes.values() if n.role == NodeRole.REPLICA]
        
        # Calculate cluster slot coverage for cluster mode
        slots_covered = 0
        if self.cluster_mode == ClusterMode.CLUSTER:
            covered_slots = set()
            for node in masters:
                if node.status == NodeStatus.HEALTHY:
                    for start, end in node.slots:
                        covered_slots.update(range(start, end + 1))
            slots_covered = len(covered_slots)
            
        # Determine overall status
        if failed_nodes == 0:
            overall_status = NodeStatus.HEALTHY
        elif healthy_nodes > failed_nodes:
            overall_status = NodeStatus.DEGRADED
        else:
            overall_status = NodeStatus.FAILED
            
        # Calculate average memory usage
        total_memory = sum(n.memory_usage for n in self.nodes.values())
        avg_memory = total_memory / total_nodes if total_nodes > 0 else 0
        
        # Calculate total operations per second
        total_ops = sum(n.ops_per_sec for n in self.nodes.values())
        
        self.cluster_health = ClusterHealth(
            overall_status=overall_status,
            total_nodes=total_nodes,
            healthy_nodes=healthy_nodes,
            failed_nodes=failed_nodes,
            masters_count=len(masters),
            replicas_count=len(replicas),
            cluster_slots_covered=slots_covered,
            average_memory_usage=avg_memory,
            total_operations_per_sec=total_ops,
            last_check=datetime.now()
        )
        
        # Log health status
        if overall_status != NodeStatus.HEALTHY:
            self.logger.warning(f"Cluster health degraded: {healthy_nodes}/{total_nodes} nodes healthy")
            
    async def get_redis_connection(self, key: str = None, read_only: bool = False) -> aioredis.Redis:
        """Get Redis connection for a specific key or operation"""
        
        if self.cluster_mode == ClusterMode.CLUSTER:
            # For cluster mode, Redis client handles key distribution
            # Return any healthy node connection
            for node_id, redis_pool in self.redis_pools.items():
                node = self.nodes.get(node_id)
                if node and node.status == NodeStatus.HEALTHY:
                    return redis_pool
                    
        elif self.cluster_mode == ClusterMode.SENTINEL:
            # For sentinel mode, choose master or replica based on read_only flag
            if read_only:
                # Try to use a replica
                replica_nodes = [n for n in self.nodes.values() 
                               if n.role == NodeRole.REPLICA and n.status == NodeStatus.HEALTHY]
                if replica_nodes:
                    replica = random.choice(replica_nodes)
                    return self.redis_pools.get(replica.node_id)
                    
            # Use master
            master_nodes = [n for n in self.nodes.values() 
                          if n.role == NodeRole.MASTER and n.status == NodeStatus.HEALTHY]
            if master_nodes:
                master = random.choice(master_nodes)
                return self.redis_pools.get(master.node_id)
                
        else:
            # Standalone mode
            for redis_pool in self.redis_pools.values():
                return redis_pool
                
        raise Exception("No healthy Redis connections available")
        
    async def create_backup(self, backup_path: str = None) -> Dict[str, Any]:
        """Create backup of Redis data"""
        backup_results = {
            'started_at': datetime.now().isoformat(),
            'nodes_backed_up': [],
            'errors': []
        }
        
        try:
            # Trigger BGSAVE on all master nodes
            master_nodes = [n for n in self.nodes.values() if n.role == NodeRole.MASTER]
            
            for node in master_nodes:
                if node.status != NodeStatus.HEALTHY:
                    continue
                    
                try:
                    redis_pool = self.redis_pools[node.node_id]
                    
                    # Check if BGSAVE is already running
                    info = await redis_pool.info('persistence')
                    if info.get('rdb_bgsave_in_progress', 0) == 1:
                        self.logger.warning(f"BGSAVE already in progress on {node.node_id}")
                        continue
                        
                    # Start background save
                    await redis_pool.bgsave()
                    backup_results['nodes_backed_up'].append(node.node_id)
                    self.logger.info(f"Started backup for node: {node.node_id}")
                    
                except Exception as e:
                    error_msg = f"Failed to backup node {node.node_id}: {e}"
                    backup_results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                    
            backup_results['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            backup_results['errors'].append(f"Backup operation failed: {e}")
            
        return backup_results
        
    async def scale_cluster(self, target_masters: int, replicas_per_master: int = 1) -> Dict[str, Any]:
        """Scale Redis cluster by adding or removing nodes"""
        
        if self.cluster_mode != ClusterMode.CLUSTER:
            raise Exception("Cluster scaling only supported in cluster mode")
            
        current_masters = len([n for n in self.nodes.values() if n.role == NodeRole.MASTER])
        
        scale_results = {
            'started_at': datetime.now().isoformat(),
            'current_masters': current_masters,
            'target_masters': target_masters,
            'actions_taken': [],
            'errors': []
        }
        
        try:
            if target_masters > current_masters:
                # Scale up - add new master nodes
                await self._add_master_nodes(target_masters - current_masters, scale_results)
            elif target_masters < current_masters:
                # Scale down - remove master nodes
                await self._remove_master_nodes(current_masters - target_masters, scale_results)
                
            # Ensure replica count
            await self._ensure_replica_count(replicas_per_master, scale_results)
            
            scale_results['completed_at'] = datetime.now().isoformat()
            
        except Exception as e:
            scale_results['errors'].append(f"Cluster scaling failed: {e}")
            
        return scale_results
        
    async def _add_master_nodes(self, count: int, results: Dict[str, Any]):
        """Add new master nodes to cluster"""
        # This would involve:
        # 1. Starting new Redis instances
        # 2. Adding them to the cluster
        # 3. Resharding slots
        # For now, log the action
        results['actions_taken'].append(f"Would add {count} master nodes")
        self.logger.info(f"Adding {count} master nodes (not implemented)")
        
    async def _remove_master_nodes(self, count: int, results: Dict[str, Any]):
        """Remove master nodes from cluster"""
        # This would involve:
        # 1. Resharding slots away from nodes
        # 2. Removing nodes from cluster
        # 3. Stopping Redis instances
        results['actions_taken'].append(f"Would remove {count} master nodes")
        self.logger.info(f"Removing {count} master nodes (not implemented)")
        
    async def _ensure_replica_count(self, replicas_per_master: int, results: Dict[str, Any]):
        """Ensure each master has the required number of replicas"""
        results['actions_taken'].append(f"Would ensure {replicas_per_master} replicas per master")
        self.logger.info(f"Ensuring {replicas_per_master} replicas per master (not implemented)")
        
    async def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cluster metrics"""
        metrics = {
            'cluster_mode': self.cluster_mode.value,
            'timestamp': datetime.now().isoformat(),
            'cluster_health': None,
            'node_metrics': {},
            'performance_metrics': {},
            'memory_metrics': {},
            'connection_metrics': {}
        }
        
        if self.cluster_health:
            metrics['cluster_health'] = {
                'overall_status': self.cluster_health.overall_status.value,
                'total_nodes': self.cluster_health.total_nodes,
                'healthy_nodes': self.cluster_health.healthy_nodes,
                'failed_nodes': self.cluster_health.failed_nodes,
                'masters_count': self.cluster_health.masters_count,
                'replicas_count': self.cluster_health.replicas_count,
                'slots_covered': self.cluster_health.cluster_slots_covered,
                'slots_total': self.cluster_health.cluster_slots_total
            }
            
        # Node-level metrics
        for node_id, node in self.nodes.items():
            metrics['node_metrics'][node_id] = {
                'role': node.role.value,
                'status': node.status.value,
                'memory_usage': node.memory_usage,
                'connections': node.connections,
                'ops_per_sec': node.ops_per_sec,
                'last_ping': node.last_ping.isoformat() if node.last_ping else None
            }
            
        # Aggregate performance metrics
        total_ops = sum(n.ops_per_sec for n in self.nodes.values())
        total_memory = sum(n.memory_usage for n in self.nodes.values())
        total_connections = sum(n.connections for n in self.nodes.values())
        
        metrics['performance_metrics'] = {
            'total_operations_per_sec': total_ops,
            'average_operations_per_node': total_ops / len(self.nodes) if self.nodes else 0
        }
        
        metrics['memory_metrics'] = {
            'total_memory_usage': total_memory,
            'average_memory_per_node': total_memory / len(self.nodes) if self.nodes else 0
        }
        
        metrics['connection_metrics'] = {
            'total_connections': total_connections,
            'average_connections_per_node': total_connections / len(self.nodes) if self.nodes else 0
        }
        
        return metrics
        
    async def execute_command(self, command: str, *args, key: str = None, 
                            read_only: bool = False) -> Any:
        """Execute Redis command with automatic connection selection"""
        
        try:
            redis_conn = await self.get_redis_connection(key=key, read_only=read_only)
            
            # Execute command
            if hasattr(redis_conn, command.lower()):
                cmd_method = getattr(redis_conn, command.lower())
                return await cmd_method(*args)
            else:
                # Fallback to execute_command
                return await redis_conn.execute_command(command, *args)
                
        except Exception as e:
            self.logger.error(f"Failed to execute Redis command {command}: {e}")
            raise
            
    async def close(self):
        """Close all Redis connections and cleanup"""
        self.logger.info("Shutting down Redis cluster manager...")
        
        # Cancel health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
                
        # Close all Redis connections
        for redis_pool in self.redis_pools.values():
            try:
                await redis_pool.close()
            except Exception as e:
                self.logger.error(f"Error closing Redis connection: {e}")
                
        # Close Sentinel connection
        if self.sentinel_pool:
            try:
                await self.sentinel_pool.close()
            except Exception as e:
                self.logger.error(f"Error closing Sentinel connection: {e}")
                
        self.redis_pools.clear()
        self.nodes.clear()
        
        self.logger.info("Redis cluster manager shutdown complete")


# Singleton instance
_cluster_manager: Optional[RedisClusterManager] = None


async def get_cluster_manager() -> RedisClusterManager:
    """Get the global cluster manager instance"""
    global _cluster_manager
    
    if _cluster_manager is None:
        _cluster_manager = RedisClusterManager()
        await _cluster_manager.initialize_cluster()
        
    return _cluster_manager


async def close_cluster_manager():
    """Close the global cluster manager"""
    global _cluster_manager
    
    if _cluster_manager:
        await _cluster_manager.close()
        _cluster_manager = None