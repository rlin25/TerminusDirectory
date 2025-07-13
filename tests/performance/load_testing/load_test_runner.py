"""
Comprehensive load testing framework for the rental ML system.

This module provides tools for running realistic load tests against API endpoints,
ML inference systems, and database operations with configurable patterns and metrics.
"""

import asyncio
import aiohttp
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import json

from ..utils.performance_config import (
    PerformanceConfig, LoadTestConfig, PerformanceMetrics, 
    PerformanceTimer, performance_monitor, LoadPattern,
    assert_performance_threshold, assert_throughput_threshold
)

logger = logging.getLogger(__name__)


@dataclass
class LoadTestUser:
    """Represents a virtual user in load testing."""
    user_id: int
    session_id: str
    think_time_min: float = 1.0
    think_time_max: float = 5.0
    error_count: int = 0
    success_count: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def total_requests(self) -> int:
        return self.success_count + self.error_count
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests
    
    def think(self) -> float:
        """Simulate user think time."""
        think_time = random.uniform(self.think_time_min, self.think_time_max)
        return think_time


@dataclass
class LoadTestRequest:
    """Configuration for a load test request."""
    method: str
    endpoint: str
    headers: Dict[str, str] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None
    weight: float = 1.0  # Relative frequency of this request
    timeout: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'endpoint': self.endpoint,
            'headers': self.headers,
            'payload': self.payload,
            'weight': self.weight,
            'timeout': self.timeout
        }


@dataclass
class LoadTestScenario:
    """A complete load test scenario configuration."""
    name: str
    description: str
    requests: List[LoadTestRequest]
    base_url: str = "http://localhost:8000"
    duration_seconds: int = 300
    concurrent_users: int = 50
    ramp_up_time: int = 60
    think_time_min: float = 1.0
    think_time_max: float = 5.0
    pattern: LoadPattern = LoadPattern.CONSTANT
    
    # Performance expectations
    expected_avg_response_time: Optional[float] = None
    expected_p95_response_time: Optional[float] = None
    expected_throughput: Optional[float] = None
    max_error_rate: float = 0.05  # 5% maximum error rate
    
    def get_weighted_request(self) -> LoadTestRequest:
        """Get a random request based on weights."""
        if not self.requests:
            raise ValueError("No requests defined in scenario")
        
        weights = [req.weight for req in self.requests]
        return random.choices(self.requests, weights=weights)[0]


class LoadTestMetrics:
    """Collector for load test metrics."""
    
    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.users_started: int = 0
        self.users_completed: int = 0
        
    def add_request(self, user_id: int, request: LoadTestRequest, 
                   response_time: float, success: bool, 
                   status_code: Optional[int] = None, error: Optional[str] = None):
        """Add a request result to metrics."""
        self.requests.append({
            'timestamp': time.time(),
            'user_id': user_id,
            'endpoint': request.endpoint,
            'method': request.method,
            'response_time': response_time,
            'success': success,
            'status_code': status_code,
            'error': error
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        if not self.requests:
            return {'error': 'No requests recorded'}
        
        # Basic stats
        total_requests = len(self.requests)
        successful_requests = sum(1 for r in self.requests if r['success'])
        failed_requests = total_requests - successful_requests
        
        # Response time statistics
        response_times = [r['response_time'] for r in self.requests]
        
        # Throughput calculation
        test_duration = (self.end_time or time.time()) - (self.start_time or time.time())
        throughput = total_requests / test_duration if test_duration > 0 else 0
        
        # Error analysis
        errors = {}
        for req in self.requests:
            if not req['success'] and req['error']:
                errors[req['error']] = errors.get(req['error'], 0) + 1
        
        # Endpoint performance
        endpoint_stats = {}
        for req in self.requests:
            endpoint = req['endpoint']
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {
                    'count': 0,
                    'success_count': 0,
                    'response_times': []
                }
            
            endpoint_stats[endpoint]['count'] += 1
            if req['success']:
                endpoint_stats[endpoint]['success_count'] += 1
            endpoint_stats[endpoint]['response_times'].append(req['response_time'])
        
        # Calculate endpoint statistics
        for endpoint, stats in endpoint_stats.items():
            rt = stats['response_times']
            if rt:
                stats.update({
                    'avg_response_time': np.mean(rt),
                    'p50_response_time': np.percentile(rt, 50),
                    'p95_response_time': np.percentile(rt, 95),
                    'p99_response_time': np.percentile(rt, 99),
                    'success_rate': stats['success_count'] / stats['count']
                })
                del stats['response_times']  # Remove raw data
        
        return {
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests,
                'test_duration': test_duration,
                'throughput_rps': throughput,
                'users_started': self.users_started,
                'users_completed': self.users_completed
            },
            'response_times': {
                'avg': np.mean(response_times),
                'min': np.min(response_times),
                'max': np.max(response_times),
                'p50': np.percentile(response_times, 50),
                'p90': np.percentile(response_times, 90),
                'p95': np.percentile(response_times, 95),
                'p99': np.percentile(response_times, 99)
            },
            'errors': errors,
            'endpoints': endpoint_stats
        }
    
    def export_to_file(self, filename: str):
        """Export metrics to JSON file."""
        summary = self.get_summary()
        summary['raw_requests'] = self.requests
        summary['export_time'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Load test metrics exported to {filename}")


class LoadTestRunner:
    """Main load test execution engine."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig.from_environment()
        self.metrics = LoadTestMetrics()
        self.active_sessions: Dict[int, aiohttp.ClientSession] = {}
        self.stop_flag = False
        
    async def run_scenario(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Run a complete load test scenario."""
        logger.info(f"Starting load test scenario: {scenario.name}")
        logger.info(f"Duration: {scenario.duration_seconds}s, Users: {scenario.concurrent_users}")
        
        self.metrics = LoadTestMetrics()
        self.metrics.start_time = time.time()
        self.stop_flag = False
        
        try:
            # Start performance monitoring
            await performance_monitor.start_monitoring(interval=5.0)
            
            # Execute the load pattern
            if scenario.pattern == LoadPattern.CONSTANT:
                await self._run_constant_load(scenario)
            elif scenario.pattern == LoadPattern.RAMP_UP:
                await self._run_ramp_up_load(scenario)
            elif scenario.pattern == LoadPattern.SPIKE:
                await self._run_spike_load(scenario)
            elif scenario.pattern == LoadPattern.STRESS:
                await self._run_stress_load(scenario)
            else:
                await self._run_constant_load(scenario)  # Default fallback
            
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            raise
        finally:
            # Cleanup
            await self._cleanup_sessions()
            await performance_monitor.stop_monitoring()
            self.metrics.end_time = time.time()
            
            logger.info(f"Load test completed: {scenario.name}")
        
        # Analyze results
        summary = self.metrics.get_summary()
        await self._validate_performance_expectations(scenario, summary)
        
        return summary
    
    async def _run_constant_load(self, scenario: LoadTestScenario):
        """Run constant load pattern."""
        # Create all users simultaneously
        user_tasks = []
        for user_id in range(scenario.concurrent_users):
            user = LoadTestUser(
                user_id=user_id,
                session_id=f"session_{user_id}_{int(time.time())}",
                think_time_min=scenario.think_time_min,
                think_time_max=scenario.think_time_max
            )
            self.metrics.users_started += 1
            
            task = asyncio.create_task(
                self._run_user_session(user, scenario)
            )
            user_tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(scenario.duration_seconds)
        self.stop_flag = True
        
        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _run_ramp_up_load(self, scenario: LoadTestScenario):
        """Run ramp-up load pattern."""
        user_tasks = []
        users_per_step = max(1, scenario.concurrent_users // 10)  # 10 steps
        step_interval = scenario.ramp_up_time / 10
        
        # Ramp up users gradually
        for step in range(10):
            users_in_step = min(users_per_step, scenario.concurrent_users - len(user_tasks))
            
            for i in range(users_in_step):
                user_id = len(user_tasks)
                user = LoadTestUser(
                    user_id=user_id,
                    session_id=f"session_{user_id}_{int(time.time())}",
                    think_time_min=scenario.think_time_min,
                    think_time_max=scenario.think_time_max
                )
                self.metrics.users_started += 1
                
                task = asyncio.create_task(
                    self._run_user_session(user, scenario)
                )
                user_tasks.append(task)
            
            if step < 9:  # Don't wait after the last step
                await asyncio.sleep(step_interval)
        
        # Run for remaining duration
        remaining_time = scenario.duration_seconds - scenario.ramp_up_time
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
        
        self.stop_flag = True
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _run_spike_load(self, scenario: LoadTestScenario):
        """Run spike load pattern."""
        # Start with 20% of users
        initial_users = max(1, scenario.concurrent_users // 5)
        spike_users = scenario.concurrent_users
        
        user_tasks = []
        
        # Start initial load
        for user_id in range(initial_users):
            user = LoadTestUser(
                user_id=user_id,
                session_id=f"session_{user_id}_{int(time.time())}",
                think_time_min=scenario.think_time_min,
                think_time_max=scenario.think_time_max
            )
            self.metrics.users_started += 1
            
            task = asyncio.create_task(
                self._run_user_session(user, scenario)
            )
            user_tasks.append(task)
        
        # Wait for 30% of duration
        await asyncio.sleep(scenario.duration_seconds * 0.3)
        
        # Add spike users
        for user_id in range(initial_users, spike_users):
            user = LoadTestUser(
                user_id=user_id,
                session_id=f"session_{user_id}_{int(time.time())}",
                think_time_min=scenario.think_time_min * 0.5,  # Faster during spike
                think_time_max=scenario.think_time_max * 0.5
            )
            self.metrics.users_started += 1
            
            task = asyncio.create_task(
                self._run_user_session(user, scenario)
            )
            user_tasks.append(task)
        
        # Wait for remaining duration
        await asyncio.sleep(scenario.duration_seconds * 0.7)
        
        self.stop_flag = True
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _run_stress_load(self, scenario: LoadTestScenario):
        """Run stress test with gradually increasing load."""
        step_duration = 60  # 1 minute per step
        max_steps = scenario.duration_seconds // step_duration
        users_per_step = max(1, scenario.concurrent_users // max_steps)
        
        user_tasks = []
        
        for step in range(max_steps):
            # Add more users each step
            new_users = min(users_per_step, scenario.concurrent_users - len(user_tasks))
            
            for i in range(new_users):
                user_id = len(user_tasks)
                user = LoadTestUser(
                    user_id=user_id,
                    session_id=f"session_{user_id}_{int(time.time())}",
                    think_time_min=scenario.think_time_min * 0.7,  # More aggressive
                    think_time_max=scenario.think_time_max * 0.7
                )
                self.metrics.users_started += 1
                
                task = asyncio.create_task(
                    self._run_user_session(user, scenario)
                )
                user_tasks.append(task)
            
            logger.info(f"Stress test step {step + 1}/{max_steps}: {len(user_tasks)} active users")
            
            # Wait for step duration
            await asyncio.sleep(step_duration)
        
        self.stop_flag = True
        await asyncio.gather(*user_tasks, return_exceptions=True)
    
    async def _run_user_session(self, user: LoadTestUser, scenario: LoadTestScenario):
        """Run a single user's session."""
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=scenario.requests[0].timeout)
        )
        self.active_sessions[user.user_id] = session
        
        try:
            while not self.stop_flag:
                # Select a request to make
                request = scenario.get_weighted_request()
                
                # Execute request with timing
                start_time = time.perf_counter()
                success = False
                status_code = None
                error = None
                
                try:
                    url = f"{scenario.base_url}{request.endpoint}"
                    
                    async with session.request(
                        method=request.method,
                        url=url,
                        headers=request.headers,
                        json=request.payload,
                        timeout=aiohttp.ClientTimeout(total=request.timeout)
                    ) as response:
                        status_code = response.status
                        await response.text()  # Consume response body
                        success = 200 <= status_code < 400
                        
                except asyncio.TimeoutError:
                    error = "timeout"
                except aiohttp.ClientError as e:
                    error = f"client_error: {str(e)}"
                except Exception as e:
                    error = f"unexpected_error: {str(e)}"
                
                response_time = time.perf_counter() - start_time
                
                # Record metrics
                self.metrics.add_request(
                    user_id=user.user_id,
                    request=request,
                    response_time=response_time,
                    success=success,
                    status_code=status_code,
                    error=error
                )
                
                # Update user stats
                if success:
                    user.success_count += 1
                else:
                    user.error_count += 1
                
                # Think time before next request
                if not self.stop_flag:
                    think_time = user.think()
                    await asyncio.sleep(think_time)
                    
        finally:
            await session.close()
            if user.user_id in self.active_sessions:
                del self.active_sessions[user.user_id]
            self.metrics.users_completed += 1
    
    async def _cleanup_sessions(self):
        """Clean up any remaining sessions."""
        for session in self.active_sessions.values():
            if not session.closed:
                await session.close()
        self.active_sessions.clear()
    
    async def _validate_performance_expectations(self, scenario: LoadTestScenario, summary: Dict[str, Any]):
        """Validate that performance meets expectations."""
        response_times = summary.get('response_times', {})
        test_summary = summary.get('summary', {})
        
        # Check success rate
        success_rate = test_summary.get('success_rate', 0)
        if success_rate < (1 - scenario.max_error_rate):
            logger.warning(
                f"Success rate {success_rate:.3f} below threshold "
                f"{1 - scenario.max_error_rate:.3f}"
            )
        
        # Check average response time
        if scenario.expected_avg_response_time:
            avg_response_time = response_times.get('avg', 0)
            if avg_response_time > scenario.expected_avg_response_time:
                logger.warning(
                    f"Average response time {avg_response_time:.3f}s exceeds "
                    f"expected {scenario.expected_avg_response_time:.3f}s"
                )
        
        # Check P95 response time
        if scenario.expected_p95_response_time:
            p95_response_time = response_times.get('p95', 0)
            if p95_response_time > scenario.expected_p95_response_time:
                logger.warning(
                    f"P95 response time {p95_response_time:.3f}s exceeds "
                    f"expected {scenario.expected_p95_response_time:.3f}s"
                )
        
        # Check throughput
        if scenario.expected_throughput:
            actual_throughput = test_summary.get('throughput_rps', 0)
            if actual_throughput < scenario.expected_throughput:
                logger.warning(
                    f"Throughput {actual_throughput:.1f} RPS below "
                    f"expected {scenario.expected_throughput:.1f} RPS"
                )


# Predefined load test scenarios for the rental ML system
def get_search_load_test() -> LoadTestScenario:
    """Get search endpoint load test scenario."""
    return LoadTestScenario(
        name="Search API Load Test",
        description="Load test for property search functionality",
        requests=[
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/search?query=apartment&location=san+francisco&max_price=3000",
                weight=3.0
            ),
            LoadTestRequest(
                method="GET", 
                endpoint="/api/v1/search?query=house&bedrooms=2&bathrooms=2",
                weight=2.0
            ),
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/search?query=studio&location=new+york",
                weight=1.0
            )
        ],
        duration_seconds=300,
        concurrent_users=50,
        expected_avg_response_time=0.5,
        expected_p95_response_time=1.0,
        expected_throughput=100.0
    )


def get_recommendation_load_test() -> LoadTestScenario:
    """Get recommendation endpoint load test scenario."""
    return LoadTestScenario(
        name="Recommendation API Load Test",
        description="Load test for ML-powered recommendations",
        requests=[
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/recommendations/user/1?limit=10",
                weight=4.0
            ),
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/recommendations/user/2?limit=20",
                weight=2.0
            ),
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/recommendations/similar/property123?limit=5",
                weight=1.0
            )
        ],
        duration_seconds=600,
        concurrent_users=30,
        expected_avg_response_time=1.0,
        expected_p95_response_time=2.0,
        expected_throughput=50.0
    )


def get_mixed_workload_test() -> LoadTestScenario:
    """Get mixed workload scenario covering all endpoints."""
    return LoadTestScenario(
        name="Mixed Workload Load Test",
        description="Realistic mixed workload across all endpoints",
        requests=[
            # Health check
            LoadTestRequest(
                method="GET",
                endpoint="/health",
                weight=1.0
            ),
            # Search operations
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/search?query=apartment&location=california",
                weight=4.0
            ),
            # Recommendations
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/recommendations/user/1?limit=10",
                weight=3.0
            ),
            # Property details
            LoadTestRequest(
                method="GET",
                endpoint="/api/v1/properties/property123",
                weight=2.0
            ),
            # User interactions
            LoadTestRequest(
                method="POST",
                endpoint="/api/v1/users/1/interactions",
                payload={
                    "property_id": "property123",
                    "interaction_type": "view",
                    "duration_seconds": 30
                },
                weight=1.0
            )
        ],
        duration_seconds=900,  # 15 minutes
        concurrent_users=75,
        pattern=LoadPattern.RAMP_UP,
        ramp_up_time=120,  # 2 minutes ramp up
        expected_avg_response_time=0.8,
        expected_p95_response_time=2.0,
        max_error_rate=0.02  # 2% max error rate
    )


async def run_predefined_scenarios():
    """Run all predefined load test scenarios."""
    runner = LoadTestRunner()
    scenarios = [
        get_search_load_test(),
        get_recommendation_load_test(),
        get_mixed_workload_test()
    ]
    
    results = {}
    
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario.name}")
        try:
            result = await runner.run_scenario(scenario)
            results[scenario.name] = result
            
            # Export results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"load_test_{scenario.name.lower().replace(' ', '_')}_{timestamp}.json"
            runner.metrics.export_to_file(filename)
            
            logger.info(f"Scenario completed: {scenario.name}")
            
        except Exception as e:
            logger.error(f"Scenario failed: {scenario.name}: {e}")
            results[scenario.name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run load tests
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Run single scenario for testing
        scenario = get_search_load_test()
        scenario.duration_seconds = 60  # Shorter for testing
        scenario.concurrent_users = 5
        
        runner = LoadTestRunner()
        results = asyncio.run(runner.run_scenario(scenario))
        print(json.dumps(results, indent=2))
    else:
        # Run all scenarios
        results = asyncio.run(run_predefined_scenarios())
        print("All load test scenarios completed")
        print(json.dumps(results, indent=2))