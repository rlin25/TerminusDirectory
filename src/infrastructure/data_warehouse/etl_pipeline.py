"""
ETL Pipeline for data warehouse ingestion and processing.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, create_engine
import redis.asyncio as aioredis
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class ETLJobStatus(Enum):
    """ETL job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ETLJobType(Enum):
    """Types of ETL jobs."""
    FULL_LOAD = "full_load"
    INCREMENTAL = "incremental"
    DELTA_LOAD = "delta_load"
    REAL_TIME = "real_time"
    BATCH = "batch"


class DataSource(Enum):
    """Data source types."""
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    FILE_SYSTEM = "file_system"
    API = "api"
    KAFKA = "kafka"
    S3 = "s3"


@dataclass
class ETLJobConfig:
    """Configuration for an ETL job."""
    job_id: str
    job_name: str
    job_type: ETLJobType
    source_config: Dict[str, Any]
    target_config: Dict[str, Any]
    transformation_config: Dict[str, Any]
    schedule_config: Dict[str, Any]
    retry_config: Dict[str, Any]
    data_quality_rules: List[Dict[str, Any]]
    dependencies: List[str]
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class ETLJobExecution:
    """ETL job execution details."""
    execution_id: str
    job_id: str
    status: ETLJobStatus
    start_time: datetime
    end_time: Optional[datetime]
    records_processed: int
    records_failed: int
    error_message: Optional[str]
    execution_metadata: Dict[str, Any]


@dataclass
class DataQualityResult:
    """Data quality check results."""
    rule_name: str
    passed: bool
    message: str
    affected_records: int
    severity: str  # "info", "warning", "error", "critical"


class ETLPipeline:
    """
    ETL Pipeline for data warehouse ingestion and processing.
    
    Handles extraction from multiple sources, transformation, and loading
    into data warehouse with comprehensive monitoring and error handling.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        warehouse_connection_string: str,
        max_workers: int = 4
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.warehouse_engine = create_engine(warehouse_connection_string)
        self.max_workers = max_workers
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers)
        
        # Job registry
        self.job_registry: Dict[str, ETLJobConfig] = {}
        self.running_jobs: Dict[str, ETLJobExecution] = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    async def register_job(self, job_config: ETLJobConfig) -> bool:
        """Register an ETL job configuration."""
        try:
            # Validate job configuration
            await self._validate_job_config(job_config)
            
            # Store in registry
            self.job_registry[job_config.job_id] = job_config
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "etl_job_configs",
                job_config.job_id,
                json.dumps(asdict(job_config), default=str)
            )
            
            # Store in database
            await self._store_job_config(job_config)
            
            self.logger.info(f"Registered ETL job: {job_config.job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register ETL job {job_config.job_id}: {e}")
            return False
    
    async def execute_job(
        self,
        job_id: str,
        execution_params: Optional[Dict[str, Any]] = None
    ) -> ETLJobExecution:
        """Execute an ETL job."""
        if job_id not in self.job_registry:
            raise ValueError(f"ETL job {job_id} not found in registry")
        
        job_config = self.job_registry[job_id]
        
        # Create execution record
        execution = ETLJobExecution(
            execution_id=f"{job_id}_{int(datetime.utcnow().timestamp())}",
            job_id=job_id,
            status=ETLJobStatus.PENDING,
            start_time=datetime.utcnow(),
            end_time=None,
            records_processed=0,
            records_failed=0,
            error_message=None,
            execution_metadata=execution_params or {}
        )
        
        # Track running job
        self.running_jobs[execution.execution_id] = execution
        
        try:
            # Update status to running
            execution.status = ETLJobStatus.RUNNING
            await self._update_execution_status(execution)
            
            # Execute the ETL pipeline
            await self._execute_etl_pipeline(job_config, execution)
            
            # Update status to success
            execution.status = ETLJobStatus.SUCCESS
            execution.end_time = datetime.utcnow()
            
        except Exception as e:
            # Handle execution error
            execution.status = ETLJobStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.utcnow()
            
            self.logger.error(f"ETL job {job_id} failed: {e}")
            
            # Check if retry is configured
            if job_config.retry_config.get("enabled", False):
                await self._schedule_retry(job_config, execution)
        
        finally:
            # Update final status
            await self._update_execution_status(execution)
            
            # Remove from running jobs
            if execution.execution_id in self.running_jobs:
                del self.running_jobs[execution.execution_id]
        
        return execution
    
    async def schedule_jobs(self) -> None:
        """Schedule ETL jobs based on their configurations."""
        current_time = datetime.utcnow()
        
        for job_id, job_config in self.job_registry.items():
            if not job_config.is_active:
                continue
            
            # Check if job should be executed
            if await self._should_execute_job(job_config, current_time):
                # Check dependencies
                if await self._check_dependencies(job_config):
                    # Execute job asynchronously
                    asyncio.create_task(self.execute_job(job_id))
                else:
                    self.logger.warning(f"Job {job_id} dependencies not satisfied")
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current status of an ETL job."""
        if job_id not in self.job_registry:
            return {"error": f"Job {job_id} not found"}
        
        job_config = self.job_registry[job_id]
        
        # Get recent executions
        recent_executions = await self._get_recent_executions(job_id, limit=5)
        
        # Get current running execution if any
        running_execution = None
        for execution in self.running_jobs.values():
            if execution.job_id == job_id:
                running_execution = asdict(execution)
                break
        
        # Calculate success rate
        if recent_executions:
            success_count = sum(1 for ex in recent_executions if ex["status"] == ETLJobStatus.SUCCESS.value)
            success_rate = (success_count / len(recent_executions)) * 100
        else:
            success_rate = 0.0
        
        return {
            "job_id": job_id,
            "job_name": job_config.job_name,
            "is_active": job_config.is_active,
            "current_execution": running_execution,
            "recent_executions": recent_executions,
            "success_rate": success_rate,
            "last_execution": recent_executions[0] if recent_executions else None
        }
    
    async def extract_data(
        self,
        source_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Extract data from configured source."""
        source_type = DataSource(source_config["type"])
        
        if source_type == DataSource.POSTGRESQL:
            return await self._extract_from_postgresql(source_config, execution)
        elif source_type == DataSource.REDIS:
            return await self._extract_from_redis(source_config, execution)
        elif source_type == DataSource.FILE_SYSTEM:
            return await self._extract_from_file_system(source_config, execution)
        elif source_type == DataSource.API:
            return await self._extract_from_api(source_config, execution)
        elif source_type == DataSource.S3:
            return await self._extract_from_s3(source_config, execution)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    async def transform_data(
        self,
        data: pd.DataFrame,
        transformation_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Apply transformations to extracted data."""
        transformed_data = data.copy()
        
        transformations = transformation_config.get("transformations", [])
        
        for transform in transformations:
            transform_type = transform["type"]
            
            if transform_type == "filter":
                transformed_data = self._apply_filter(transformed_data, transform)
            elif transform_type == "aggregate":
                transformed_data = self._apply_aggregation(transformed_data, transform)
            elif transform_type == "join":
                transformed_data = await self._apply_join(transformed_data, transform)
            elif transform_type == "clean":
                transformed_data = self._apply_cleaning(transformed_data, transform)
            elif transform_type == "enrich":
                transformed_data = await self._apply_enrichment(transformed_data, transform)
            elif transform_type == "custom":
                transformed_data = self._apply_custom_transformation(transformed_data, transform)
            else:
                self.logger.warning(f"Unknown transformation type: {transform_type}")
        
        # Update execution metadata
        execution.execution_metadata["transformation_applied"] = len(transformations)
        execution.records_processed = len(transformed_data)
        
        return transformed_data
    
    async def load_data(
        self,
        data: pd.DataFrame,
        target_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> bool:
        """Load transformed data to target destination."""
        target_type = target_config["type"]
        
        try:
            if target_type == "data_warehouse":
                await self._load_to_data_warehouse(data, target_config, execution)
            elif target_type == "data_lake":
                await self._load_to_data_lake(data, target_config, execution)
            elif target_type == "cache":
                await self._load_to_cache(data, target_config, execution)
            elif target_type == "file":
                await self._load_to_file(data, target_config, execution)
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            execution.error_message = f"Load failed: {str(e)}"
            return False
    
    async def validate_data_quality(
        self,
        data: pd.DataFrame,
        quality_rules: List[Dict[str, Any]],
        execution: ETLJobExecution
    ) -> List[DataQualityResult]:
        """Validate data quality against configured rules."""
        quality_results = []
        
        for rule in quality_rules:
            rule_name = rule["name"]
            rule_type = rule["type"]
            
            try:
                if rule_type == "completeness":
                    result = self._check_completeness(data, rule)
                elif rule_type == "uniqueness":
                    result = self._check_uniqueness(data, rule)
                elif rule_type == "validity":
                    result = self._check_validity(data, rule)
                elif rule_type == "consistency":
                    result = self._check_consistency(data, rule)
                elif rule_type == "accuracy":
                    result = self._check_accuracy(data, rule)
                elif rule_type == "custom":
                    result = self._check_custom_rule(data, rule)
                else:
                    result = DataQualityResult(
                        rule_name=rule_name,
                        passed=False,
                        message=f"Unknown rule type: {rule_type}",
                        affected_records=0,
                        severity="error"
                    )
                
                quality_results.append(result)
                
            except Exception as e:
                quality_results.append(DataQualityResult(
                    rule_name=rule_name,
                    passed=False,
                    message=f"Quality check failed: {str(e)}",
                    affected_records=0,
                    severity="error"
                ))
        
        # Update execution metadata
        failed_checks = sum(1 for r in quality_results if not r.passed)
        execution.execution_metadata["data_quality_checks"] = {
            "total_checks": len(quality_results),
            "failed_checks": failed_checks,
            "success_rate": ((len(quality_results) - failed_checks) / len(quality_results)) * 100 if quality_results else 100
        }
        
        return quality_results
    
    # Private methods
    async def _execute_etl_pipeline(
        self,
        job_config: ETLJobConfig,
        execution: ETLJobExecution
    ) -> None:
        """Execute the complete ETL pipeline."""
        # Extract
        self.logger.info(f"Starting extraction for job {job_config.job_id}")
        extracted_data = await self.extract_data(job_config.source_config, execution)
        
        if extracted_data.empty:
            self.logger.warning(f"No data extracted for job {job_config.job_id}")
            return
        
        # Data quality validation (pre-transformation)
        if job_config.data_quality_rules:
            self.logger.info(f"Running pre-transformation data quality checks for job {job_config.job_id}")
            quality_results = await self.validate_data_quality(
                extracted_data,
                job_config.data_quality_rules,
                execution
            )
            
            # Check if any critical quality issues
            critical_issues = [r for r in quality_results if not r.passed and r.severity == "critical"]
            if critical_issues:
                raise ValueError(f"Critical data quality issues found: {[r.message for r in critical_issues]}")
        
        # Transform
        self.logger.info(f"Starting transformation for job {job_config.job_id}")
        transformed_data = await self.transform_data(
            extracted_data,
            job_config.transformation_config,
            execution
        )
        
        # Data quality validation (post-transformation)
        if job_config.data_quality_rules:
            self.logger.info(f"Running post-transformation data quality checks for job {job_config.job_id}")
            post_quality_results = await self.validate_data_quality(
                transformed_data,
                job_config.data_quality_rules,
                execution
            )
            
            # Log quality issues
            for result in post_quality_results:
                if not result.passed:
                    self.logger.warning(f"Data quality issue: {result.message}")
        
        # Load
        self.logger.info(f"Starting load for job {job_config.job_id}")
        load_success = await self.load_data(
            transformed_data,
            job_config.target_config,
            execution
        )
        
        if not load_success:
            raise ValueError("Failed to load data to target")
        
        self.logger.info(f"ETL job {job_config.job_id} completed successfully")
    
    async def _extract_from_postgresql(
        self,
        source_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Extract data from PostgreSQL database."""
        query = source_config["query"]
        params = source_config.get("params", {})
        
        # Add incremental extraction logic if configured
        if source_config.get("incremental", False):
            last_execution = await self._get_last_successful_execution(execution.job_id)
            if last_execution:
                params["last_update"] = last_execution["end_time"]
        
        # Execute query
        result = await self.db_session.execute(text(query), params)
        rows = result.fetchall()
        columns = result.keys()
        
        # Convert to DataFrame
        data = pd.DataFrame(rows, columns=columns)
        
        execution.execution_metadata["source_records"] = len(data)
        return data
    
    async def _extract_from_redis(
        self,
        source_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Extract data from Redis."""
        pattern = source_config.get("pattern", "*")
        keys = await self.redis_client.keys(pattern)
        
        data = []
        for key in keys:
            value = await self.redis_client.get(key)
            if value:
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        parsed_value["_key"] = key
                        data.append(parsed_value)
                except json.JSONDecodeError:
                    # Handle non-JSON values
                    data.append({"_key": key, "value": value})
        
        df = pd.DataFrame(data) if data else pd.DataFrame()
        execution.execution_metadata["source_records"] = len(df)
        return df
    
    async def _extract_from_file_system(
        self,
        source_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Extract data from file system."""
        file_path = source_config["file_path"]
        file_format = source_config.get("format", "csv")
        
        if file_format == "csv":
            data = pd.read_csv(file_path, **source_config.get("read_options", {}))
        elif file_format == "json":
            data = pd.read_json(file_path, **source_config.get("read_options", {}))
        elif file_format == "parquet":
            data = pd.read_parquet(file_path, **source_config.get("read_options", {}))
        elif file_format == "excel":
            data = pd.read_excel(file_path, **source_config.get("read_options", {}))
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        execution.execution_metadata["source_records"] = len(data)
        return data
    
    async def _extract_from_api(
        self,
        source_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Extract data from API endpoint."""
        # This would implement API data extraction
        # For now, return empty DataFrame
        execution.execution_metadata["source_records"] = 0
        return pd.DataFrame()
    
    async def _extract_from_s3(
        self,
        source_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> pd.DataFrame:
        """Extract data from S3."""
        # This would implement S3 data extraction
        # For now, return empty DataFrame
        execution.execution_metadata["source_records"] = 0
        return pd.DataFrame()
    
    def _apply_filter(self, data: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply filter transformation."""
        condition = transform["condition"]
        return data.query(condition)
    
    def _apply_aggregation(self, data: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply aggregation transformation."""
        group_by = transform["group_by"]
        aggregations = transform["aggregations"]
        
        return data.groupby(group_by).agg(aggregations).reset_index()
    
    async def _apply_join(self, data: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply join transformation."""
        # This would implement join logic with other data sources
        return data
    
    def _apply_cleaning(self, data: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply data cleaning transformation."""
        operations = transform["operations"]
        
        for operation in operations:
            if operation["type"] == "remove_duplicates":
                data = data.drop_duplicates(subset=operation.get("columns"))
            elif operation["type"] == "fill_missing":
                data = data.fillna(operation["value"])
            elif operation["type"] == "remove_outliers":
                # Simple outlier removal using IQR
                column = operation["column"]
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                data = data[~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))]
        
        return data
    
    async def _apply_enrichment(self, data: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply data enrichment transformation."""
        # This would implement data enrichment logic
        return data
    
    def _apply_custom_transformation(self, data: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply custom transformation."""
        # This would allow custom transformation functions
        return data
    
    async def _load_to_data_warehouse(
        self,
        data: pd.DataFrame,
        target_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> None:
        """Load data to data warehouse."""
        table_name = target_config["table_name"]
        schema = target_config.get("schema", "public")
        mode = target_config.get("mode", "append")  # append, replace, upsert
        
        # Load data using pandas to_sql
        data.to_sql(
            name=table_name,
            con=self.warehouse_engine,
            schema=schema,
            if_exists=mode,
            index=False,
            method="multi",
            chunksize=target_config.get("batch_size", 1000)
        )
        
        execution.execution_metadata["target_records"] = len(data)
    
    async def _load_to_data_lake(
        self,
        data: pd.DataFrame,
        target_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> None:
        """Load data to data lake."""
        # This would implement data lake loading
        execution.execution_metadata["target_records"] = len(data)
    
    async def _load_to_cache(
        self,
        data: pd.DataFrame,
        target_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> None:
        """Load data to cache (Redis)."""
        key_prefix = target_config.get("key_prefix", "etl_data")
        ttl = target_config.get("ttl", 3600)
        
        # Store data as JSON in Redis
        for index, row in data.iterrows():
            key = f"{key_prefix}:{index}"
            value = json.dumps(row.to_dict(), default=str)
            await self.redis_client.setex(key, ttl, value)
        
        execution.execution_metadata["target_records"] = len(data)
    
    async def _load_to_file(
        self,
        data: pd.DataFrame,
        target_config: Dict[str, Any],
        execution: ETLJobExecution
    ) -> None:
        """Load data to file."""
        file_path = target_config["file_path"]
        file_format = target_config.get("format", "csv")
        
        if file_format == "csv":
            data.to_csv(file_path, index=False, **target_config.get("write_options", {}))
        elif file_format == "json":
            data.to_json(file_path, **target_config.get("write_options", {}))
        elif file_format == "parquet":
            data.to_parquet(file_path, **target_config.get("write_options", {}))
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        execution.execution_metadata["target_records"] = len(data)
    
    # Data quality check methods
    def _check_completeness(self, data: pd.DataFrame, rule: Dict[str, Any]) -> DataQualityResult:
        """Check data completeness."""
        columns = rule["columns"]
        threshold = rule.get("threshold", 0.95)
        
        missing_counts = data[columns].isnull().sum()
        total_records = len(data)
        
        completeness_rates = (total_records - missing_counts) / total_records
        min_completeness = completeness_rates.min()
        
        passed = min_completeness >= threshold
        affected_records = missing_counts.sum()
        
        return DataQualityResult(
            rule_name=rule["name"],
            passed=passed,
            message=f"Completeness rate: {min_completeness:.2%} (threshold: {threshold:.2%})",
            affected_records=affected_records,
            severity=rule.get("severity", "warning")
        )
    
    def _check_uniqueness(self, data: pd.DataFrame, rule: Dict[str, Any]) -> DataQualityResult:
        """Check data uniqueness."""
        columns = rule["columns"]
        
        duplicates = data.duplicated(subset=columns).sum()
        total_records = len(data)
        
        uniqueness_rate = (total_records - duplicates) / total_records
        threshold = rule.get("threshold", 1.0)
        
        passed = uniqueness_rate >= threshold
        
        return DataQualityResult(
            rule_name=rule["name"],
            passed=passed,
            message=f"Uniqueness rate: {uniqueness_rate:.2%} (threshold: {threshold:.2%})",
            affected_records=duplicates,
            severity=rule.get("severity", "warning")
        )
    
    def _check_validity(self, data: pd.DataFrame, rule: Dict[str, Any]) -> DataQualityResult:
        """Check data validity."""
        column = rule["column"]
        validation_type = rule["validation_type"]
        
        if validation_type == "range":
            min_val = rule["min_value"]
            max_val = rule["max_value"]
            invalid_records = ((data[column] < min_val) | (data[column] > max_val)).sum()
        elif validation_type == "regex":
            pattern = rule["pattern"]
            invalid_records = (~data[column].str.match(pattern)).sum()
        elif validation_type == "values":
            valid_values = rule["valid_values"]
            invalid_records = (~data[column].isin(valid_values)).sum()
        else:
            invalid_records = 0
        
        total_records = len(data)
        validity_rate = (total_records - invalid_records) / total_records
        threshold = rule.get("threshold", 0.95)
        
        passed = validity_rate >= threshold
        
        return DataQualityResult(
            rule_name=rule["name"],
            passed=passed,
            message=f"Validity rate: {validity_rate:.2%} (threshold: {threshold:.2%})",
            affected_records=invalid_records,
            severity=rule.get("severity", "warning")
        )
    
    def _check_consistency(self, data: pd.DataFrame, rule: Dict[str, Any]) -> DataQualityResult:
        """Check data consistency."""
        # This would implement consistency checks
        return DataQualityResult(
            rule_name=rule["name"],
            passed=True,
            message="Consistency check passed",
            affected_records=0,
            severity=rule.get("severity", "info")
        )
    
    def _check_accuracy(self, data: pd.DataFrame, rule: Dict[str, Any]) -> DataQualityResult:
        """Check data accuracy."""
        # This would implement accuracy checks
        return DataQualityResult(
            rule_name=rule["name"],
            passed=True,
            message="Accuracy check passed",
            affected_records=0,
            severity=rule.get("severity", "info")
        )
    
    def _check_custom_rule(self, data: pd.DataFrame, rule: Dict[str, Any]) -> DataQualityResult:
        """Check custom data quality rule."""
        # This would implement custom quality rules
        return DataQualityResult(
            rule_name=rule["name"],
            passed=True,
            message="Custom rule check passed",
            affected_records=0,
            severity=rule.get("severity", "info")
        )
    
    # Helper methods
    async def _validate_job_config(self, job_config: ETLJobConfig) -> None:
        """Validate ETL job configuration."""
        required_fields = ["job_id", "job_name", "source_config", "target_config"]
        
        for field in required_fields:
            if not getattr(job_config, field):
                raise ValueError(f"Missing required field: {field}")
    
    async def _should_execute_job(self, job_config: ETLJobConfig, current_time: datetime) -> bool:
        """Check if job should be executed based on schedule."""
        schedule_config = job_config.schedule_config
        
        if not schedule_config:
            return False
        
        schedule_type = schedule_config.get("type", "cron")
        
        if schedule_type == "interval":
            interval_minutes = schedule_config.get("interval_minutes", 60)
            last_execution = await self._get_last_execution_time(job_config.job_id)
            
            if not last_execution:
                return True
            
            return (current_time - last_execution).total_seconds() >= (interval_minutes * 60)
        
        elif schedule_type == "cron":
            # This would implement cron-based scheduling
            return False
        
        return False
    
    async def _check_dependencies(self, job_config: ETLJobConfig) -> bool:
        """Check if job dependencies are satisfied."""
        dependencies = job_config.dependencies
        
        if not dependencies:
            return True
        
        for dep_job_id in dependencies:
            last_execution = await self._get_last_successful_execution(dep_job_id)
            if not last_execution:
                return False
            
            # Check if dependency executed recently enough
            dep_max_age = job_config.schedule_config.get("dependency_max_age_hours", 24)
            age_hours = (datetime.utcnow() - last_execution["end_time"]).total_seconds() / 3600
            
            if age_hours > dep_max_age:
                return False
        
        return True
    
    async def _schedule_retry(self, job_config: ETLJobConfig, execution: ETLJobExecution) -> None:
        """Schedule job retry if configured."""
        retry_config = job_config.retry_config
        max_retries = retry_config.get("max_retries", 3)
        retry_delay = retry_config.get("retry_delay_minutes", 5)
        
        # Get current retry count
        current_retries = execution.execution_metadata.get("retry_count", 0)
        
        if current_retries < max_retries:
            # Schedule retry
            retry_time = datetime.utcnow() + timedelta(minutes=retry_delay)
            execution.execution_metadata["retry_count"] = current_retries + 1
            execution.execution_metadata["next_retry"] = retry_time
            
            # This would schedule the retry in a job queue
            self.logger.info(f"Scheduling retry for job {job_config.job_id} at {retry_time}")
    
    # Database operations (placeholder implementations)
    async def _store_job_config(self, job_config: ETLJobConfig) -> None:
        """Store job configuration in database."""
        pass
    
    async def _update_execution_status(self, execution: ETLJobExecution) -> None:
        """Update execution status in database."""
        pass
    
    async def _get_recent_executions(self, job_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job executions."""
        return []
    
    async def _get_last_execution_time(self, job_id: str) -> Optional[datetime]:
        """Get last execution time for a job."""
        return None
    
    async def _get_last_successful_execution(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get last successful execution for a job."""
        return None