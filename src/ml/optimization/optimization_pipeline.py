"""
Comprehensive Model Optimization Pipeline for Production Deployment.

This module provides:
- Automated model optimization workflows
- Multi-target optimization (latency, memory, accuracy)
- Hardware-specific optimizations (CPU, GPU, TPU, Edge)
- Model compression techniques (quantization, pruning, distillation)
- Cross-platform model conversion (ONNX, TensorFlow Lite)
- Performance benchmarking and validation
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
import onnx
import onnxruntime as ort
from tensorflow_model_optimization.python.core.sparsity.keras import prune, strip_pruning
from tensorflow_model_optimization.python.core.quantization.keras import quantize_model
import psutil

from .model_quantizer import ModelQuantizer
from .model_pruner import ModelPruner  
from .model_distiller import ModelDistiller
from .onnx_converter import ONNXConverter
from .tflite_converter import TFLiteConverter


@dataclass
class OptimizationTarget:
    """Target specifications for model optimization"""
    target_platform: str  # 'cpu', 'gpu', 'tpu', 'mobile', 'edge'
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_accuracy: Optional[float] = None
    target_format: str = 'tensorflow'  # 'tensorflow', 'onnx', 'tflite'
    hardware_constraints: Dict[str, Any] = None


@dataclass
class OptimizationConfig:
    """Configuration for optimization pipeline"""
    # Quantization settings
    enable_quantization: bool = True
    quantization_type: str = 'int8'  # 'int8', 'int16', 'float16'
    post_training_quantization: bool = True
    quantization_aware_training: bool = False
    
    # Pruning settings
    enable_pruning: bool = True
    target_sparsity: float = 0.5
    pruning_schedule: str = 'polynomial'  # 'polynomial', 'constant'
    
    # Distillation settings
    enable_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    
    # Conversion settings
    export_onnx: bool = True
    export_tflite: bool = True
    optimize_for_inference: bool = True
    
    # Performance settings
    benchmark_iterations: int = 100
    validate_accuracy: bool = True
    accuracy_threshold: float = 0.95  # Relative to original model


@dataclass
class OptimizationResult:
    """Results from model optimization"""
    original_model_path: str
    optimized_models: Dict[str, str]  # format -> path
    performance_metrics: Dict[str, Any]
    accuracy_metrics: Dict[str, Any]
    optimization_config: OptimizationConfig
    target: OptimizationTarget
    success: bool
    errors: List[str]
    optimization_time: float


class OptimizationPipeline:
    """
    Comprehensive model optimization pipeline for production deployment.
    
    Features:
    - Multi-stage optimization (quantization, pruning, distillation)
    - Hardware-specific optimizations
    - Cross-platform model conversion
    - Performance benchmarking and validation
    - Automated optimization workflows
    - Quality assurance and rollback mechanisms
    """
    
    def __init__(self,
                 workspace_dir: str = "/tmp/model_optimization",
                 enable_gpu: bool = True):
        
        self.workspace_dir = Path(workspace_dir)
        self.enable_gpu = enable_gpu
        
        self.logger = logging.getLogger(__name__)
        
        # Create workspace
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimization components
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner()
        self.distiller = ModelDistiller()
        self.onnx_converter = ONNXConverter()
        self.tflite_converter = TFLiteConverter()
        
        # GPU configuration
        if self.enable_gpu:
            self._configure_gpu()
        
        # Performance tracking
        self.optimization_history = []
    
    def _configure_gpu(self):
        """Configure GPU settings for optimization"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Configured {len(gpus)} GPUs for optimization")
            else:
                self.logger.info("No GPUs available, using CPU")
        except Exception as e:
            self.logger.warning(f"GPU configuration failed: {e}")
    
    async def optimize_model(self,
                           model_path: str,
                           target: OptimizationTarget,
                           config: OptimizationConfig,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> OptimizationResult:
        """
        Optimize model for target deployment platform.
        
        Args:
            model_path: Path to the original model
            target: Target platform and constraints
            config: Optimization configuration
            validation_data: Data for accuracy validation
            
        Returns:
            OptimizationResult with optimized models and metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting optimization for {target.target_platform}")
            
            # Create optimization workspace
            optimization_id = f"opt_{int(time.time())}"
            work_dir = self.workspace_dir / optimization_id
            work_dir.mkdir(exist_ok=True)
            
            # Load original model
            original_model = await self._load_model(model_path)
            
            # Baseline performance
            baseline_metrics = await self._benchmark_model(
                original_model, target, validation_data, work_dir / "baseline"
            )
            
            # Optimization stages
            optimized_models = {}
            current_model = original_model
            current_path = model_path
            errors = []
            
            # Stage 1: Pruning (if enabled)
            if config.enable_pruning:
                try:
                    pruned_model, pruned_path = await self._apply_pruning(
                        current_model, config, work_dir, validation_data
                    )
                    if pruned_model is not None:
                        current_model = pruned_model
                        current_path = pruned_path
                        optimized_models['pruned'] = pruned_path
                        self.logger.info("Pruning completed successfully")
                except Exception as e:
                    errors.append(f"Pruning failed: {e}")
                    self.logger.error(f"Pruning failed: {e}")
            
            # Stage 2: Quantization (if enabled)
            if config.enable_quantization:
                try:
                    quantized_model, quantized_path = await self._apply_quantization(
                        current_model, config, work_dir, validation_data
                    )
                    if quantized_model is not None:
                        current_model = quantized_model
                        current_path = quantized_path
                        optimized_models['quantized'] = quantized_path
                        self.logger.info("Quantization completed successfully")
                except Exception as e:
                    errors.append(f"Quantization failed: {e}")
                    self.logger.error(f"Quantization failed: {e}")
            
            # Stage 3: Distillation (if enabled)
            if config.enable_distillation and config.teacher_model_path:
                try:
                    distilled_model, distilled_path = await self._apply_distillation(
                        current_model, config, work_dir, validation_data
                    )
                    if distilled_model is not None:
                        current_model = distilled_model
                        current_path = distilled_path
                        optimized_models['distilled'] = distilled_path
                        self.logger.info("Distillation completed successfully")
                except Exception as e:
                    errors.append(f"Distillation failed: {e}")
                    self.logger.error(f"Distillation failed: {e}")
            
            # Stage 4: Format conversion
            await self._convert_models(
                current_model, current_path, target, config, work_dir, optimized_models
            )
            
            # Performance evaluation
            final_metrics = await self._evaluate_optimized_models(
                optimized_models, baseline_metrics, target, validation_data, work_dir
            )
            
            # Accuracy validation
            accuracy_metrics = {}
            if config.validate_accuracy and validation_data:
                accuracy_metrics = await self._validate_accuracy(
                    optimized_models, original_model, validation_data, config
                )
            
            # Determine success
            success = len(optimized_models) > 0 and len(errors) == 0
            if config.validate_accuracy:
                success = success and self._meets_accuracy_threshold(
                    accuracy_metrics, config.accuracy_threshold
                )
            
            optimization_time = time.time() - start_time
            
            # Create result
            result = OptimizationResult(
                original_model_path=model_path,
                optimized_models=optimized_models,
                performance_metrics=final_metrics,
                accuracy_metrics=accuracy_metrics,
                optimization_config=config,
                target=target,
                success=success,
                errors=errors,
                optimization_time=optimization_time
            )
            
            # Store optimization history
            self.optimization_history.append(result)
            
            # Save optimization report
            await self._save_optimization_report(result, work_dir)
            
            self.logger.info(f"Optimization completed in {optimization_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization pipeline failed: {e}")
            
            return OptimizationResult(
                original_model_path=model_path,
                optimized_models={},
                performance_metrics={},
                accuracy_metrics={},
                optimization_config=config,
                target=target,
                success=False,
                errors=[str(e)],
                optimization_time=time.time() - start_time
            )
    
    async def batch_optimize(self,
                           model_paths: List[str],
                           targets: List[OptimizationTarget],
                           config: OptimizationConfig,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[OptimizationResult]:
        """Optimize multiple models for multiple targets"""
        try:
            results = []
            
            for model_path in model_paths:
                for target in targets:
                    self.logger.info(f"Optimizing {model_path} for {target.target_platform}")
                    
                    result = await self.optimize_model(
                        model_path, target, config, validation_data
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch optimization failed: {e}")
            return []
    
    async def auto_optimize(self,
                          model_path: str,
                          target_platform: str,
                          validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> OptimizationResult:
        """Automatically optimize model with platform-specific best practices"""
        try:
            # Define platform-specific configurations
            platform_configs = {
                'mobile': OptimizationConfig(
                    enable_quantization=True,
                    quantization_type='int8',
                    enable_pruning=True,
                    target_sparsity=0.7,
                    export_tflite=True,
                    export_onnx=False
                ),
                'edge': OptimizationConfig(
                    enable_quantization=True,
                    quantization_type='int8',
                    enable_pruning=True,
                    target_sparsity=0.8,
                    export_tflite=True,
                    export_onnx=True
                ),
                'cpu': OptimizationConfig(
                    enable_quantization=True,
                    quantization_type='int8',
                    enable_pruning=True,
                    target_sparsity=0.5,
                    export_onnx=True,
                    export_tflite=False
                ),
                'gpu': OptimizationConfig(
                    enable_quantization=True,
                    quantization_type='float16',
                    enable_pruning=False,
                    export_onnx=True,
                    export_tflite=False
                )
            }
            
            # Platform-specific targets
            platform_targets = {
                'mobile': OptimizationTarget(
                    target_platform='mobile',
                    max_latency_ms=100,
                    max_memory_mb=50,
                    target_format='tflite'
                ),
                'edge': OptimizationTarget(
                    target_platform='edge',
                    max_latency_ms=50,
                    max_memory_mb=100,
                    target_format='tflite'
                ),
                'cpu': OptimizationTarget(
                    target_platform='cpu',
                    max_latency_ms=200,
                    target_format='onnx'
                ),
                'gpu': OptimizationTarget(
                    target_platform='gpu',
                    max_latency_ms=50,
                    target_format='tensorflow'
                )
            }
            
            config = platform_configs.get(target_platform, OptimizationConfig())
            target = platform_targets.get(target_platform, OptimizationTarget(target_platform=target_platform))
            
            return await self.optimize_model(model_path, target, config, validation_data)
            
        except Exception as e:
            self.logger.error(f"Auto optimization failed: {e}")
            raise
    
    async def _load_model(self, model_path: str) -> tf.keras.Model:
        """Load TensorFlow model"""
        try:
            if model_path.endswith('.h5'):
                return tf.keras.models.load_model(model_path)
            elif model_path.endswith('.pb') or os.path.isdir(model_path):
                return tf.saved_model.load(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    async def _apply_pruning(self,
                           model: tf.keras.Model,
                           config: OptimizationConfig,
                           work_dir: Path,
                           validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[tf.keras.Model, str]:
        """Apply model pruning"""
        try:
            # Apply pruning
            pruned_model = await self.pruner.prune_model(
                model,
                target_sparsity=config.target_sparsity,
                schedule=config.pruning_schedule
            )
            
            # Fine-tune if validation data available
            if validation_data and pruned_model:
                X_val, y_val = validation_data
                pruned_model = await self.pruner.fine_tune_pruned_model(
                    pruned_model, X_val, y_val, epochs=10
                )
            
            # Save pruned model
            pruned_path = str(work_dir / "pruned_model.h5")
            if pruned_model:
                pruned_model.save(pruned_path)
                return pruned_model, pruned_path
            
            return None, ""
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            raise
    
    async def _apply_quantization(self,
                                model: tf.keras.Model,
                                config: OptimizationConfig,
                                work_dir: Path,
                                validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[tf.keras.Model, str]:
        """Apply model quantization"""
        try:
            if config.post_training_quantization:
                # Post-training quantization
                quantized_model = await self.quantizer.post_training_quantize(
                    model,
                    quantization_type=config.quantization_type,
                    representative_data=validation_data[0] if validation_data else None
                )
            else:
                # Quantization-aware training
                quantized_model = await self.quantizer.quantization_aware_training(
                    model,
                    training_data=validation_data,
                    quantization_type=config.quantization_type
                )
            
            # Save quantized model
            quantized_path = str(work_dir / "quantized_model.h5")
            if quantized_model:
                quantized_model.save(quantized_path)
                return quantized_model, quantized_path
            
            return None, ""
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise
    
    async def _apply_distillation(self,
                                student_model: tf.keras.Model,
                                config: OptimizationConfig,
                                work_dir: Path,
                                validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[tf.keras.Model, str]:
        """Apply knowledge distillation"""
        try:
            if not config.teacher_model_path or not validation_data:
                return None, ""
            
            # Load teacher model
            teacher_model = await self._load_model(config.teacher_model_path)
            
            # Apply distillation
            distilled_model = await self.distiller.distill_model(
                teacher_model=teacher_model,
                student_model=student_model,
                training_data=validation_data,
                temperature=config.distillation_temperature,
                alpha=config.distillation_alpha
            )
            
            # Save distilled model
            distilled_path = str(work_dir / "distilled_model.h5")
            if distilled_model:
                distilled_model.save(distilled_path)
                return distilled_model, distilled_path
            
            return None, ""
            
        except Exception as e:
            self.logger.error(f"Distillation failed: {e}")
            raise
    
    async def _convert_models(self,
                            model: tf.keras.Model,
                            model_path: str,
                            target: OptimizationTarget,
                            config: OptimizationConfig,
                            work_dir: Path,
                            optimized_models: Dict[str, str]):
        """Convert models to different formats"""
        try:
            # ONNX conversion
            if config.export_onnx or target.target_format == 'onnx':
                try:
                    onnx_path = str(work_dir / "model.onnx")
                    success = await self.onnx_converter.convert_to_onnx(model, onnx_path)
                    if success:
                        optimized_models['onnx'] = onnx_path
                        self.logger.info("ONNX conversion completed")
                except Exception as e:
                    self.logger.warning(f"ONNX conversion failed: {e}")
            
            # TensorFlow Lite conversion
            if config.export_tflite or target.target_format == 'tflite':
                try:
                    tflite_path = str(work_dir / "model.tflite")
                    success = await self.tflite_converter.convert_to_tflite(
                        model, tflite_path, optimization=True
                    )
                    if success:
                        optimized_models['tflite'] = tflite_path
                        self.logger.info("TensorFlow Lite conversion completed")
                except Exception as e:
                    self.logger.warning(f"TensorFlow Lite conversion failed: {e}")
            
            # TensorFlow SavedModel (optimized)
            if target.target_format == 'tensorflow' or config.optimize_for_inference:
                try:
                    saved_model_path = str(work_dir / "optimized_savedmodel")
                    tf.saved_model.save(model, saved_model_path)
                    optimized_models['tensorflow'] = saved_model_path
                    self.logger.info("Optimized TensorFlow SavedModel created")
                except Exception as e:
                    self.logger.warning(f"SavedModel optimization failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Model conversion failed: {e}")
    
    async def _benchmark_model(self,
                             model: tf.keras.Model,
                             target: OptimizationTarget,
                             validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                             work_dir: Path) -> Dict[str, Any]:
        """Benchmark model performance"""
        try:
            metrics = {
                'model_size_mb': 0,
                'memory_usage_mb': 0,
                'inference_latency_ms': 0,
                'throughput_fps': 0
            }
            
            # Model size
            if hasattr(model, 'count_params'):
                param_count = model.count_params()
                metrics['model_size_mb'] = (param_count * 4) / (1024 * 1024)  # Assume float32
            
            # Memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)
            
            # Inference latency
            if validation_data:
                X_val = validation_data[0]
                sample_input = X_val[:1] if len(X_val) > 0 else np.random.random((1, 10))
                
                # Warmup
                for _ in range(10):
                    _ = model(sample_input)
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    _ = model(sample_input)
                end_time = time.time()
                
                metrics['inference_latency_ms'] = ((end_time - start_time) / 100) * 1000
                metrics['throughput_fps'] = 1000 / metrics['inference_latency_ms']
            
            # Memory usage after inference
            memory_after = process.memory_info().rss / (1024 * 1024)
            metrics['memory_usage_mb'] = memory_after - memory_before
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            return {}
    
    async def _evaluate_optimized_models(self,
                                       optimized_models: Dict[str, str],
                                       baseline_metrics: Dict[str, Any],
                                       target: OptimizationTarget,
                                       validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                                       work_dir: Path) -> Dict[str, Any]:
        """Evaluate performance of optimized models"""
        try:
            evaluation_results = {
                'baseline': baseline_metrics,
                'optimized': {}
            }
            
            for format_name, model_path in optimized_models.items():
                try:
                    if format_name == 'onnx':
                        metrics = await self._benchmark_onnx_model(model_path, validation_data)
                    elif format_name == 'tflite':
                        metrics = await self._benchmark_tflite_model(model_path, validation_data)
                    else:
                        # TensorFlow model
                        model = await self._load_model(model_path)
                        metrics = await self._benchmark_model(model, target, validation_data, work_dir)
                    
                    evaluation_results['optimized'][format_name] = metrics
                    
                    # Calculate improvements
                    if baseline_metrics:
                        improvements = {}
                        for metric, value in metrics.items():
                            if metric in baseline_metrics and baseline_metrics[metric] > 0:
                                if metric in ['model_size_mb', 'memory_usage_mb', 'inference_latency_ms']:
                                    # Lower is better
                                    improvement = ((baseline_metrics[metric] - value) / baseline_metrics[metric]) * 100
                                else:
                                    # Higher is better
                                    improvement = ((value - baseline_metrics[metric]) / baseline_metrics[metric]) * 100
                                improvements[f"{metric}_improvement_percent"] = improvement
                        
                        evaluation_results['optimized'][format_name]['improvements'] = improvements
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate {format_name} model: {e}")
                    evaluation_results['optimized'][format_name] = {'error': str(e)}
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {}
    
    async def _benchmark_onnx_model(self, model_path: str, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Benchmark ONNX model performance"""
        try:
            metrics = {}
            
            # Model size
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            metrics['model_size_mb'] = model_size
            
            # ONNX runtime inference
            if validation_data:
                session = ort.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                
                X_val = validation_data[0]
                sample_input = X_val[:1] if len(X_val) > 0 else np.random.random((1, 10)).astype(np.float32)
                
                # Warmup
                for _ in range(10):
                    _ = session.run(None, {input_name: sample_input})
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    _ = session.run(None, {input_name: sample_input})
                end_time = time.time()
                
                metrics['inference_latency_ms'] = ((end_time - start_time) / 100) * 1000
                metrics['throughput_fps'] = 1000 / metrics['inference_latency_ms']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"ONNX benchmarking failed: {e}")
            return {}
    
    async def _benchmark_tflite_model(self, model_path: str, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Benchmark TensorFlow Lite model performance"""
        try:
            metrics = {}
            
            # Model size
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            metrics['model_size_mb'] = model_size
            
            # TensorFlow Lite inference
            if validation_data:
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                X_val = validation_data[0]
                sample_input = X_val[:1] if len(X_val) > 0 else np.random.random((1, 10)).astype(np.float32)
                
                # Resize input if needed
                if sample_input.shape != input_details[0]['shape']:
                    sample_input = np.resize(sample_input, input_details[0]['shape'])
                
                # Warmup
                for _ in range(10):
                    interpreter.set_tensor(input_details[0]['index'], sample_input)
                    interpreter.invoke()
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    interpreter.set_tensor(input_details[0]['index'], sample_input)
                    interpreter.invoke()
                end_time = time.time()
                
                metrics['inference_latency_ms'] = ((end_time - start_time) / 100) * 1000
                metrics['throughput_fps'] = 1000 / metrics['inference_latency_ms']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"TensorFlow Lite benchmarking failed: {e}")
            return {}
    
    async def _validate_accuracy(self,
                               optimized_models: Dict[str, str],
                               original_model: tf.keras.Model,
                               validation_data: Tuple[np.ndarray, np.ndarray],
                               config: OptimizationConfig) -> Dict[str, Any]:
        """Validate accuracy of optimized models"""
        try:
            X_val, y_val = validation_data
            
            # Get baseline accuracy
            baseline_predictions = original_model.predict(X_val)
            baseline_accuracy = self._calculate_accuracy(y_val, baseline_predictions)
            
            accuracy_results = {
                'baseline_accuracy': baseline_accuracy,
                'optimized_accuracies': {}
            }
            
            for format_name, model_path in optimized_models.items():
                try:
                    if format_name == 'onnx':
                        predictions = await self._predict_onnx(model_path, X_val)
                    elif format_name == 'tflite':
                        predictions = await self._predict_tflite(model_path, X_val)
                    else:
                        model = await self._load_model(model_path)
                        predictions = model.predict(X_val)
                    
                    if predictions is not None:
                        accuracy = self._calculate_accuracy(y_val, predictions)
                        relative_accuracy = accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
                        
                        accuracy_results['optimized_accuracies'][format_name] = {
                            'accuracy': accuracy,
                            'relative_accuracy': relative_accuracy,
                            'accuracy_drop': baseline_accuracy - accuracy
                        }
                    
                except Exception as e:
                    self.logger.warning(f"Accuracy validation failed for {format_name}: {e}")
                    accuracy_results['optimized_accuracies'][format_name] = {'error': str(e)}
            
            return accuracy_results
            
        except Exception as e:
            self.logger.error(f"Accuracy validation failed: {e}")
            return {}
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy based on predictions"""
        try:
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Classification - use argmax
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true
                return np.mean(y_pred_classes == y_true_classes)
            else:
                # Regression - use R²
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-8))
                
        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")
            return 0.0
    
    async def _predict_onnx(self, model_path: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions with ONNX model"""
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: X.astype(np.float32)})
            return outputs[0]
        except Exception as e:
            self.logger.error(f"ONNX prediction failed: {e}")
            return None
    
    async def _predict_tflite(self, model_path: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions with TensorFlow Lite model"""
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            predictions = []
            for i in range(len(X)):
                sample = X[i:i+1]
                if sample.shape != input_details[0]['shape']:
                    sample = np.resize(sample, input_details[0]['shape'])
                
                interpreter.set_tensor(input_details[0]['index'], sample.astype(np.float32))
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                predictions.append(prediction[0])
            
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"TensorFlow Lite prediction failed: {e}")
            return None
    
    def _meets_accuracy_threshold(self, accuracy_metrics: Dict[str, Any], threshold: float) -> bool:
        """Check if optimized models meet accuracy threshold"""
        try:
            optimized_accuracies = accuracy_metrics.get('optimized_accuracies', {})
            
            for format_name, metrics in optimized_accuracies.items():
                if 'relative_accuracy' in metrics:
                    if metrics['relative_accuracy'] < threshold:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Accuracy threshold check failed: {e}")
            return False
    
    async def _save_optimization_report(self, result: OptimizationResult, work_dir: Path):
        """Save detailed optimization report"""
        try:
            report = {
                'optimization_summary': {
                    'timestamp': datetime.utcnow().isoformat(),
                    'original_model': result.original_model_path,
                    'target_platform': result.target.target_platform,
                    'success': result.success,
                    'optimization_time': result.optimization_time,
                    'errors': result.errors
                },
                'optimized_models': result.optimized_models,
                'performance_metrics': result.performance_metrics,
                'accuracy_metrics': result.accuracy_metrics,
                'configuration': asdict(result.optimization_config),
                'target_specifications': asdict(result.target)
            }
            
            report_path = work_dir / "optimization_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Optimization report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization report: {e}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization runs"""
        try:
            history = []
            for result in self.optimization_history:
                history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'original_model': result.original_model_path,
                    'target_platform': result.target.target_platform,
                    'success': result.success,
                    'optimization_time': result.optimization_time,
                    'optimized_formats': list(result.optimized_models.keys()),
                    'errors': result.errors
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization history: {e}")
            return []
    
    async def cleanup_workspace(self, keep_recent: int = 5):
        """Clean up old optimization workspaces"""
        try:
            # Get all optimization directories
            opt_dirs = [d for d in self.workspace_dir.iterdir() if d.is_dir() and d.name.startswith('opt_')]
            
            # Sort by creation time (newest first)
            opt_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
            
            # Remove old directories
            removed_count = 0
            for dir_to_remove in opt_dirs[keep_recent:]:
                try:
                    shutil.rmtree(dir_to_remove)
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {dir_to_remove}: {e}")
            
            self.logger.info(f"Cleaned up {removed_count} old optimization workspaces")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Workspace cleanup failed: {e}")
            return 0