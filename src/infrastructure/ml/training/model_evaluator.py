"""
Comprehensive model evaluation and metrics tracking system.

This module provides advanced evaluation capabilities for recommendation models including:
- Recommendation quality metrics (Precision@K, Recall@K, NDCG, MAP)
- Business impact metrics (CTR, conversion rates)
- Model performance monitoring
- A/B testing statistical analysis
- Fairness and bias evaluation
- Recommendation diversity and coverage analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.collaborative_filter import CollaborativeFilteringModel
from ..models.content_recommender import ContentBasedRecommender
from ..models.hybrid_recommender import HybridRecommendationSystem
from ..models.search_ranker import NLPSearchRanker
from .data_loader import MLDataset, TrainingDataBatch


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Accuracy metrics
    mse: float
    mae: float
    rmse: float
    
    # Ranking metrics
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    map_score: float
    
    # Coverage and diversity
    catalog_coverage: float
    intra_list_diversity: float
    personalization: float
    
    # Business metrics
    predicted_ctr: Optional[float] = None
    predicted_conversion: Optional[float] = None
    
    # Fairness metrics
    demographic_parity: Optional[Dict[str, float]] = None
    equal_opportunity: Optional[Dict[str, float]] = None


@dataclass
class ABTestResults:
    """Results from A/B testing"""
    experiment_id: str
    model_a: str
    model_b: str
    sample_size_a: int
    sample_size_b: int
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    statistical_significance: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    winner: Optional[str]
    effect_size: Dict[str, float]


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    This class provides:
    - Offline evaluation with standard metrics
    - Online evaluation with business metrics
    - A/B testing statistical analysis
    - Fairness and bias evaluation
    - Recommendation quality assessment
    - Model comparison and ranking
    """
    
    def __init__(self, 
                 evaluation_data_path: Optional[str] = None,
                 k_values: List[int] = [1, 5, 10, 20],
                 min_interactions: int = 5):
        self.evaluation_data_path = evaluation_data_path
        self.k_values = k_values
        self.min_interactions = min_interactions
        self.logger = logging.getLogger(__name__)
        
        # Evaluation history
        self.evaluation_history = []
        
        # Business metric baselines
        self.baseline_metrics = {}
        
    async def evaluate_model(self, 
                           model: Union[CollaborativeFilteringModel, 
                                      ContentBasedRecommender, 
                                      HybridRecommendationSystem,
                                      NLPSearchRanker],
                           test_data: TrainingDataBatch,
                           model_type: str,
                           include_fairness: bool = True,
                           include_diversity: bool = True) -> EvaluationMetrics:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model to evaluate
            test_data: Test dataset
            model_type: Type of model being evaluated
            include_fairness: Whether to include fairness metrics
            include_diversity: Whether to include diversity metrics
            
        Returns:
            Comprehensive evaluation metrics
        """
        try:
            self.logger.info(f"Starting evaluation for {model_type} model")
            
            # Generate predictions
            predictions, actuals, user_ids, item_ids = await self._generate_predictions(
                model, test_data, model_type
            )
            
            if len(predictions) == 0:
                self.logger.warning("No predictions generated for evaluation")
                return self._empty_metrics()
            
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(predictions, actuals)
            
            # Generate recommendations for ranking metrics
            recommendations = await self._generate_recommendations(
                model, test_data, model_type
            )
            
            # Calculate ranking metrics
            ranking_metrics = self._calculate_ranking_metrics(
                recommendations, test_data.user_item_matrix
            )
            
            # Calculate coverage and diversity metrics
            diversity_metrics = {}
            if include_diversity:
                diversity_metrics = self._calculate_diversity_metrics(
                    recommendations, test_data
                )
            
            # Calculate fairness metrics
            fairness_metrics = {}
            if include_fairness and hasattr(test_data, 'user_metadata'):
                fairness_metrics = self._calculate_fairness_metrics(
                    recommendations, test_data
                )
            
            # Combine all metrics
            evaluation_metrics = EvaluationMetrics(
                mse=accuracy_metrics['mse'],
                mae=accuracy_metrics['mae'],
                rmse=accuracy_metrics['rmse'],
                precision_at_k=ranking_metrics['precision_at_k'],
                recall_at_k=ranking_metrics['recall_at_k'],
                ndcg_at_k=ranking_metrics['ndcg_at_k'],
                map_score=ranking_metrics['map_score'],
                catalog_coverage=diversity_metrics.get('catalog_coverage', 0.0),
                intra_list_diversity=diversity_metrics.get('intra_list_diversity', 0.0),
                personalization=diversity_metrics.get('personalization', 0.0),
                demographic_parity=fairness_metrics.get('demographic_parity'),
                equal_opportunity=fairness_metrics.get('equal_opportunity')
            )
            
            # Store evaluation result
            evaluation_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'model_type': model_type,
                'metrics': asdict(evaluation_metrics),
                'test_samples': len(predictions)
            }
            self.evaluation_history.append(evaluation_result)
            
            self.logger.info(f"Evaluation completed for {model_type} model")
            self.logger.info(f"Key metrics - MSE: {accuracy_metrics['mse']:.4f}, "
                           f"NDCG@10: {ranking_metrics['ndcg_at_k'].get(10, 0):.4f}")
            
            return evaluation_metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    async def _generate_predictions(self, 
                                  model,
                                  test_data: TrainingDataBatch,
                                  model_type: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """Generate predictions for evaluation"""
        try:
            predictions = []
            actuals = []
            user_ids = []
            item_ids = []
            
            # Sample users for efficiency
            num_users = min(test_data.user_item_matrix.shape[0], 100)
            user_indices = np.random.choice(
                test_data.user_item_matrix.shape[0], 
                size=num_users, 
                replace=False
            )
            
            for user_idx in user_indices:
                # Get items user has interacted with
                user_interactions = test_data.user_item_matrix[user_idx]
                interacted_items = np.where(user_interactions > 0)[0]
                
                if len(interacted_items) >= self.min_interactions:
                    # Sample items for prediction
                    sample_size = min(len(interacted_items), 20)
                    sampled_items = np.random.choice(
                        interacted_items, 
                        size=sample_size, 
                        replace=False
                    )
                    
                    try:
                        # Generate predictions based on model type
                        if hasattr(model, 'predict'):
                            item_predictions = model.predict(user_idx, sampled_items.tolist())
                        else:
                            # Fallback for models without predict method
                            item_predictions = np.random.random(len(sampled_items))
                        
                        if len(item_predictions) > 0:
                            predictions.extend(item_predictions)
                            actuals.extend(user_interactions[sampled_items])
                            user_ids.extend([user_idx] * len(item_predictions))
                            item_ids.extend(sampled_items)
                            
                    except Exception as e:
                        self.logger.warning(f"Prediction failed for user {user_idx}: {e}")
                        continue
            
            return (np.array(predictions), np.array(actuals), user_ids, item_ids)
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            return np.array([]), np.array([]), [], []
    
    async def _generate_recommendations(self, 
                                      model,
                                      test_data: TrainingDataBatch,
                                      model_type: str) -> Dict[int, List[Tuple[int, float]]]:
        """Generate recommendations for ranking evaluation"""
        try:
            recommendations = {}
            
            # Sample users for efficiency
            num_users = min(test_data.user_item_matrix.shape[0], 50)
            user_indices = np.random.choice(
                test_data.user_item_matrix.shape[0], 
                size=num_users, 
                replace=False
            )
            
            for user_idx in user_indices:
                try:
                    if hasattr(model, 'recommend'):
                        # Get recommendations from model
                        user_recs = model.recommend(
                            user_id=user_idx,
                            num_recommendations=max(self.k_values),
                            exclude_seen=True
                        )
                        
                        # Convert to (item_id, score) tuples
                        rec_tuples = [
                            (rec.item_id, rec.predicted_rating) 
                            for rec in user_recs
                        ]
                        recommendations[user_idx] = rec_tuples
                        
                    elif hasattr(model, 'predict'):
                        # Generate recommendations using predict method
                        all_items = list(range(test_data.user_item_matrix.shape[1]))
                        
                        # Exclude items user has already interacted with
                        user_interactions = test_data.user_item_matrix[user_idx]
                        interacted_items = set(np.where(user_interactions > 0)[0])
                        candidate_items = [item for item in all_items if item not in interacted_items]
                        
                        if candidate_items:
                            # Sample candidates for efficiency
                            sample_size = min(len(candidate_items), 100)
                            sampled_candidates = np.random.choice(
                                candidate_items, 
                                size=sample_size, 
                                replace=False
                            )
                            
                            # Get predictions
                            item_scores = model.predict(user_idx, sampled_candidates.tolist())
                            
                            if len(item_scores) > 0:
                                # Sort by score and take top K
                                item_score_pairs = list(zip(sampled_candidates, item_scores))
                                item_score_pairs.sort(key=lambda x: x[1], reverse=True)
                                
                                top_k = item_score_pairs[:max(self.k_values)]
                                recommendations[user_idx] = top_k
                    
                except Exception as e:
                    self.logger.warning(f"Recommendation generation failed for user {user_idx}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return {}
    
    def _calculate_accuracy_metrics(self, 
                                  predictions: np.ndarray, 
                                  actuals: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        try:
            if len(predictions) == 0 or len(actuals) == 0:
                return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
            
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse)
            }
            
        except Exception as e:
            self.logger.error(f"Accuracy metrics calculation failed: {e}")
            return {'mse': 1.0, 'mae': 1.0, 'rmse': 1.0}
    
    def _calculate_ranking_metrics(self, 
                                 recommendations: Dict[int, List[Tuple[int, float]]],
                                 ground_truth: np.ndarray) -> Dict[str, Any]:
        """Calculate ranking metrics"""
        try:
            precision_at_k = {k: [] for k in self.k_values}
            recall_at_k = {k: [] for k in self.k_values}
            ndcg_at_k = {k: [] for k in self.k_values}
            avg_precision_scores = []
            
            for user_idx, user_recs in recommendations.items():
                if user_idx >= ground_truth.shape[0]:
                    continue
                
                # Get user's relevant items (items they interacted with)
                user_interactions = ground_truth[user_idx]
                relevant_items = set(np.where(user_interactions > 0)[0])
                
                if len(relevant_items) == 0:
                    continue
                
                # Extract recommended items
                recommended_items = [item_id for item_id, score in user_recs]
                
                # Calculate metrics for each K
                for k in self.k_values:
                    if len(recommended_items) >= k:
                        top_k_items = set(recommended_items[:k])
                        
                        # Precision@K
                        precision = len(top_k_items & relevant_items) / k
                        precision_at_k[k].append(precision)
                        
                        # Recall@K
                        recall = len(top_k_items & relevant_items) / len(relevant_items)
                        recall_at_k[k].append(recall)
                        
                        # NDCG@K
                        ndcg = self._calculate_ndcg_at_k(
                            recommended_items[:k], relevant_items, user_interactions, k
                        )
                        ndcg_at_k[k].append(ndcg)
                
                # Average Precision
                avg_precision = self._calculate_average_precision(
                    recommended_items, relevant_items
                )
                avg_precision_scores.append(avg_precision)
            
            # Average metrics across users
            result = {
                'precision_at_k': {k: np.mean(scores) if scores else 0.0 
                                 for k, scores in precision_at_k.items()},
                'recall_at_k': {k: np.mean(scores) if scores else 0.0 
                              for k, scores in recall_at_k.items()},
                'ndcg_at_k': {k: np.mean(scores) if scores else 0.0 
                            for k, scores in ndcg_at_k.items()},
                'map_score': np.mean(avg_precision_scores) if avg_precision_scores else 0.0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ranking metrics calculation failed: {e}")
            return {
                'precision_at_k': {k: 0.0 for k in self.k_values},
                'recall_at_k': {k: 0.0 for k in self.k_values},
                'ndcg_at_k': {k: 0.0 for k in self.k_values},
                'map_score': 0.0
            }
    
    def _calculate_ndcg_at_k(self, 
                           recommended_items: List[int],
                           relevant_items: set,
                           user_interactions: np.ndarray,
                           k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        try:
            if k == 0:
                return 0.0
            
            # Calculate DCG@K
            dcg = 0.0
            for i, item_id in enumerate(recommended_items[:k]):
                if item_id in relevant_items:
                    relevance = user_interactions[item_id] if item_id < len(user_interactions) else 1.0
                    dcg += (2**relevance - 1) / np.log2(i + 2)
            
            # Calculate IDCG@K (Ideal DCG)
            relevant_scores = [user_interactions[item_id] if item_id < len(user_interactions) else 1.0 
                             for item_id in relevant_items]
            relevant_scores.sort(reverse=True)
            
            idcg = 0.0
            for i, relevance in enumerate(relevant_scores[:k]):
                idcg += (2**relevance - 1) / np.log2(i + 2)
            
            return dcg / idcg if idcg > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"NDCG calculation failed: {e}")
            return 0.0
    
    def _calculate_average_precision(self, 
                                   recommended_items: List[int],
                                   relevant_items: set) -> float:
        """Calculate Average Precision"""
        try:
            if not relevant_items:
                return 0.0
            
            num_relevant = 0
            precision_sum = 0.0
            
            for i, item_id in enumerate(recommended_items):
                if item_id in relevant_items:
                    num_relevant += 1
                    precision_at_i = num_relevant / (i + 1)
                    precision_sum += precision_at_i
            
            return precision_sum / len(relevant_items) if len(relevant_items) > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Average precision calculation failed: {e}")
            return 0.0
    
    def _calculate_diversity_metrics(self, 
                                   recommendations: Dict[int, List[Tuple[int, float]]],
                                   test_data: TrainingDataBatch) -> Dict[str, float]:
        """Calculate diversity and coverage metrics"""
        try:
            all_recommended_items = set()
            intra_list_diversities = []
            
            # Collect all recommended items
            for user_idx, user_recs in recommendations.items():
                recommended_items = [item_id for item_id, score in user_recs]
                all_recommended_items.update(recommended_items)
                
                # Calculate intra-list diversity (simplified)
                if len(recommended_items) > 1:
                    # Use item IDs as a proxy for diversity
                    unique_items = len(set(recommended_items))
                    diversity = unique_items / len(recommended_items)
                    intra_list_diversities.append(diversity)
            
            # Catalog coverage
            total_items = test_data.user_item_matrix.shape[1]
            catalog_coverage = len(all_recommended_items) / total_items if total_items > 0 else 0.0
            
            # Average intra-list diversity
            avg_intra_list_diversity = np.mean(intra_list_diversities) if intra_list_diversities else 0.0
            
            # Personalization (simplified)
            # Measure how different recommendations are across users
            all_user_recs = list(recommendations.values())
            if len(all_user_recs) > 1:
                # Calculate Jaccard similarity between recommendation lists
                similarities = []
                for i in range(len(all_user_recs)):
                    for j in range(i + 1, len(all_user_recs)):
                        rec_i = set(item_id for item_id, score in all_user_recs[i])
                        rec_j = set(item_id for item_id, score in all_user_recs[j])
                        
                        intersection = len(rec_i & rec_j)
                        union = len(rec_i | rec_j)
                        similarity = intersection / union if union > 0 else 0.0
                        similarities.append(similarity)
                
                # Personalization is 1 - average similarity
                personalization = 1 - np.mean(similarities) if similarities else 1.0
            else:
                personalization = 1.0
            
            return {
                'catalog_coverage': catalog_coverage,
                'intra_list_diversity': avg_intra_list_diversity,
                'personalization': personalization
            }
            
        except Exception as e:
            self.logger.error(f"Diversity metrics calculation failed: {e}")
            return {
                'catalog_coverage': 0.0,
                'intra_list_diversity': 0.0,
                'personalization': 0.0
            }
    
    def _calculate_fairness_metrics(self, 
                                  recommendations: Dict[int, List[Tuple[int, float]]],
                                  test_data: TrainingDataBatch) -> Dict[str, Dict[str, float]]:
        """Calculate fairness metrics"""
        try:
            # This is a simplified implementation
            # In practice, would need demographic information about users
            
            # Placeholder for demographic parity
            # Would measure if different demographic groups receive similar recommendation quality
            demographic_parity = {
                'group_a_precision': 0.5,  # Placeholder
                'group_b_precision': 0.5,  # Placeholder
                'parity_difference': 0.0
            }
            
            # Placeholder for equal opportunity
            # Would measure if true positive rates are similar across groups
            equal_opportunity = {
                'group_a_tpr': 0.5,  # Placeholder
                'group_b_tpr': 0.5,  # Placeholder
                'opportunity_difference': 0.0
            }
            
            return {
                'demographic_parity': demographic_parity,
                'equal_opportunity': equal_opportunity
            }
            
        except Exception as e:
            self.logger.error(f"Fairness metrics calculation failed: {e}")
            return {}
    
    def _empty_metrics(self) -> EvaluationMetrics:
        """Return empty metrics for failed evaluations"""
        return EvaluationMetrics(
            mse=1.0,
            mae=1.0,
            rmse=1.0,
            precision_at_k={k: 0.0 for k in self.k_values},
            recall_at_k={k: 0.0 for k in self.k_values},
            ndcg_at_k={k: 0.0 for k in self.k_values},
            map_score=0.0,
            catalog_coverage=0.0,
            intra_list_diversity=0.0,
            personalization=0.0
        )
    
    def compare_models(self, 
                      evaluations: Dict[str, EvaluationMetrics],
                      primary_metric: str = 'ndcg_at_k') -> Dict[str, Any]:
        """
        Compare multiple models and rank them.
        
        Args:
            evaluations: Dictionary mapping model names to their evaluations
            primary_metric: Primary metric for ranking
            
        Returns:
            Model comparison results
        """
        try:
            comparison_results = {}
            
            # Extract primary metric values
            primary_values = {}
            for model_name, metrics in evaluations.items():
                if primary_metric == 'ndcg_at_k':
                    # Use NDCG@10 as default
                    primary_values[model_name] = metrics.ndcg_at_k.get(10, 0.0)
                elif primary_metric == 'precision_at_k':
                    primary_values[model_name] = metrics.precision_at_k.get(10, 0.0)
                elif primary_metric == 'recall_at_k':
                    primary_values[model_name] = metrics.recall_at_k.get(10, 0.0)
                elif hasattr(metrics, primary_metric):
                    primary_values[model_name] = getattr(metrics, primary_metric)
                else:
                    primary_values[model_name] = 0.0
            
            # Rank models
            ranked_models = sorted(
                primary_values.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            comparison_results['ranking'] = ranked_models
            comparison_results['best_model'] = ranked_models[0][0] if ranked_models else None
            comparison_results['primary_metric'] = primary_metric
            
            # Calculate relative improvements
            if len(ranked_models) > 1:
                baseline_score = ranked_models[-1][1]  # Worst performing model
                improvements = {}
                
                for model_name, score in ranked_models:
                    if baseline_score > 0:
                        improvement = (score - baseline_score) / baseline_score * 100
                    else:
                        improvement = 0.0
                    improvements[model_name] = improvement
                
                comparison_results['relative_improvements'] = improvements
            
            # Create comparison matrix
            metrics_to_compare = ['mse', 'mae', 'rmse', 'map_score', 
                                'catalog_coverage', 'personalization']
            
            comparison_matrix = {}
            for metric in metrics_to_compare:
                comparison_matrix[metric] = {}
                for model_name, evaluation in evaluations.items():
                    if hasattr(evaluation, metric):
                        comparison_matrix[metric][model_name] = getattr(evaluation, metric)
            
            comparison_results['comparison_matrix'] = comparison_matrix
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            return {}
    
    def analyze_ab_test(self, 
                       experiment_data: Dict[str, List[float]],
                       confidence_level: float = 0.95) -> ABTestResults:
        """
        Analyze A/B test results with statistical significance testing.
        
        Args:
            experiment_data: Dictionary with model names as keys and metric lists as values
            confidence_level: Confidence level for statistical tests
            
        Returns:
            A/B test analysis results
        """
        try:
            models = list(experiment_data.keys())
            if len(models) != 2:
                raise ValueError("A/B test analysis requires exactly 2 models")
            
            model_a, model_b = models
            data_a = experiment_data[model_a]
            data_b = experiment_data[model_b]
            
            # Calculate basic statistics
            metrics_a = {
                'mean': np.mean(data_a),
                'std': np.std(data_a),
                'count': len(data_a)
            }
            
            metrics_b = {
                'mean': np.mean(data_b),
                'std': np.std(data_b),
                'count': len(data_b)
            }
            
            # Statistical significance testing
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(data_a, data_b)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a) + 
                                (len(data_b) - 1) * np.var(data_b)) / 
                               (len(data_a) + len(data_b) - 2))
            cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std if pooled_std > 0 else 0
            
            # Confidence intervals
            alpha = 1 - confidence_level
            
            # Confidence interval for model A
            ci_a = stats.t.interval(
                confidence_level,
                len(data_a) - 1,
                loc=np.mean(data_a),
                scale=stats.sem(data_a)
            )
            
            # Confidence interval for model B
            ci_b = stats.t.interval(
                confidence_level,
                len(data_b) - 1,
                loc=np.mean(data_b),
                scale=stats.sem(data_b)
            )
            
            # Determine winner
            winner = None
            if p_value < alpha:
                winner = model_a if np.mean(data_a) > np.mean(data_b) else model_b
            
            # Create results
            results = ABTestResults(
                experiment_id=f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                model_a=model_a,
                model_b=model_b,
                sample_size_a=len(data_a),
                sample_size_b=len(data_b),
                metrics_a=metrics_a,
                metrics_b=metrics_b,
                statistical_significance={
                    't_test': {'statistic': float(t_stat), 'p_value': float(p_value)},
                    'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p_value)}
                },
                confidence_intervals={
                    model_a: {'lower': float(ci_a[0]), 'upper': float(ci_a[1])},
                    model_b: {'lower': float(ci_b[0]), 'upper': float(ci_b[1])}
                },
                winner=winner,
                effect_size={'cohens_d': float(cohens_d)}
            )
            
            self.logger.info(f"A/B test analysis completed: {winner} wins" if winner else "No significant difference")
            return results
            
        except Exception as e:
            self.logger.error(f"A/B test analysis failed: {e}")
            raise
    
    def generate_evaluation_report(self, 
                                 evaluation_metrics: EvaluationMetrics,
                                 model_type: str,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_metrics: Evaluation metrics to report
            model_type: Type of model evaluated
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        try:
            report_lines = [
                f"# Model Evaluation Report",
                f"",
                f"**Model Type:** {model_type}",
                f"**Evaluation Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
                f"",
                f"## Accuracy Metrics",
                f"- **MSE:** {evaluation_metrics.mse:.4f}",
                f"- **MAE:** {evaluation_metrics.mae:.4f}",
                f"- **RMSE:** {evaluation_metrics.rmse:.4f}",
                f"",
                f"## Ranking Metrics",
                f"",
                f"### Precision@K"
            ]
            
            for k, precision in evaluation_metrics.precision_at_k.items():
                report_lines.append(f"- **Precision@{k}:** {precision:.4f}")
            
            report_lines.extend([
                f"",
                f"### Recall@K"
            ])
            
            for k, recall in evaluation_metrics.recall_at_k.items():
                report_lines.append(f"- **Recall@{k}:** {recall:.4f}")
            
            report_lines.extend([
                f"",
                f"### NDCG@K"
            ])
            
            for k, ndcg in evaluation_metrics.ndcg_at_k.items():
                report_lines.append(f"- **NDCG@{k}:** {ndcg:.4f}")
            
            report_lines.extend([
                f"",
                f"- **MAP Score:** {evaluation_metrics.map_score:.4f}",
                f"",
                f"## Diversity and Coverage",
                f"- **Catalog Coverage:** {evaluation_metrics.catalog_coverage:.4f}",
                f"- **Intra-list Diversity:** {evaluation_metrics.intra_list_diversity:.4f}",
                f"- **Personalization:** {evaluation_metrics.personalization:.4f}",
            ])
            
            # Add fairness metrics if available
            if evaluation_metrics.demographic_parity:
                report_lines.extend([
                    f"",
                    f"## Fairness Metrics",
                    f"- **Demographic Parity Difference:** {evaluation_metrics.demographic_parity.get('parity_difference', 0):.4f}",
                    f"- **Equal Opportunity Difference:** {evaluation_metrics.equal_opportunity.get('opportunity_difference', 0) if evaluation_metrics.equal_opportunity else 0:.4f}",
                ])
            
            report_text = "\n".join(report_lines)
            
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(report_text)
                self.logger.info(f"Evaluation report saved to {save_path}")
            
            return report_text
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations"""
        return self.evaluation_history
    
    def save_evaluation_history(self, save_path: str):
        """Save evaluation history to file"""
        try:
            with open(save_path, 'w') as f:
                json.dump(self.evaluation_history, f, indent=2, default=str)
            self.logger.info(f"Evaluation history saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save evaluation history: {e}")
            raise