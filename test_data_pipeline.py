#!/usr/bin/env python3
"""
Test Data Pipeline Script

A simplified script to test the data ingestion pipeline without the complexity
of the full production setup. Use this for initial testing and validation.
"""

import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.presentation.demo.sample_data import SampleDataGenerator
from src.domain.entities.property import Property
from src.domain.entities.user import User

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataPipelineTest:
    """Simple test of the data pipeline components"""
    
    def __init__(self):
        self.generator = SampleDataGenerator()
        self.test_results = {}
        
    def test_sample_data_generation(self) -> bool:
        """Test sample data generation"""
        try:
            logger.info("Testing sample data generation...")
            
            # Generate test data
            properties = self.generator.generate_properties(count=10)
            users = self.generator.generate_users(count=5)
            interactions = self.generator.generate_interactions(users, properties, count=20)
            
            # Validate data
            assert len(properties) == 10, "Property count mismatch"
            assert len(users) == 5, "User count mismatch"
            assert len(interactions) == 20, "Interaction count mismatch"
            
            # Check property data quality
            for prop in properties:
                assert prop.title, "Property missing title"
                assert prop.price > 0, "Property price invalid"
                assert prop.location, "Property missing location"
                assert isinstance(prop.amenities, list), "Amenities not a list"
                
            # Check user data quality
            for user in users:
                assert user.email, "User missing email"
                assert "@" in user.email, "Invalid email format"
                assert user.preferences, "User missing preferences"
                
            # Check interaction data
            for interaction in interactions:
                assert interaction.property_id, "Interaction missing property_id"
                assert interaction.interaction_type, "Interaction missing type"
                assert interaction.timestamp, "Interaction missing timestamp"
                
            logger.info("✅ Sample data generation test passed")
            self.test_results["sample_data_generation"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Sample data generation test failed: {e}")
            self.test_results["sample_data_generation"] = False
            return False
            
    def test_property_features(self) -> bool:
        """Test property feature extraction"""
        try:
            logger.info("Testing property feature extraction...")
            
            # Generate test properties
            properties = self.generator.generate_properties(count=5)
            
            # Extract features
            for prop in properties:
                features = self.generator._extract_property_features(prop)
                
                assert isinstance(features, list), "Features not a list"
                assert len(features) > 0, "No features extracted"
                assert all(isinstance(f, (int, float)) for f in features), "Non-numeric features"
                
            logger.info("✅ Property feature extraction test passed")
            self.test_results["property_features"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Property feature extraction test failed: {e}")
            self.test_results["property_features"] = False
            return False
            
    def test_user_features(self) -> bool:
        """Test user feature extraction"""
        try:
            logger.info("Testing user feature extraction...")
            
            # Generate test users with interactions
            properties = self.generator.generate_properties(count=10)
            users = self.generator.generate_users(count=3)
            interactions = self.generator.generate_interactions(users, properties, count=15)
            
            # Extract features
            for user in users:
                features = self.generator._extract_user_features(user)
                
                assert isinstance(features, list), "Features not a list"
                assert len(features) > 0, "No features extracted"
                assert all(isinstance(f, (int, float)) for f in features), "Non-numeric features"
                
            logger.info("✅ User feature extraction test passed")
            self.test_results["user_features"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ User feature extraction test failed: {e}")
            self.test_results["user_features"] = False
            return False
            
    def test_ml_training_data(self) -> bool:
        """Test ML training data generation"""
        try:
            logger.info("Testing ML training data generation...")
            
            # Generate test data
            properties = self.generator.generate_properties(count=20)
            users = self.generator.generate_users(count=10)
            interactions = self.generator.generate_interactions(users, properties, count=50)
            
            # Generate ML training data
            ml_data = self.generator.generate_ml_training_data(users, properties)
            
            # Validate ML data structure
            required_keys = [
                "user_item_matrix", "property_features", "user_features",
                "user_mapping", "property_mapping", "feature_names"
            ]
            
            for key in required_keys:
                assert key in ml_data, f"Missing key: {key}"
                
            # Check dimensions
            matrix = ml_data["user_item_matrix"]
            assert matrix.shape == (len(users), len(properties)), "Matrix dimension mismatch"
            
            prop_features = ml_data["property_features"]
            assert prop_features.shape[0] == len(properties), "Property features dimension mismatch"
            
            user_features = ml_data["user_features"]
            assert user_features.shape[0] == len(users), "User features dimension mismatch"
            
            logger.info("✅ ML training data generation test passed")
            self.test_results["ml_training_data"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ ML training data generation test failed: {e}")
            self.test_results["ml_training_data"] = False
            return False
            
    def test_data_export(self) -> bool:
        """Test data export functionality"""
        try:
            logger.info("Testing data export...")
            
            # Generate test data
            properties = self.generator.generate_properties(count=5)
            users = self.generator.generate_users(count=3)
            interactions = self.generator.generate_interactions(users, properties, count=10)
            
            # Export to temporary file
            temp_file = "test_export.json"
            self.generator.export_sample_data(temp_file, users, properties, interactions)
            
            # Verify export
            assert os.path.exists(temp_file), "Export file not created"
            
            # Load and validate exported data
            with open(temp_file, 'r') as f:
                exported_data = json.load(f)
                
            required_sections = ["users", "properties", "interactions"]
            for section in required_sections:
                assert section in exported_data, f"Missing section: {section}"
                
            assert len(exported_data["users"]) == 3, "User count mismatch in export"
            assert len(exported_data["properties"]) == 5, "Property count mismatch in export"
            assert len(exported_data["interactions"]) == 10, "Interaction count mismatch in export"
            
            # Cleanup
            os.remove(temp_file)
            
            logger.info("✅ Data export test passed")
            self.test_results["data_export"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Data export test failed: {e}")
            self.test_results["data_export"] = False
            if os.path.exists("test_export.json"):
                os.remove("test_export.json")
            return False
            
    def test_performance_metrics(self) -> bool:
        """Test performance metrics generation"""
        try:
            logger.info("Testing performance metrics generation...")
            
            metrics = self.generator.generate_performance_metrics()
            
            # Validate metrics structure
            required_sections = ["accuracy", "response_times", "user_engagement", "system_health"]
            for section in required_sections:
                assert section in metrics, f"Missing metrics section: {section}"
                
            # Check accuracy metrics
            accuracy = metrics["accuracy"]
            assert "collaborative_filtering" in accuracy, "Missing collaborative filtering accuracy"
            assert "dates" in accuracy, "Missing dates in accuracy metrics"
            assert len(accuracy["dates"]) == 30, "Incorrect number of date entries"
            
            # Check response times
            response_times = metrics["response_times"]
            assert "avg_response_ms" in response_times, "Missing average response time"
            assert len(response_times["hours"]) == 24, "Incorrect number of hour entries"
            
            logger.info("✅ Performance metrics test passed")
            self.test_results["performance_metrics"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            self.test_results["performance_metrics"] = False
            return False
            
    def run_all_tests(self) -> Dict[str, any]:
        """Run all tests and return results"""
        logger.info("🧪 Starting data pipeline tests...")
        start_time = time.time()
        
        tests = [
            self.test_sample_data_generation,
            self.test_property_features,
            self.test_user_features,
            self.test_ml_training_data,
            self.test_data_export,
            self.test_performance_metrics
        ]
        
        results = {
            "total_tests": len(tests),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": self.test_results,
            "duration_seconds": 0,
            "overall_success": False
        }
        
        # Run each test
        for test in tests:
            try:
                if test():
                    results["passed_tests"] += 1
                else:
                    results["failed_tests"] += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} crashed: {e}")
                results["failed_tests"] += 1
                
        # Calculate results
        results["duration_seconds"] = round(time.time() - start_time, 2)
        results["overall_success"] = results["failed_tests"] == 0
        
        return results
        
    def create_sample_dataset(self, output_dir: str = "sample_data") -> Dict[str, any]:
        """Create a comprehensive sample dataset for development"""
        logger.info("📊 Creating comprehensive sample dataset...")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate comprehensive data
            properties = self.generator.generate_properties(count=100)
            users = self.generator.generate_users(count=50)
            interactions = self.generator.generate_interactions(users, properties, count=500)
            
            # Generate ML data
            ml_data = self.generator.generate_ml_training_data(users, properties)
            
            # Generate performance metrics
            performance_metrics = self.generator.generate_performance_metrics()
            
            # Export all data
            self.generator.export_sample_data(
                os.path.join(output_dir, "sample_dataset.json"),
                users, properties, interactions
            )
            
            # Save ML training data
            import numpy as np
            np.savez(
                os.path.join(output_dir, "ml_training_data.npz"),
                user_item_matrix=ml_data["user_item_matrix"],
                property_features=ml_data["property_features"],
                user_features=ml_data["user_features"]
            )
            
            # Save mappings and metadata
            with open(os.path.join(output_dir, "ml_metadata.json"), 'w') as f:
                json.dump({
                    "user_mapping": ml_data["user_mapping"],
                    "property_mapping": ml_data["property_mapping"],
                    "feature_names": ml_data["feature_names"]
                }, f, indent=2)
                
            # Save performance metrics
            with open(os.path.join(output_dir, "performance_metrics.json"), 'w') as f:
                json.dump(performance_metrics, f, indent=2)
                
            # Create summary
            summary = {
                "dataset_created_at": datetime.now().isoformat(),
                "properties_count": len(properties),
                "users_count": len(users),
                "interactions_count": len(interactions),
                "ml_features": {
                    "property_features": ml_data["property_features"].shape[1],
                    "user_features": ml_data["user_features"].shape[1]
                },
                "files_created": [
                    "sample_dataset.json",
                    "ml_training_data.npz",
                    "ml_metadata.json",
                    "performance_metrics.json",
                    "dataset_summary.json"
                ]
            }
            
            with open(os.path.join(output_dir, "dataset_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"✅ Sample dataset created in '{output_dir}' directory")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to create sample dataset: {e}")
            return {"error": str(e)}


def main():
    """Main execution function"""
    print("🏠 Rental ML System - Data Pipeline Test")
    print("=" * 40)
    
    tester = SimpleDataPipelineTest()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Display results
    print(f"\n📊 Test Results:")
    print(f"Total tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']} ✅")
    print(f"Failed: {results['failed_tests']} ❌")
    print(f"Duration: {results['duration_seconds']}s")
    
    if results["overall_success"]:
        print("\n🎉 All tests passed!")
        
        # Create sample dataset
        print("\n📦 Creating sample dataset...")
        dataset_summary = tester.create_sample_dataset()
        
        if "error" not in dataset_summary:
            print(f"✅ Sample dataset created with:")
            print(f"  - {dataset_summary['properties_count']} properties")
            print(f"  - {dataset_summary['users_count']} users")
            print(f"  - {dataset_summary['interactions_count']} interactions")
            print(f"  - {dataset_summary['ml_features']['property_features']} property features")
            print(f"  - {dataset_summary['ml_features']['user_features']} user features")
            
            print(f"\n📁 Files created in 'sample_data' directory:")
            for file in dataset_summary['files_created']:
                print(f"  - {file}")
                
            print("\n🚀 Next steps:")
            print("1. Use this sample data to test your ML models")
            print("2. Set up the full data ingestion pipeline")
            print("3. Test with real scraped data")
            
        else:
            print(f"❌ Failed to create dataset: {dataset_summary['error']}")
            
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        print("\nFailed tests:")
        for test_name, passed in results["test_details"].items():
            if not passed:
                print(f"  - {test_name}")


if __name__ == "__main__":
    main()