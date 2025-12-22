"""
Model Retrainer Module

Automated model retraining pipeline that:
1. Loads recent data from database
2. Performs feature engineering using saved artifacts
3. Trains new XGBoost model with same hyperparameters
4. Evaluates against validation/test sets
5. Compares performance with current production model
6. Promotes new model if performance is better
7. Logs all experiments to MLflow
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

# ML Libraries
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# MLflow for tracking
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Automated model retraining pipeline for the risk profiling model.
    
    Triggered when drift is detected (PSI >= 0.25 or multiple features with CSI >= 0.25).
    Uses the same architecture and hyperparameters as the original model.
    """
    
    def __init__(
        self,
        db_path: str = "./data/risk_profiling.db",
        model_dir: str = "./models",
        mlflow_tracking_uri: str = "file:./mlruns",
        experiment_name: str = "risk_profiling_retraining"
    ):
        """
        Initialize the model retrainer.
        
        Args:
            db_path: Path to SQLite database
            model_dir: Directory containing model artifacts
            mlflow_tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
        """
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        
        # Load existing artifacts
        self.selected_features = self._load_json("selected_features.json")
        self.best_params = self._load_json("best_params.json")
        self.feature_eng_config = self._load_json("feature_engineering_config.json")
        self.label_encoder = self._load_pickle("label_encoder.pkl")
        
        # Extract categorical encoders from feature_engineering_config if available
        self.categorical_encoders = self._extract_categorical_encoders()
        
        # Setup MLflow
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
        
        logger.info("ModelRetrainer initialized successfully")
    
    def _extract_categorical_encoders(self) -> Optional[Dict]:
        """
        Extract categorical encoders from feature_engineering_config.
        
        The notebook saves encoding info in feature_engineering_config.json
        under categorical_encoding.categorical_features with the mapping.
        
        Returns:
            Dictionary of feature -> encoding mapping
        """
        if self.feature_eng_config is None:
            return None
        
        cat_encoding = self.feature_eng_config.get('categorical_encoding', {})
        cat_features = cat_encoding.get('categorical_features', {})
        
        if not cat_features:
            return None
        
        # Convert to encoding dictionaries
        encoders = {}
        for feature_name, encoding_info in cat_features.items():
            encoders[feature_name] = encoding_info.get('encoding', {})
        
        logger.info(f"Loaded categorical encoders for {len(encoders)} features")
        return encoders
    
    def _load_json(self, filename: str) -> Optional[dict]:
        """Load JSON artifact file."""
        filepath = self.model_dir / filename
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return None
    
    def _load_pickle(self, filename: str) -> Optional[object]:
        """Load pickle artifact file."""
        filepath = self.model_dir / filename
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return None
    
    def _save_json(self, data: dict, filename: str) -> str:
        """Save data to JSON file."""
        filepath = self.model_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return str(filepath)
    
    def _save_pickle(self, obj: object, filename: str) -> str:
        """Save object to pickle file."""
        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return str(filepath)
    
    def load_recent_data(
        self,
        snapshot_date: str = None,
        months_back: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load recent data for retraining.
        
        Args:
            snapshot_date: Specific snapshot to use
            months_back: Number of months of data to include
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get latest snapshot if not specified
        if snapshot_date is None:
            query = "SELECT MAX(snapshot_date) FROM risk_profiling_monthly_data"
            cursor = conn.cursor()
            cursor.execute(query)
            snapshot_date = cursor.fetchone()[0]
        
        logger.info(f"Loading data for snapshot: {snapshot_date}")
        
        # Load data by split
        train_df = pd.read_sql(
            f"SELECT * FROM risk_profiling_monthly_data WHERE snapshot_date = '{snapshot_date}' AND data_split = 'Train'",
            conn
        )
        val_df = pd.read_sql(
            f"SELECT * FROM risk_profiling_monthly_data WHERE snapshot_date = '{snapshot_date}' AND data_split = 'Validation'",
            conn
        )
        test_df = pd.read_sql(
            f"SELECT * FROM risk_profiling_monthly_data WHERE snapshot_date = '{snapshot_date}' AND data_split = 'Test'",
            conn
        )
        
        conn.close()
        
        logger.info(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering based on saved configuration.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        if self.feature_eng_config is None:
            logger.warning("No feature engineering config found, using raw features")
            return df
        
        # Apply derived features
        for feature_spec in self.feature_eng_config.get('derived_features', []):
            feature_name = feature_spec['name']
            transform = self.feature_eng_config['feature_transformations'].get(feature_name)
            
            if transform:
                numerator = transform['numerator']
                denominator = transform['denominator']
                offset = transform.get('offset', 1)
                
                if numerator in df.columns and denominator in df.columns:
                    df[feature_name] = df[numerator] / (df[denominator] + offset)
                    logger.debug(f"Created derived feature: {feature_name}")
        
        # Apply categorical encoding using dict mappings from feature_engineering_config
        if self.categorical_encoders:
            for col, encoding_map in self.categorical_encoders.items():
                if col in df.columns and encoding_map:
                    try:
                        # Get default value for unknown categories (first key's value)
                        default_value = next(iter(encoding_map.values()), 0)
                        df[col] = df[col].apply(
                            lambda x: encoding_map.get(str(x), default_value)
                        )
                        logger.debug(f"Encoded categorical: {col}")
                    except Exception as e:
                        logger.warning(f"Error encoding {col}: {e}")
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Prepare features for training/prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, labels)
        """
        # Apply feature engineering
        df = self.apply_feature_engineering(df)
        
        # Extract target variable if present
        labels = None
        if 'risk_profile' in df.columns:
            if self.label_encoder:
                labels = self.label_encoder.transform(df['risk_profile'])
            else:
                # Create new label encoder
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(df['risk_profile'])
        
        # Select only the required features
        if self.selected_features:
            available_features = [f for f in self.selected_features if f in df.columns]
            features_df = df[available_features]
        else:
            # Exclude non-feature columns
            exclude_cols = ['customer_id', 'risk_profile', 'data_split', 'snapshot_date', 
                          'gender', 'marital_status']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            features_df = df[feature_cols]
        
        logger.info(f"Prepared {len(features_df.columns)} features")
        
        return features_df, labels
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        params: Dict = None
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model hyperparameters (uses best_params if None)
            
        Returns:
            Trained XGBoost model
        """
        # Use saved best params or provided params
        model_params = params or self.best_params or {}
        
        # Default parameters if none available
        if not model_params:
            model_params = {
                'learning_rate': 0.3,
                'max_depth': 3,
                'n_estimators': 300,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'min_child_weight': 1,
                'gamma': 0.2,
                'reg_alpha': 0.1,
                'reg_lambda': 0,
                'scale_pos_weight': 1,
                'random_state': 42,
                'eval_metric': 'auc',
                'use_label_encoder': False
            }
        
        logger.info(f"Training with parameters: {model_params}")
        
        # Create and train model
        model = xgb.XGBClassifier(**model_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info(f"Model trained with {model.n_estimators} estimators")
        
        return model
    
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        logger.info(f"{dataset_name} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def compare_models(
        self,
        new_metrics: Dict,
        current_metrics_path: str = None
    ) -> Tuple[bool, Dict]:
        """
        Compare new model performance with current production model.
        
        Args:
            new_metrics: Metrics from newly trained model
            current_metrics_path: Path to current model metrics
            
        Returns:
            Tuple of (should_promote, comparison_dict)
        """
        # Load current model metrics
        current_metrics_path = current_metrics_path or (self.model_dir / "evaluation_metrics.json")
        
        try:
            with open(current_metrics_path, 'r') as f:
                current_metrics = json.load(f)
            
            # Extract test metrics
            if 'test' in current_metrics:
                current_test = current_metrics['test']
            else:
                current_test = current_metrics
                
        except FileNotFoundError:
            logger.warning("No current model metrics found, will promote new model")
            return True, {"reason": "No existing model metrics"}
        
        # Compare key metrics
        comparison = {
            'current_accuracy': current_test.get('accuracy', 0),
            'new_accuracy': new_metrics.get('accuracy', 0),
            'current_roc_auc': current_test.get('roc_auc', 0),
            'new_roc_auc': new_metrics.get('roc_auc', 0),
            'accuracy_change': new_metrics.get('accuracy', 0) - current_test.get('accuracy', 0),
            'roc_auc_change': new_metrics.get('roc_auc', 0) - current_test.get('roc_auc', 0)
        }
        
        # Promote if new model is better or within acceptable degradation (1%)
        should_promote = (
            new_metrics.get('roc_auc', 0) >= current_test.get('roc_auc', 0) - 0.01 and
            new_metrics.get('accuracy', 0) >= current_test.get('accuracy', 0) - 0.01
        )
        
        comparison['should_promote'] = should_promote
        comparison['reason'] = (
            "New model performance is acceptable" if should_promote 
            else "New model performance degraded significantly"
        )
        
        logger.info(f"Model comparison: {comparison}")
        
        return should_promote, comparison
    
    def save_model_artifacts(
        self,
        model: xgb.XGBClassifier,
        metrics: Dict,
        feature_importance: pd.DataFrame,
        version_suffix: str = None
    ) -> Dict[str, str]:
        """
        Save all model artifacts.
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            feature_importance: Feature importance DataFrame
            version_suffix: Optional version suffix for backup
            
        Returns:
            Dictionary of saved file paths
        """
        saved_paths = {}
        
        # Create backup with timestamp if requested
        if version_suffix:
            backup_dir = self.model_dir / f"backup_{version_suffix}"
            backup_dir.mkdir(exist_ok=True)
            # Backup current model
            for f in ['risk_model.json', 'evaluation_metrics.json', 'model_metadata.json']:
                src = self.model_dir / f
                if src.exists():
                    import shutil
                    shutil.copy(src, backup_dir / f)
            logger.info(f"Backed up current model to: {backup_dir}")
        
        # Save XGBoost model in JSON format
        model_path = self.model_dir / "risk_model.json"
        model.save_model(str(model_path))
        saved_paths['model'] = str(model_path)
        
        # Save as pickle too
        pickle_path = self.model_dir / "risk_profiling_model.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        saved_paths['model_pkl'] = str(pickle_path)
        
        # Save evaluation metrics
        metrics_path = self._save_json({
            'train': metrics.get('train', {}),
            'validation': metrics.get('validation', {}),
            'test': metrics.get('test', {})
        }, "evaluation_metrics.json")
        saved_paths['metrics'] = metrics_path
        
        # Save feature importance
        fi_path = self.model_dir / "feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        saved_paths['feature_importance'] = str(fi_path)
        
        # Save model metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'n_features': len(self.selected_features) if self.selected_features else 0,
            'n_estimators': model.n_estimators,
            'best_params': self.best_params,
            'test_accuracy': metrics.get('test', {}).get('accuracy'),
            'test_roc_auc': metrics.get('test', {}).get('roc_auc'),
            'retraining_triggered_by': 'drift_detection'
        }
        metadata_path = self._save_json(metadata, "model_metadata.json")
        saved_paths['metadata'] = metadata_path
        
        logger.info(f"Saved model artifacts to: {self.model_dir}")
        
        return saved_paths
    
    def retrain(
        self,
        snapshot_date: str = None,
        force: bool = False
    ) -> Dict:
        """
        Execute full retraining pipeline.
        
        Args:
            snapshot_date: Specific snapshot to train on
            force: Force retraining even if new model is worse
            
        Returns:
            Dictionary with retraining results
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting retraining pipeline - Run ID: {run_id}")
        
        result = {
            'run_id': run_id,
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Start MLflow run
            if MLFLOW_AVAILABLE:
                mlflow.start_run(run_name=f"retrain_{run_id}")
            
            # Load data
            train_df, val_df, test_df = self.load_recent_data(snapshot_date)
            result['data_loaded'] = {
                'train': len(train_df),
                'validation': len(val_df),
                'test': len(test_df)
            }
            
            # Prepare features
            X_train, y_train = self.prepare_features(train_df)
            X_val, y_val = self.prepare_features(val_df)
            X_test, y_test = self.prepare_features(test_df)
            
            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val)
            
            # Evaluate on all datasets
            train_metrics = self.evaluate_model(model, X_train, y_train, "Train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "Validation")
            test_metrics = self.evaluate_model(model, X_test, y_test, "Test")
            
            all_metrics = {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            }
            result['metrics'] = all_metrics
            
            # Log to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_params(self.best_params or {})
                for dataset, metrics in all_metrics.items():
                    for metric, value in metrics.items():
                        mlflow.log_metric(f"{dataset}_{metric}", value)
            
            # Compare with current model
            should_promote, comparison = self.compare_models(test_metrics)
            result['comparison'] = comparison
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Decide whether to promote
            if should_promote or force:
                # Save new model
                version_suffix = run_id
                saved_paths = self.save_model_artifacts(
                    model, all_metrics, feature_importance, version_suffix
                )
                result['saved_paths'] = saved_paths
                result['model_promoted'] = True
                result['status'] = 'success'
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_artifacts(str(self.model_dir))
                    mlflow.xgboost.log_model(model, "model")
                
                logger.info("✅ New model promoted to production")
            else:
                result['model_promoted'] = False
                result['status'] = 'completed_no_promotion'
                logger.warning("⚠️ New model not promoted due to performance degradation")
            
            # End MLflow run
            if MLFLOW_AVAILABLE:
                mlflow.end_run()
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"Retraining failed: {e}")
            
            if MLFLOW_AVAILABLE:
                mlflow.end_run(status='FAILED')
            
            import traceback
            traceback.print_exc()
        
        # Save retraining report
        report_dir = Path("./logs/retraining_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"retraining_report_{run_id}.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        result['report_path'] = str(report_path)
        logger.info(f"Retraining report saved to: {report_path}")
        
        return result


def trigger_retraining(
    snapshot_date: str = None,
    force: bool = False
) -> Dict:
    """
    Convenience function to trigger model retraining.
    
    Args:
        snapshot_date: Specific snapshot to train on
        force: Force promotion even if performance is worse
        
    Returns:
        Retraining result dictionary
    """
    retrainer = ModelRetrainer()
    return retrainer.retrain(snapshot_date=snapshot_date, force=force)


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL RETRAINING PIPELINE")
    print("=" * 60)
    
    result = trigger_retraining()
    
    print(f"\nRetraining Status: {result['status']}")
    print(f"Model Promoted: {result.get('model_promoted', False)}")
    
    if 'metrics' in result:
        print("\nTest Metrics:")
        for metric, value in result['metrics'].get('test', {}).items():
            print(f"  {metric}: {value:.4f}")
    
    if 'comparison' in result:
        print("\nModel Comparison:")
        print(f"  Accuracy Change: {result['comparison'].get('accuracy_change', 0):.4f}")
        print(f"  ROC-AUC Change: {result['comparison'].get('roc_auc_change', 0):.4f}")
    
    print(f"\nReport: {result.get('report_path', 'N/A')}")
