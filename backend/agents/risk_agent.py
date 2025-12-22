"""
Risk Agent - Assesses user's risk profile using XGBoost
Responsibilities: Analyze financial data, categorize risk tolerance
"""
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from loguru import logger
import pickle
import os
import sqlite3
import json


class RiskAgent:
    """Agent responsible for risk profiling using XGBoost"""
    
    def __init__(
        self, 
        model_path: str = None,
        features_path: str = None,
        encoder_path: str = None,
        feature_eng_config_path: str = None,
        db_path: str = None
    ):
        self.model_path = model_path or "./models/risk_model.json"
        self.features_path = features_path or "./models/selected_features.json"
        self.encoder_path = encoder_path or "./models/label_encoder.pkl"
        self.feature_eng_config_path = feature_eng_config_path or "./models/feature_engineering_config.json"
        self.db_path = db_path or "./data/risk_profiling.db"
        
        self.model = None
        self.feature_names = []
        self.label_encoder = None
        self.feature_engineering_config = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the trained XGBoost model and load feature names"""
        try:
            # Load the trained model
            if os.path.exists(self.model_path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info(f"Loaded XGBoost model from {self.model_path}")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load selected features
            if os.path.exists(self.features_path):
                with open(self.features_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} features from {self.features_path}")
            else:
                logger.error(f"Features file not found: {self.features_path}")
                raise FileNotFoundError(f"Features file not found: {self.features_path}")
            
            # Load target label encoder
            if os.path.exists(self.encoder_path):
                with open(self.encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"Loaded target label encoder with classes: {self.label_encoder.classes_}")
            else:
                logger.error(f"Target label encoder not found: {self.encoder_path}")
                raise FileNotFoundError(f"Target label encoder not found: {self.encoder_path}")
            
            # Load feature engineering configuration (includes all transformations and encoding)
            if os.path.exists(self.feature_eng_config_path):
                with open(self.feature_eng_config_path, 'r') as f:
                    self.feature_engineering_config = json.load(f)
                
                derived_count = len(self.feature_engineering_config.get('derived_features', []))
                cat_features = self.feature_engineering_config.get('categorical_encoding', {}).get('categorical_features', {})
                
                logger.info(f"Loaded feature engineering config:")
                logger.info(f"  - {derived_count} derived features")
                logger.info(f"  - {len(cat_features)} categorical features with encoding")
            else:
                logger.error(f"Feature engineering config not found: {self.feature_eng_config_path}")
                raise FileNotFoundError(f"Feature engineering config required: {self.feature_eng_config_path}")
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def fetch_customer_data(self, customer_id: int) -> Optional[pd.DataFrame]:
        """
        Fetch customer data from SQLite database
        
        Args:
            customer_id: The customer ID to fetch data for
            
        Returns:
            DataFrame with customer data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Fetch customer data
            query = f"""
            SELECT * FROM risk_profiling_monthly_data 
            WHERE customer_id = {customer_id}
            LIMIT 1
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"No data found for customer_id: {customer_id}")
                return None
            
            logger.info(f"Fetched data for customer_id: {customer_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching customer data: {e}")
            return None
    
    def prepare_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction by selecting only the required features.
        Uses artifact-based feature engineering config when available.
        
        Args:
            customer_data: Raw customer data from database
            
        Returns:
            DataFrame with selected features in correct order
        """
        try:
            # Make a copy to avoid modifying original data
            data = customer_data.copy()
            
            # Engineer missing features using config if available
            if self.feature_engineering_config:
                logger.debug("Using artifact-based feature engineering")
                for feature_spec in self.feature_engineering_config.get('derived_features', []):
                    feature_name = feature_spec['name']
                    
                    # Skip if feature already exists
                    if feature_name in data.columns:
                        continue
                    
                    # Get transformation details
                    transform = self.feature_engineering_config['feature_transformations'].get(feature_name, {})
                    
                    if transform:
                        numerator = transform.get('numerator')
                        denominator = transform.get('denominator')
                        offset = transform.get('offset', 0)
                        operation = transform.get('operation', 'divide')
                        
                        # Check if required columns exist
                        if numerator in data.columns and denominator in data.columns:
                            if operation == 'divide':
                                data[feature_name] = data[numerator] / (data[denominator] + offset)
                                logger.debug(f"Created {feature_name} using config: {numerator} / ({denominator} + {offset})")
                        else:
                            logger.warning(f"Missing columns for {feature_name}: {numerator}, {denominator}")
                            data[feature_name] = 0.0
                    else:
                        logger.warning(f"No transformation config found for {feature_name}")
                        data[feature_name] = 0.0
            else:
                # Fallback to hardcoded feature engineering for backward compatibility
                logger.debug("Using hardcoded feature engineering (fallback)")
                if 'income_per_dependent' not in data.columns:
                    if 'annual_income' in data.columns and 'dependents' in data.columns:
                        data['income_per_dependent'] = data['annual_income'] / (data['dependents'] + 1)
                    else:
                        data['income_per_dependent'] = 0.0
                
                if 'debt_coverage_ratio' not in data.columns:
                    if 'annual_income' in data.columns and 'total_debt' in data.columns:
                        data['debt_coverage_ratio'] = data['annual_income'] / (data['total_debt'] + 1)
                    else:
                        data['debt_coverage_ratio'] = 10.0
                
                if 'investment_efficiency' not in data.columns:
                    if 'annual_investment_return' in data.columns and 'investment_portfolio_value' in data.columns:
                        data['investment_efficiency'] = data['annual_investment_return'] / (data['investment_portfolio_value'] + 1)
                    else:
                        data['investment_efficiency'] = 0.0
            
            # Select only the features used by the model
            features_df = data[self.feature_names].copy()
            
            # Handle categorical encoding using feature engineering config
            categorical_features = self.feature_engineering_config.get('categorical_encoding', {}).get('categorical_features', {})
            
            for col in features_df.columns:
                if features_df[col].dtype == 'object':
                    if col in categorical_features:
                        # Use encoding from config
                        encoding_map = categorical_features[col]['encoding']
                        features_df[col] = features_df[col].map(encoding_map).fillna(0)
                        logger.debug(f"Encoded {col} using config mapping")
                    else:
                        logger.warning(f"No encoding found for {col} in config, setting to 0")
                        features_df[col] = 0
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    
    def predict_risk_profile(self, customer_id: int) -> Dict:
        """
        Predict risk profile for a customer based on their ID
        
        Args:
            customer_id: The customer ID to predict for
            
        Returns:
            Dictionary containing risk prediction and probabilities
        """
        try:
            # Fetch customer data from database
            customer_data = self.fetch_customer_data(customer_id)
            
            if customer_data is None:
                logger.error(f"Cannot predict: No data found for customer_id {customer_id}")
                return {
                    "customer_id": customer_id,
                    "error": "Customer not found in database",
                    "risk_category": "Unknown",
                    "risk_score": 0.0,
                    "risk_probabilities": {}
                }
            
            # Prepare features
            features_df = self.prepare_features(customer_data)
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # Decode prediction
            risk_category = self.label_encoder.inverse_transform([prediction])[0]
            
            # Create probability dictionary
            risk_probabilities = {
                str(self.label_encoder.classes_[i]): float(probabilities[i])
                for i in range(len(self.label_encoder.classes_))
            }
            
            # Get customer info for response
            customer_info = {
                "customer_id": int(customer_data['customer_id'].values[0]),
                "age": int(customer_data['age'].values[0]) if 'age' in customer_data.columns else None,
                "annual_income": float(customer_data['annual_income'].values[0]) if 'annual_income' in customer_data.columns else None,
                "net_worth": float(customer_data['net_worth'].values[0]) if 'net_worth' in customer_data.columns else None,
            }
            
            result = {
                **customer_info,
                "risk_category": risk_category,
                "risk_score": float(probabilities[prediction]),
                "risk_probabilities": risk_probabilities,
                "prediction_confidence": float(max(probabilities)),
                "features_used": len(self.feature_names)
            }
            
            logger.info(f"Predicted risk profile for customer {customer_id}: {risk_category} (confidence: {result['prediction_confidence']:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting risk profile: {e}")
            return {
                "customer_id": customer_id,
                "error": str(e),
                "risk_category": "Error",
                "risk_score": 0.0,
                "risk_probabilities": {}
            }
    
    def assess_risk(
        self,
        customer_id: Optional[int] = None,
        monthly_income: Optional[float] = None,
        investment_amount: Optional[float] = None,
        investment_duration_months: Optional[int] = None,
        monthly_expenses: Optional[float] = None,
        age: Optional[int] = None,
        user_risk_tolerance: Optional[str] = None
    ) -> Dict:
        """
        Assess user's risk profile
        
        If customer_id is provided, fetches data from database and predicts.
        Otherwise, uses provided parameters for legacy support.
        
        Returns risk category, score, and supporting data
        """
        # If customer_id is provided, use the new prediction method
        if customer_id is not None:
            return self.predict_risk_profile(customer_id)
        
        # Legacy method for backward compatibility
        logger.warning("Using legacy risk assessment method. Consider using customer_id instead.")
        
        # Return minimal response for legacy calls
        return {
            "risk_category": user_risk_tolerance or "Conservative",
            "predicted_risk": user_risk_tolerance or "Conservative",
            "risk_score": 0.5,
            "risk_probabilities": {
                "Conservative": 0.6,
                "Aggressive": 0.4
            },
            "message": "Legacy assessment mode. Use customer_id for ML-based prediction."
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance.tolist()))
            return feature_importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
