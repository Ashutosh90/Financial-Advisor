"""
XAI Agent - Provides explainability for ML predictions using SHAP and LIME
Responsibilities: Generate interpretable explanations for model decisions,
                  simplify explanations using GPT-4o mini for end users
"""
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from openai import OpenAI
import os
import json


class XAIAgent:
    """
    Agent responsible for explainability and interpretability of ML model predictions.
    
    Uses:
    - SHAP (SHapley Additive exPlanations) for global and local feature importance
    - LIME (Local Interpretable Model-agnostic Explanations) for instance-level explanations
    - OpenAI GPT-4o mini for simplifying technical explanations into user-friendly language
    """
    
    def __init__(self, risk_agent=None):
        """
        Initialize XAI Agent with risk model reference
        
        Args:
            risk_agent: RiskAgent instance with trained XGBoost model
        """
        self.risk_agent = risk_agent
        self.shap_explainer = None
        self.lime_explainer = None
        self.openai_client = None
        self._background_data = None
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client for explanation simplification"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for explanation simplification")
            else:
                logger.warning("OPENAI_API_KEY not found - will use rule-based explanations")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.openai_client = None
    
    def _initialize_shap_explainer(self, background_data: pd.DataFrame = None):
        """
        Initialize SHAP TreeExplainer for XGBoost model
        
        Args:
            background_data: Background dataset for SHAP (optional, for KernelSHAP)
        """
        if self.risk_agent is None or self.risk_agent.model is None:
            logger.warning("Risk model not available for SHAP initialization")
            return
        
        try:
            # Use TreeExplainer for XGBoost (fast and exact)
            self.shap_explainer = shap.TreeExplainer(self.risk_agent.model)
            self._background_data = background_data
            logger.info("SHAP TreeExplainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _initialize_lime_explainer(self, training_data: pd.DataFrame, feature_names: List[str], class_names: List[str]):
        """
        Initialize LIME explainer for tabular data
        
        Args:
            training_data: Training data for LIME
            feature_names: List of feature names
            class_names: List of class names (e.g., ['Conservative', 'Aggressive'])
        """
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data.values,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification',
                discretize_continuous=True
            )
            logger.info("LIME explainer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LIME explainer: {e}")
            self.lime_explainer = None
    
    def get_shap_explanation(
        self,
        features_df: pd.DataFrame,
        customer_id: int = None
    ) -> Dict:
        """
        Generate SHAP explanation for a prediction
        
        Args:
            features_df: Prepared features DataFrame for prediction
            customer_id: Optional customer ID for logging
            
        Returns:
            Dict with SHAP values, feature importances, and interpretation
        """
        if self.risk_agent is None or self.risk_agent.model is None:
            return {"error": "Risk model not available", "shap_values": None}
        
        try:
            # Initialize SHAP explainer if not done
            if self.shap_explainer is None:
                self._initialize_shap_explainer()
            
            if self.shap_explainer is None:
                return {"error": "SHAP explainer initialization failed", "shap_values": None}
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(features_df)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # For binary/multi-class, get values for predicted class
                prediction = self.risk_agent.model.predict(features_df)[0]
                class_shap_values = shap_values[prediction]
            else:
                class_shap_values = shap_values
            
            # Get feature names and values
            feature_names = self.risk_agent.feature_names
            feature_values = features_df.values[0]
            
            # Create feature importance dict sorted by absolute SHAP value
            if len(class_shap_values.shape) > 1:
                shap_array = class_shap_values[0]
            else:
                shap_array = class_shap_values
            
            feature_contributions = []
            for i, (name, shap_val, feat_val) in enumerate(zip(feature_names, shap_array, feature_values)):
                feature_contributions.append({
                    "feature": name,
                    "shap_value": float(shap_val),
                    "feature_value": float(feat_val),
                    "impact": "positive" if shap_val > 0 else "negative",
                    "impact_magnitude": abs(float(shap_val))
                })
            
            # Sort by absolute impact
            feature_contributions.sort(key=lambda x: x["impact_magnitude"], reverse=True)
            
            # Get top contributing features
            top_positive = [f for f in feature_contributions if f["impact"] == "positive"][:5]
            top_negative = [f for f in feature_contributions if f["impact"] == "negative"][:5]
            
            result = {
                "customer_id": customer_id,
                "method": "SHAP (TreeExplainer)",
                "base_value": float(self.shap_explainer.expected_value[0]) if hasattr(self.shap_explainer.expected_value, '__len__') else float(self.shap_explainer.expected_value),
                "feature_contributions": feature_contributions,
                "top_positive_factors": top_positive,
                "top_negative_factors": top_negative,
                "total_features": len(feature_names)
            }
            
            logger.info(f"Generated SHAP explanation for customer {customer_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {"error": str(e), "shap_values": None}
    
    def get_lime_explanation(
        self,
        features_df: pd.DataFrame,
        customer_id: int = None,
        num_features: int = 10
    ) -> Dict:
        """
        Generate LIME explanation for a prediction
        
        Args:
            features_df: Prepared features DataFrame for prediction
            customer_id: Optional customer ID for logging
            num_features: Number of top features to include in explanation
            
        Returns:
            Dict with LIME explanation
        """
        if self.risk_agent is None or self.risk_agent.model is None:
            return {"error": "Risk model not available", "lime_explanation": None}
        
        try:
            # Initialize LIME explainer if not done
            if self.lime_explainer is None:
                # Create a simple background dataset from the single instance
                # In production, use actual training data
                self._initialize_lime_explainer(
                    training_data=features_df,
                    feature_names=self.risk_agent.feature_names,
                    class_names=self.risk_agent.target_classes
                )
            
            if self.lime_explainer is None:
                return {"error": "LIME explainer initialization failed", "lime_explanation": None}
            
            # Get LIME explanation
            instance = features_df.values[0]
            
            def predict_fn(X):
                return self.risk_agent.model.predict_proba(X)
            
            explanation = self.lime_explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                top_labels=2
            )
            
            # Get predicted class
            prediction = self.risk_agent.model.predict(features_df)[0]
            predicted_class = self.risk_agent.target_classes[int(prediction)]
            
            # Extract feature weights for predicted class
            feature_weights = explanation.as_list(label=prediction)
            
            lime_features = []
            for feature_rule, weight in feature_weights:
                lime_features.append({
                    "rule": feature_rule,
                    "weight": float(weight),
                    "direction": "supports" if weight > 0 else "opposes",
                    "prediction": predicted_class
                })
            
            result = {
                "customer_id": customer_id,
                "method": "LIME (Local Interpretable Model-agnostic Explanations)",
                "predicted_class": predicted_class,
                "prediction_probability": float(explanation.predict_proba[prediction]),
                "feature_rules": lime_features,
                "intercept": float(explanation.intercept[prediction]) if hasattr(explanation, 'intercept') else None,
                "local_prediction": float(explanation.local_pred[0]) if hasattr(explanation, 'local_pred') else None
            }
            
            logger.info(f"Generated LIME explanation for customer {customer_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return {"error": str(e), "lime_explanation": None}
    
    def simplify_explanation_with_llm(
        self,
        shap_explanation: Dict,
        lime_explanation: Dict,
        risk_profile: Dict,
        customer_data: Dict = None
    ) -> str:
        """
        Use GPT-4o mini to simplify technical SHAP/LIME explanations into user-friendly language
        
        Args:
            shap_explanation: SHAP explanation dict
            lime_explanation: LIME explanation dict  
            risk_profile: Risk profile prediction result
            customer_data: Optional customer data for context
            
        Returns:
            Simplified, user-friendly explanation string
        """
        if self.openai_client is None:
            logger.warning("OpenAI client not available, using rule-based simplification")
            return self._rule_based_simplification(shap_explanation, lime_explanation, risk_profile, customer_data)
        
        try:
            # Prepare context for LLM
            risk_category = risk_profile.get('risk_category', 'Unknown')
            confidence = risk_profile.get('prediction_confidence', 0)
            
            # Get top SHAP factors
            top_shap_factors = ""
            if shap_explanation and "top_positive_factors" in shap_explanation:
                top_shap_factors = "\n".join([
                    f"- {f['feature']}: value={f['feature_value']:.2f}, impact={f['shap_value']:.4f}"
                    for f in shap_explanation.get("top_positive_factors", [])[:3]
                ])
                top_shap_factors += "\n" + "\n".join([
                    f"- {f['feature']}: value={f['feature_value']:.2f}, impact={f['shap_value']:.4f}"
                    for f in shap_explanation.get("top_negative_factors", [])[:3]
                ])
            
            # Get LIME rules
            lime_rules = ""
            if lime_explanation and "feature_rules" in lime_explanation:
                lime_rules = "\n".join([
                    f"- {r['rule']}: {r['direction']} prediction (weight={r['weight']:.4f})"
                    for r in lime_explanation.get("feature_rules", [])[:5]
                ])
            
            # Customer context
            customer_context = ""
            if customer_data:
                customer_context = f"""
Customer Profile:
- Age: {customer_data.get('age', 'N/A')}
- Annual Income: ₹{customer_data.get('annual_income', 0):,.0f}
- Net Worth: ₹{customer_data.get('net_worth', 0):,.0f}
- Total Debt: ₹{customer_data.get('total_debt', 0):,.0f}
- Investment Portfolio: ₹{customer_data.get('investment_portfolio_value', 0):,.0f}
"""
            
            prompt = f"""You are a financial advisor explaining an AI model's risk assessment to a customer in simple, friendly language.

The ML model has classified this customer as: **{risk_category}** risk profile (confidence: {confidence:.1%})

{customer_context}

Technical SHAP Analysis (feature importance):
{top_shap_factors if top_shap_factors else "Not available"}

Technical LIME Analysis (decision rules):
{lime_rules if lime_rules else "Not available"}

Please provide a clear, jargon-free explanation of:
1. Why the model classified this customer as {risk_category}
2. The top 3-4 most important factors that influenced this decision
3. What this means for their investment recommendations

Use simple language a non-technical person can understand. Format with bullet points for clarity.
Keep the explanation concise (under 200 words) but informative.
Use Indian Rupee (₹) for currency values.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly financial advisor who explains complex AI decisions in simple terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            simplified_explanation = response.choices[0].message.content
            logger.info(f"Generated simplified explanation using GPT-4o-mini")
            return simplified_explanation
            
        except Exception as e:
            logger.error(f"Error simplifying explanation with LLM: {e}")
            return self._rule_based_simplification(shap_explanation, lime_explanation, risk_profile, customer_data)
    
    def _rule_based_simplification(
        self,
        shap_explanation: Dict,
        lime_explanation: Dict,
        risk_profile: Dict,
        customer_data: Dict = None
    ) -> str:
        """
        Fallback rule-based explanation when LLM is not available
        """
        risk_category = risk_profile.get('risk_category', 'Moderate')
        confidence = risk_profile.get('prediction_confidence', 0.5)
        
        explanation_parts = [
            f"## Risk Assessment: {risk_category}\n",
            f"**Confidence Level:** {confidence:.1%}\n\n"
        ]
        
        # Add SHAP-based insights if available
        if shap_explanation and "top_positive_factors" in shap_explanation:
            explanation_parts.append("### Key Factors Supporting This Assessment:\n")
            for factor in shap_explanation.get("top_positive_factors", [])[:3]:
                feature_name = factor['feature'].replace('_', ' ').title()
                explanation_parts.append(f"• **{feature_name}**: Your value of {factor['feature_value']:.2f} contributes positively\n")
            
            if shap_explanation.get("top_negative_factors"):
                explanation_parts.append("\n### Factors That Could Shift Your Profile:\n")
                for factor in shap_explanation.get("top_negative_factors", [])[:2]:
                    feature_name = factor['feature'].replace('_', ' ').title()
                    explanation_parts.append(f"• **{feature_name}**: This factor suggests some caution\n")
        
        # Add customer context
        if customer_data:
            explanation_parts.append("\n### Your Financial Snapshot:\n")
            if customer_data.get('age'):
                explanation_parts.append(f"• Age: {customer_data['age']} years\n")
            if customer_data.get('annual_income'):
                explanation_parts.append(f"• Annual Income: ₹{customer_data['annual_income']:,.0f}\n")
            if customer_data.get('net_worth'):
                explanation_parts.append(f"• Net Worth: ₹{customer_data['net_worth']:,.0f}\n")
        
        return ''.join(explanation_parts)
    
    def explain_risk_profile(
        self,
        risk_profile: Dict,
        customer_data: Dict = None,
        features_df: pd.DataFrame = None,
        use_shap: bool = True,
        use_lime: bool = True,
        simplify_with_llm: bool = True
    ) -> Dict:
        """
        Generate comprehensive explanation for risk profile prediction
        
        Combines SHAP and LIME explanations with LLM simplification for
        a complete, user-friendly explanation.
        
        Args:
            risk_profile: Risk profile dict with risk_category, confidence, etc.
            customer_data: Optional customer data from database
            features_df: Prepared features DataFrame (required for SHAP/LIME)
            use_shap: Whether to generate SHAP explanation
            use_lime: Whether to generate LIME explanation
            simplify_with_llm: Whether to use GPT-4o-mini for simplification
            
        Returns:
            Dict with comprehensive explanation
        """
        try:
            customer_id = risk_profile.get('customer_id') or (customer_data.get('customer_id') if customer_data else None)
            risk_category = risk_profile.get('risk_category', 'Moderate')
            confidence = risk_profile.get('prediction_confidence', 0.5)
            
            result = {
                "customer_id": customer_id,
                "risk_category": risk_category,
                "confidence": confidence,
                "shap_explanation": None,
                "lime_explanation": None,
                "simplified_explanation": None,
                "key_factors": [],
                "explanation_methods_used": []
            }
            
            # Generate SHAP explanation if features available
            shap_explanation = None
            if use_shap and features_df is not None and self.risk_agent is not None:
                shap_explanation = self.get_shap_explanation(features_df, customer_id)
                if "error" not in shap_explanation:
                    result["shap_explanation"] = shap_explanation
                    result["explanation_methods_used"].append("SHAP")
                    
                    # Extract key factors from SHAP
                    for factor in shap_explanation.get("top_positive_factors", [])[:3]:
                        result["key_factors"].append({
                            "factor": factor['feature'],
                            "value": factor['feature_value'],
                            "impact": factor['shap_value'],
                            "direction": "positive"
                        })
            
            # Generate LIME explanation if features available
            lime_explanation = None
            if use_lime and features_df is not None and self.risk_agent is not None:
                lime_explanation = self.get_lime_explanation(features_df, customer_id)
                if "error" not in lime_explanation:
                    result["lime_explanation"] = lime_explanation
                    result["explanation_methods_used"].append("LIME")
            
            # Generate simplified explanation using LLM
            if simplify_with_llm:
                simplified = self.simplify_explanation_with_llm(
                    shap_explanation or {},
                    lime_explanation or {},
                    risk_profile,
                    customer_data
                )
                result["simplified_explanation"] = simplified
                result["explanation_methods_used"].append("GPT-4o-mini")
            else:
                # Use rule-based simplification
                result["simplified_explanation"] = self._rule_based_simplification(
                    shap_explanation or {},
                    lime_explanation or {},
                    risk_profile,
                    customer_data
                )
            
            # Generate legacy key_factors for backward compatibility
            if not result["key_factors"] and customer_data:
                result["key_factors"] = self._extract_key_factors_from_data(customer_data, risk_category)
            
            logger.info(f"Generated comprehensive explanation for {risk_category} profile using: {result['explanation_methods_used']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk profile explanation: {e}")
            return {
                "customer_id": customer_id if 'customer_id' in dir() else None,
                "risk_category": risk_profile.get('risk_category', 'Moderate'),
                "confidence": risk_profile.get('prediction_confidence', 0.5),
                "simplified_explanation": f"Risk profile assessed as {risk_profile.get('risk_category', 'Moderate')}",
                "key_factors": [],
                "explanation_methods_used": ["fallback"],
                "error": str(e)
            }
    
    def _extract_key_factors_from_data(self, customer_data: Dict, risk_category: str) -> List[Dict]:
        """Extract key factors from customer data for explanation"""
        key_factors = []
        
        age = customer_data.get('age', 0)
        annual_income = customer_data.get('annual_income', 0)
        net_worth = customer_data.get('net_worth', 0)
        total_debt = customer_data.get('total_debt', 0)
        dependents = customer_data.get('dependents', 0)
        
        if age:
            if age < 35:
                key_factors.append({
                    "factor": "age",
                    "description": f"Young age ({age}) allows for higher risk tolerance",
                    "direction": "supports_aggressive" if risk_category == "Aggressive" else "neutral"
                })
            elif age > 50:
                key_factors.append({
                    "factor": "age", 
                    "description": f"Age ({age}) suggests focus on capital preservation",
                    "direction": "supports_conservative" if risk_category == "Conservative" else "neutral"
                })
        
        if net_worth:
            if net_worth > 0:
                key_factors.append({
                    "factor": "net_worth",
                    "description": f"Positive net worth (₹{net_worth:,.0f}) provides financial cushion",
                    "direction": "positive"
                })
            else:
                key_factors.append({
                    "factor": "net_worth",
                    "description": f"Negative net worth suggests need for conservative approach",
                    "direction": "supports_conservative"
                })
        
        if total_debt and annual_income and total_debt > annual_income:
            key_factors.append({
                "factor": "debt_to_income",
                "description": "High debt-to-income ratio indicates need for cautious investment",
                "direction": "supports_conservative"
            })
        
        if dependents and dependents > 2:
            key_factors.append({
                "factor": "dependents",
                "description": f"Multiple dependents ({dependents}) require stable investments",
                "direction": "supports_conservative"
            })
        
        return key_factors
    
    def get_global_feature_importance(self) -> Dict:
        """
        Get global feature importance from the trained XGBoost model
        
        Returns:
            Dict with feature names and their importance scores
        """
        if self.risk_agent is None or self.risk_agent.model is None:
            return {"error": "Risk model not available", "feature_importance": {}}
        
        try:
            importance = self.risk_agent.model.feature_importances_
            feature_names = self.risk_agent.feature_names
            
            # Create sorted feature importance
            importance_dict = {}
            for name, imp in zip(feature_names, importance):
                importance_dict[name] = float(imp)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return {
                "model_type": "XGBoost",
                "total_features": len(feature_names),
                "feature_importance": sorted_importance,
                "top_10_features": dict(list(sorted_importance.items())[:10])
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {"error": str(e), "feature_importance": {}}

    def explain_recommendation(
        self,
        recommendation: Dict,
        risk_profile: Dict,
        market_data: Dict,
        simplify_with_llm: bool = True
    ) -> Dict:
        """
        Explain why specific investment recommendations were made
        
        Uses technical analysis and optionally GPT-4o-mini for user-friendly explanation
        """
        portfolio = recommendation.get("portfolio", {})
        risk_level = recommendation.get("risk_level", "Moderate")
        
        # Analyze portfolio composition
        equity_allocation = sum(
            v for k, v in portfolio.items() 
            if "equity" in k.lower() or "stock" in k.lower()
        )
        
        debt_allocation = sum(
            v for k, v in portfolio.items() 
            if "debt" in k.lower() or "fd" in k.lower() or "fixed" in k.lower()
        )
        
        technical_explanation = {
            "portfolio_breakdown": {
                "equity_exposure": equity_allocation,
                "debt_exposure": debt_allocation,
                "other_exposure": 100 - equity_allocation - debt_allocation
            },
            "risk_alignment": self._explain_risk_alignment(
                equity_allocation,
                risk_level
            ),
            "market_context": self._explain_market_context(market_data),
            "expected_outcome": self._explain_expected_outcome(
                recommendation.get("expected_return_annual", 0),
                risk_level
            )
        }
        
        # Generate LLM-simplified explanation if requested
        simplified_explanation = None
        if simplify_with_llm and self.openai_client:
            simplified_explanation = self._simplify_recommendation_with_llm(
                recommendation, risk_profile, market_data, technical_explanation
            )
        
        return {
            "technical_analysis": technical_explanation,
            "simplified_explanation": simplified_explanation,
            "portfolio": portfolio,
            "risk_level": risk_level
        }
    
    def _simplify_recommendation_with_llm(
        self,
        recommendation: Dict,
        risk_profile: Dict,
        market_data: Dict,
        technical_explanation: Dict
    ) -> str:
        """Use GPT-4o-mini to simplify recommendation explanation"""
        try:
            portfolio = recommendation.get("portfolio", {})
            risk_level = recommendation.get("risk_level", "Moderate")
            expected_return = recommendation.get("expected_return_annual", 0)
            
            portfolio_str = "\n".join([f"- {k}: {v}%" for k, v in portfolio.items()])
            
            prompt = f"""You are a financial advisor explaining investment recommendations to a customer.

Customer Risk Profile: {risk_profile.get('risk_category', 'Moderate')}
Recommended Portfolio Risk Level: {risk_level}
Expected Annual Return: {expected_return:.1f}%

Portfolio Allocation:
{portfolio_str}

Market Context:
- Repo Rate: {market_data.get('repo_rate', 6.5)}%
- Inflation Rate: {market_data.get('inflation_rate', 5.0)}%

Technical Analysis:
- {technical_explanation['risk_alignment']}
- {technical_explanation['market_context']}

Please explain:
1. Why this portfolio is suitable for the customer
2. How each asset class serves their financial goals
3. What returns they might expect and associated risks

Use simple language, bullet points, and keep it under 200 words.
Use Indian Rupee (₹) for any currency references.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly financial advisor explaining investment recommendations simply."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error simplifying recommendation: {e}")
            return None
    
    def _explain_risk_alignment(self, equity_pct: float, risk_level: str) -> str:
        """Explain how portfolio aligns with risk profile"""
        if risk_level == "Conservative":
            if equity_pct < 20:
                return "Portfolio is well-aligned with conservative risk profile, prioritizing capital safety."
            else:
                return "Portfolio has slightly higher equity exposure than typical conservative allocation."
        elif risk_level == "Moderate":
            if 25 <= equity_pct <= 50:
                return "Balanced portfolio appropriate for moderate risk tolerance."
            else:
                return "Portfolio allocation is adjusted based on market conditions."
        else:  # Aggressive
            if equity_pct > 50:
                return "High equity allocation suitable for aggressive growth strategy."
            else:
                return "Conservative positioning considering current market volatility."
    
    def _explain_market_context(self, market_data: Dict) -> str:
        """Explain current market conditions"""
        repo_rate = market_data.get("repo_rate", 6.5)
        inflation = market_data.get("inflation_rate", 5.0)
        
        if repo_rate > 6.5:
            context = "Interest rates are elevated, making debt instruments more attractive."
        elif repo_rate < 5.5:
            context = "Low interest rate environment favors growth assets."
        else:
            context = "Balanced interest rate scenario supports diversified allocation."
        
        if inflation > 6.0:
            context += " High inflation necessitates inflation-beating returns."
        
        return context
    
    def _explain_expected_outcome(self, expected_return: float, risk_level: str) -> str:
        """Explain expected returns in context"""
        return (
            f"Expected annual return of {expected_return:.1f}% is consistent with "
            f"{risk_level.lower()} risk strategy. Returns may vary based on market "
            f"conditions and specific instrument performance."
        )
