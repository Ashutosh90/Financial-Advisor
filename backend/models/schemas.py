"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, List
from datetime import datetime


class UserCreate(BaseModel):
    """Schema for creating a new user"""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: Optional[int] = Field(None, ge=18, le=100)
    occupation: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    name: str
    email: str
    age: Optional[int]
    occupation: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class FinancialProfileCreate(BaseModel):
    """Schema for creating financial profile"""
    user_id: int
    monthly_income: float = Field(..., gt=0)
    monthly_expenses: Optional[float] = Field(None, ge=0)
    current_savings: float = Field(default=0.0, ge=0)
    investment_amount: float = Field(..., gt=0)
    investment_duration_months: int = Field(..., gt=0, le=360)
    financial_goal: str
    risk_tolerance: Optional[str] = Field(default="Moderate")


class QueryRequest(BaseModel):
    """Schema for user query"""
    user_id: int
    query: str
    monthly_income: Optional[float] = Field(default=0, ge=0)
    investment_amount: Optional[float] = Field(default=0, ge=0)
    investment_duration_months: Optional[int] = Field(default=0, ge=0, le=360)
    financial_goal: Optional[str] = Field(default="Not Specified")
    risk_tolerance: Optional[str] = Field(default="Not Specified")
    monthly_expenses: Optional[float] = Field(default=0, ge=0)
    current_savings: Optional[float] = Field(default=0.0, ge=0)


class RiskProfileResponse(BaseModel):
    """Schema for risk profile response"""
    risk_category: str
    risk_score: float
    savings_ratio: float
    investment_capacity: float
    features: Dict[str, float]


class RecommendationResponse(BaseModel):
    """Schema for investment recommendation"""
    portfolio: Dict[str, float]
    expected_return: float
    risk_level: str
    reasoning: str
    asset_allocation: Dict[str, Dict[str, float]]


class ExplainabilityResponse(BaseModel):
    """Schema for XAI explanation"""
    shap_values: Dict[str, float]
    feature_importance: Dict[str, float]
    explanation_text: str
    lime_explanation: Optional[Dict] = None


class AdvisoryResponse(BaseModel):
    """Complete advisory response"""
    user_id: int
    session_id: str
    query: str
    risk_profile: RiskProfileResponse
    recommendation: RecommendationResponse
    explainability: ExplainabilityResponse
    timestamp: datetime


class FeedbackRequest(BaseModel):
    """Schema for user feedback"""
    session_id: str
    feedback: str = Field(..., pattern="^(Useful|Not useful)$")
    comments: Optional[str] = None
