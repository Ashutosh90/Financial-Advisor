"""
Database models for Financial Advisor system
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User profile model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True)
    age = Column(Integer)
    occupation = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    financial_profiles = relationship("FinancialProfile", back_populates="user")
    sessions = relationship("Session", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")


class FinancialProfile(Base):
    """User's financial information"""
    __tablename__ = "financial_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Financial data
    monthly_income = Column(Float, nullable=False)
    monthly_expenses = Column(Float)
    current_savings = Column(Float, default=0.0)
    investment_amount = Column(Float)
    investment_duration_months = Column(Integer)
    
    # Goals
    financial_goal = Column(String(200))
    risk_tolerance = Column(String(50))  # Conservative, Moderate, Aggressive
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="financial_profiles")


class Session(Base):
    """User interaction sessions"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), unique=True, index=True)
    
    query = Column(Text)
    recommendation = Column(JSON)
    risk_profile = Column(JSON)
    feedback = Column(String(50))  # Useful, Not useful
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class UserPreference(Base):
    """Learned user preferences from Memory Agent"""
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    preference_key = Column(String(100))  # e.g., "preferred_asset_class"
    preference_value = Column(Text)  # e.g., "Fixed Deposits"
    confidence_score = Column(Float, default=0.5)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="preferences")


class MarketData(Base):
    """Cached market data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    data_type = Column(String(50))  # fd_rates, mf_nav, inflation, repo_rate
    source = Column(String(100))
    data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)


# ============ New Memory-related Tables ============

class CustomerSession(Base):
    """Customer advisory sessions - tracks interactions per customer"""
    __tablename__ = "customer_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, index=True, nullable=False)  # From risk_profiling database
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    feedback = Column(String(50), nullable=True)  # Useful, Not useful, etc.
    
    # Summary of session
    total_interactions = Column(Integer, default=0)
    risk_profile_used = Column(String(50), nullable=True)  # Conservative, Aggressive, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ConversationMessage(Base):
    """Individual messages in a conversation - for long-term memory storage"""
    __tablename__ = "conversation_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("customer_sessions.session_id"), index=True)
    
    role = Column(String(20), nullable=False)  # 'human', 'ai', 'system'
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, nullable=True)  # Additional context
    
    created_at = Column(DateTime, default=datetime.utcnow)


class InteractionMetadata(Base):
    """Stores metadata about interactions for learning"""
    __tablename__ = "interaction_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("customer_sessions.session_id"), index=True)
    
    # Recommendation details
    interaction_data = Column(JSON, nullable=True)
    
    # For learning
    was_helpful = Column(Integer, nullable=True)  # 1 = yes, 0 = no, null = unknown
    
    created_at = Column(DateTime, default=datetime.utcnow)


class CustomerPreference(Base):
    """Learned preferences for customers (by customer_id from risk_profiling DB)"""
    __tablename__ = "customer_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, index=True, nullable=False)
    
    preference_key = Column(String(100), nullable=False)  # e.g., "preferred_risk_level"
    preference_value = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.5)
    
    # Tracking
    times_observed = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
