"""
FastAPI Main Application
Financial Advisory Multi-Agent System
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from loguru import logger
import sys

from backend.config import settings
from backend.models.database import User, FinancialProfile
from backend.models.schemas import (
    UserCreate, UserResponse, FinancialProfileCreate,
    QueryRequest, AdvisoryResponse, FeedbackRequest
)
from backend.models.db_manager import get_db_session, init_db
from backend.agents.orchestrator import Orchestrator

# Configure logging
logger.remove()
logger.add(sys.stderr, level=settings.log_level)
logger.add(settings.log_file, rotation="10 MB", level=settings.log_level)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Advisory Multi-Agent System",
    description="Personalized investment advisory using AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Starting Financial Advisory System")
    init_db()
    logger.info("Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Financial Advisory System")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Advisory Multi-Agent System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== Customer Data ====================

@app.get("/api/customer/search/{prefix}")
async def search_customer_ids(prefix: str, limit: int = 20):
    """Search for customer IDs matching the given prefix"""
    import sqlite3
    
    try:
        db_path = "./data/risk_profiling.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Search for customer IDs starting with prefix
        query = """
        SELECT DISTINCT customer_id 
        FROM risk_profiling_monthly_data 
        WHERE CAST(customer_id AS TEXT) LIKE ?
        ORDER BY customer_id
        LIMIT ?
        """
        
        cursor.execute(query, (f"{prefix}%", limit))
        rows = cursor.fetchall()
        conn.close()
        
        customer_ids = [row[0] for row in rows]
        logger.info(f"Found {len(customer_ids)} customer IDs matching prefix '{prefix}'")
        return {"customer_ids": customer_ids, "count": len(customer_ids)}
        
    except Exception as e:
        logger.error(f"Error searching customer IDs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search customer IDs"
        )


@app.get("/api/customer/{customer_id}")
async def get_customer_data(customer_id: int):
    """Get customer data from risk profiling database"""
    import sqlite3
    
    try:
        # Connect to risk profiling database
        db_path = "./data/risk_profiling.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query customer data
        query = """
        SELECT customer_id, age, annual_income, net_worth, risk_profile,
               customer_segment, education, dependents, total_debt,
               investment_portfolio_value, annual_investment_return
        FROM risk_profiling_monthly_data 
        WHERE customer_id = ?
        LIMIT 1
        """
        
        cursor.execute(query, (customer_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found"
            )
        
        # Convert to dictionary
        customer_data = dict(row)
        logger.info(f"Retrieved data for customer {customer_id}")
        return customer_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving customer data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve customer data"
        )


# ==================== User Management ====================

@app.post("/api/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db_session)
):
    """Create a new user"""
    try:
        # Check if user exists
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        db_user = User(
            name=user.name,
            email=user.email,
            age=user.age,
            occupation=user.occupation
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"Created user: {user.email}")
        return db_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db_session)
):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@app.get("/api/users/email/{email}", response_model=UserResponse)
async def get_user_by_email(
    email: str,
    db: Session = Depends(get_db_session)
):
    """Get user by email"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@app.post("/api/financial-profile/", status_code=status.HTTP_201_CREATED)
async def create_financial_profile(
    profile: FinancialProfileCreate,
    db: Session = Depends(get_db_session)
):
    """Create or update financial profile"""
    try:
        # Check if user exists
        user = db.query(User).filter(User.id == profile.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create financial profile
        db_profile = FinancialProfile(
            user_id=profile.user_id,
            monthly_income=profile.monthly_income,
            monthly_expenses=profile.monthly_expenses,
            current_savings=profile.current_savings,
            investment_amount=profile.investment_amount,
            investment_duration_months=profile.investment_duration_months,
            financial_goal=profile.financial_goal,
            risk_tolerance=profile.risk_tolerance
        )
        
        db.add(db_profile)
        db.commit()
        db.refresh(db_profile)
        
        logger.info(f"Created financial profile for user {profile.user_id}")
        return {"message": "Financial profile created successfully", "profile_id": db_profile.id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating financial profile: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create financial profile"
        )


# ==================== Advisory Endpoints ====================

@app.post("/api/advisory/customer/{customer_id}")
async def get_customer_advice(
    customer_id: int,
    query: str = "Provide investment recommendations based on my risk profile",
    session_id: str = None,
    db: Session = Depends(get_db_session)
):
    """
    Get investment advice using customer_id from risk profiling database
    Automatically fetches all customer data and generates recommendations
    Uses LangChain memory for session continuity
    
    Args:
        customer_id: Customer ID from the database
        query: User's question or request
        session_id: Optional session ID to continue an existing conversation
    """
    try:
        # Get customer data from risk profiling database
        import sqlite3
        db_path = "./data/risk_profiling.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM risk_profiling_monthly_data WHERE customer_id = ? LIMIT 1",
            (customer_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found"
            )
        
        customer_data = dict(row)
        
        # Initialize orchestrator
        orchestrator = Orchestrator(db)
        
        # Process query with customer data and session ID
        result = orchestrator.process_customer_query(
            customer_id=customer_id,
            customer_data=customer_data,
            query=query,
            session_id=session_id  # Pass session ID for memory continuity
        )
        
        logger.info(f"Generated advice for customer {customer_id}, session: {result.get('session_id')}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating customer advice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate advice: {str(e)}"
        )


@app.post("/api/advisory/query")
async def get_investment_advice(
    request: QueryRequest,
    db: Session = Depends(get_db_session)
):
    """
    Main endpoint: Get personalized investment advice using LangGraph
    
    This orchestrates all agents through a LangGraph state machine:
    1. Memory Agent retrieves user context
    2. Check for missing information
    3. Data Agent fetches market data (if info complete)
    4. Risk Agent assesses risk profile
    5. Advisor Agent generates recommendations (using LangChain)
    6. XAI Agent provides explanations
    7. Store session in memory
    """
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Initialize orchestrator
        orchestrator = Orchestrator(db)
        
        # Process query through all agents
        result = orchestrator.process_query(
            user_id=request.user_id,
            query=request.query,
            monthly_income=request.monthly_income,
            investment_amount=request.investment_amount,
            investment_duration_months=request.investment_duration_months,
            financial_goal=request.financial_goal,
            risk_tolerance=request.risk_tolerance,
            monthly_expenses=request.monthly_expenses
        )
        
        logger.info(f"Advisory generated for user {request.user_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating advisory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate advisory: {str(e)}"
        )


@app.post("/api/advisory/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db_session)
):
    """Submit feedback for a session"""
    try:
        orchestrator = Orchestrator(db)
        success = orchestrator.submit_feedback(
            session_id=feedback.session_id,
            feedback=feedback.feedback
        )
        
        if success:
            return {
                "message": "Feedback submitted successfully",
                "session_id": feedback.session_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@app.get("/api/advisory/history/{user_id}")
async def get_user_history(
    user_id: int,
    limit: int = 5,
    db: Session = Depends(get_db_session)
):
    """Get user's advisory history"""
    try:
        orchestrator = Orchestrator(db)
        history = orchestrator.get_user_history(user_id, limit)
        return history
        
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history"
        )


# ==================== Market Data Endpoints ====================

@app.get("/api/market/data")
async def get_market_data():
    """Get current market data"""
    try:
        from backend.agents.data_agent import DataAgent
        data_agent = DataAgent()
        market_data = data_agent.get_all_market_data()
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch market data"
        )


@app.get("/api/market/fd-rates")
async def get_fd_rates():
    """Get Fixed Deposit rates"""
    try:
        from backend.agents.data_agent import DataAgent
        data_agent = DataAgent()
        return data_agent.get_fd_rates()
    except Exception as e:
        logger.error(f"Error fetching FD rates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch FD rates"
        )


# ==================== Risk Assessment Endpoint ====================

@app.post("/api/risk/assess")
async def assess_risk(
    monthly_income: float,
    investment_amount: float,
    investment_duration_months: int,
    monthly_expenses: float = None,
    age: int = 35
):
    """Standalone risk assessment"""
    try:
        from backend.agents.risk_agent import RiskAgent
        risk_agent = RiskAgent()
        
        risk_profile = risk_agent.assess_risk(
            monthly_income=monthly_income,
            investment_amount=investment_amount,
            investment_duration_months=investment_duration_months,
            monthly_expenses=monthly_expenses,
            age=age
        )
        
        return risk_profile
        
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess risk"
        )


# ==================== Session Management Endpoints ====================

@app.get("/api/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    db: Session = Depends(get_db_session)
):
    """
    Get full conversation history for a session
    Returns all messages exchanged in the session
    """
    try:
        orchestrator = Orchestrator(db)
        history = orchestrator.get_session_history(session_id)
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
        
    except Exception as e:
        logger.error(f"Error retrieving session history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history"
        )


@app.get("/api/customer/{customer_id}/sessions")
async def get_customer_sessions(
    customer_id: int,
    limit: int = 10,
    db: Session = Depends(get_db_session)
):
    """
    Get recent sessions for a customer
    Returns list of past advisory sessions
    """
    try:
        orchestrator = Orchestrator(db)
        sessions = orchestrator.get_customer_sessions(customer_id, limit)
        
        return {
            "customer_id": customer_id,
            "session_count": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Error retrieving customer sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve customer sessions"
        )


@app.post("/api/session/{session_id}/end")
async def end_session(
    session_id: str,
    feedback: str = None,
    db: Session = Depends(get_db_session)
):
    """
    End a session and optionally provide feedback
    Stores feedback for learning and improvement
    """
    try:
        orchestrator = Orchestrator(db)
        orchestrator.end_customer_session(session_id, feedback)
        
        return {
            "session_id": session_id,
            "status": "ended",
            "feedback_received": feedback is not None
        }
        
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
