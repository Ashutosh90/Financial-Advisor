"""
Memory Agent - Stores and recalls user preferences and past interactions
Responsibilities: Learn from feedback, personalize future recommendations
"""
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from backend.models.database import User, Session as DBSession, UserPreference
from loguru import logger
from datetime import datetime


class MemoryAgent:
    """Agent responsible for memory and personalization"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def store_session(
        self,
        user_id: int,
        session_id: str,
        query: str,
        recommendation: Dict,
        risk_profile: Dict
    ) -> bool:
        """Store user session for future reference"""
        try:
            session = DBSession(
                user_id=user_id,
                session_id=session_id,
                query=query,
                recommendation=recommendation,
                risk_profile=risk_profile
            )
            
            self.db.add(session)
            self.db.commit()
            
            logger.info(f"Stored session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing session: {e}")
            self.db.rollback()
            return False
    
    def update_feedback(self, session_id: str, feedback: str) -> bool:
        """Update session with user feedback"""
        try:
            session = self.db.query(DBSession).filter(
                DBSession.session_id == session_id
            ).first()
            
            if session:
                session.feedback = feedback
                self.db.commit()
                
                # Learn from feedback
                self._learn_from_feedback(session)
                
                logger.info(f"Updated feedback for session {session_id}: {feedback}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating feedback: {e}")
            self.db.rollback()
            return False
    
    def _learn_from_feedback(self, session: DBSession):
        """
        Learn preferences from user feedback
        """
        try:
            if session.feedback == "Useful":
                # Extract preferences from successful recommendation
                recommendation = session.recommendation
                
                if recommendation and "portfolio" in recommendation:
                    portfolio = recommendation["portfolio"]
                    
                    # Find dominant asset class
                    dominant_asset = max(portfolio.items(), key=lambda x: x[1])
                    
                    # Store or update preference
                    self._update_preference(
                        session.user_id,
                        "preferred_asset_class",
                        dominant_asset[0],
                        confidence_score=0.7
                    )
                    
                    # Store risk preference
                    if "risk_level" in recommendation:
                        self._update_preference(
                            session.user_id,
                            "preferred_risk_level",
                            recommendation["risk_level"],
                            confidence_score=0.8
                        )
            
            logger.info(f"Learned from feedback for user {session.user_id}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def _update_preference(
        self,
        user_id: int,
        key: str,
        value: str,
        confidence_score: float
    ):
        """Update or create user preference"""
        try:
            # Check if preference exists
            pref = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.preference_key == key
            ).first()
            
            if pref:
                # Update existing preference
                pref.preference_value = value
                pref.confidence_score = min(
                    1.0,
                    pref.confidence_score + 0.1  # Increase confidence
                )
                pref.updated_at = datetime.utcnow()
            else:
                # Create new preference
                pref = UserPreference(
                    user_id=user_id,
                    preference_key=key,
                    preference_value=value,
                    confidence_score=confidence_score
                )
                self.db.add(pref)
            
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
            self.db.rollback()
    
    def get_user_preferences(self, user_id: int) -> Dict[str, str]:
        """Retrieve all preferences for a user"""
        try:
            preferences = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id
            ).all()
            
            pref_dict = {
                pref.preference_key: pref.preference_value 
                for pref in preferences
                if pref.confidence_score > 0.5
            }
            
            logger.info(f"Retrieved {len(pref_dict)} preferences for user {user_id}")
            return pref_dict
            
        except Exception as e:
            logger.error(f"Error retrieving preferences: {e}")
            return {}
    
    def get_past_sessions(
        self,
        user_id: int,
        limit: int = 5
    ) -> List[Dict]:
        """Get past sessions for a user"""
        try:
            sessions = self.db.query(DBSession).filter(
                DBSession.user_id == user_id
            ).order_by(
                DBSession.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    "session_id": s.session_id,
                    "query": s.query,
                    "recommendation": s.recommendation,
                    "feedback": s.feedback,
                    "created_at": s.created_at.isoformat()
                }
                for s in sessions
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving past sessions: {e}")
            return []
    
    def get_personalized_context(self, user_id: int) -> Dict:
        """
        Get personalized context for recommendations
        """
        try:
            preferences = self.get_user_preferences(user_id)
            past_sessions = self.get_past_sessions(user_id, limit=3)
            
            # Extract patterns
            preferred_assets = []
            successful_strategies = []
            
            for session in past_sessions:
                if session.get("feedback") == "Useful":
                    rec = session.get("recommendation", {})
                    if "portfolio" in rec:
                        portfolio = rec["portfolio"]
                        top_asset = max(portfolio.items(), key=lambda x: x[1])
                        preferred_assets.append(top_asset[0])
            
            context = {
                "preferences": preferences,
                "preferred_assets": list(set(preferred_assets)),
                "past_feedback_summary": {
                    "total_sessions": len(past_sessions),
                    "useful_count": sum(
                        1 for s in past_sessions 
                        if s.get("feedback") == "Useful"
                    )
                }
            }
            
            logger.info(f"Generated personalized context for user {user_id}")
            return context
            
        except Exception as e:
            logger.error(f"Error getting personalized context: {e}")
            return {
                "preferences": {},
                "preferred_assets": [],
                "past_feedback_summary": {}
            }
    
    def clear_old_sessions(self, days: int = 90) -> int:
        """Clear sessions older than specified days"""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            deleted = self.db.query(DBSession).filter(
                DBSession.created_at < cutoff_date
            ).delete()
            
            self.db.commit()
            logger.info(f"Cleared {deleted} old sessions")
            return deleted
            
        except Exception as e:
            logger.error(f"Error clearing old sessions: {e}")
            self.db.rollback()
            return 0
