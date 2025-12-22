"""
Session Memory Manager - LangChain-based short-term and SQL-based long-term memory
Handles conversation history within a session and persists to database for long-term storage
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from loguru import logger


class SQLChatMessageHistory(BaseChatMessageHistory):
    """
    SQL-backed chat message history for long-term storage
    Stores conversation messages in the database for persistence across sessions
    """
    
    def __init__(
        self,
        session_id: str,
        db: DBSession,
        table_name: str = "conversation_history"
    ):
        self.session_id = session_id
        self.db = db
        self.table_name = table_name
        self._messages: List[BaseMessage] = []
        
        # Load existing messages from database
        self._load_from_db()
    
    def _load_from_db(self):
        """Load messages from database"""
        try:
            from backend.models.database import ConversationMessage
            
            messages = self.db.query(ConversationMessage).filter(
                ConversationMessage.session_id == self.session_id
            ).order_by(ConversationMessage.created_at).all()
            
            for msg in messages:
                if msg.role == "human":
                    self._messages.append(HumanMessage(content=msg.content))
                elif msg.role == "ai":
                    self._messages.append(AIMessage(content=msg.content))
                elif msg.role == "system":
                    self._messages.append(SystemMessage(content=msg.content))
                    
            logger.debug(f"Loaded {len(self._messages)} messages for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error loading messages from DB: {e}")
            self._messages = []
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Return all messages"""
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history and persist to database"""
        self._messages.append(message)
        self._persist_message(message)
    
    def _persist_message(self, message: BaseMessage) -> None:
        """Persist a single message to database"""
        try:
            from backend.models.database import ConversationMessage
            
            role = "human" if isinstance(message, HumanMessage) else \
                   "ai" if isinstance(message, AIMessage) else "system"
            
            db_message = ConversationMessage(
                session_id=self.session_id,
                role=role,
                content=message.content,
                message_metadata=getattr(message, 'additional_kwargs', {})
            )
            
            self.db.add(db_message)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error persisting message: {e}")
            self.db.rollback()
    
    def add_user_message(self, message: str) -> None:
        """Add a user message"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message"""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear the message history"""
        try:
            from backend.models.database import ConversationMessage
            
            self.db.query(ConversationMessage).filter(
                ConversationMessage.session_id == self.session_id
            ).delete()
            self.db.commit()
            
            self._messages = []
            logger.info(f"Cleared conversation history for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing messages: {e}")
            self.db.rollback()


class SessionMemoryManager:
    """
    Manages both short-term (in-session) and long-term (database) memory
    
    Short-term: LangChain ConversationBufferWindowMemory for current conversation context
    Long-term: SQL-backed storage for persistence across sessions
    """
    
    def __init__(self, db: DBSession, window_size: int = 10):
        """
        Initialize memory manager
        
        Args:
            db: SQLAlchemy database session
            window_size: Number of recent messages to keep in short-term memory
        """
        self.db = db
        self.window_size = window_size
        self._session_memories: Dict[str, ConversationBufferWindowMemory] = {}
        self._chat_histories: Dict[str, SQLChatMessageHistory] = {}
        
        logger.info(f"SessionMemoryManager initialized with window_size={window_size}")
    
    def get_or_create_session(self, customer_id: int, session_id: str = None) -> str:
        """
        Get existing session or create new one for a customer
        
        Args:
            customer_id: Customer ID
            session_id: Optional existing session ID
            
        Returns:
            Session ID
        """
        # If session_id provided and already in memory, return it
        if session_id and session_id in self._session_memories:
            return session_id
        
        # If session_id provided, check if it exists in database
        if session_id:
            existing = self._load_existing_session(session_id)
            if existing:
                return session_id
        
        # Generate new session ID
        new_session_id = session_id or f"session_{customer_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create short-term memory (windowed conversation buffer)
        self._session_memories[new_session_id] = ConversationBufferWindowMemory(
            k=self.window_size,
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )
        
        # Create long-term history (SQL-backed)
        self._chat_histories[new_session_id] = SQLChatMessageHistory(
            session_id=new_session_id,
            db=self.db
        )
        
        # Store session in database (only if not already exists)
        if not session_id:  # Only create new record for new sessions
            self._create_session_record(customer_id, new_session_id)
        
        logger.info(f"Created new session {new_session_id} for customer {customer_id}")
        return new_session_id
    
    def _load_existing_session(self, session_id: str) -> bool:
        """Load existing session from database into memory"""
        try:
            from backend.models.database import CustomerSession
            
            session = self.db.query(CustomerSession).filter(
                CustomerSession.session_id == session_id
            ).first()
            
            if session:
                # Create short-term memory
                self._session_memories[session_id] = ConversationBufferWindowMemory(
                    k=self.window_size,
                    return_messages=True,
                    memory_key="chat_history",
                    input_key="input",
                    output_key="output"
                )
                
                # Load long-term history from database
                self._chat_histories[session_id] = SQLChatMessageHistory(
                    session_id=session_id,
                    db=self.db
                )
                
                # Populate short-term memory from long-term history
                messages = self._chat_histories[session_id].messages
                for i in range(0, len(messages) - 1, 2):
                    if i + 1 < len(messages):
                        user_msg = messages[i]
                        ai_msg = messages[i + 1]
                        if isinstance(user_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                            self._session_memories[session_id].save_context(
                                {"input": user_msg.content},
                                {"output": ai_msg.content}
                            )
                
                logger.info(f"Loaded existing session {session_id} with {len(messages)} messages")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error loading existing session: {e}")
            return False
    
    def _create_session_record(self, customer_id: int, session_id: str):
        """Create session record in database"""
        try:
            from backend.models.database import CustomerSession
            
            session_record = CustomerSession(
                customer_id=customer_id,
                session_id=session_id,
                started_at=datetime.utcnow()
            )
            
            self.db.add(session_record)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error creating session record: {e}")
            self.db.rollback()
    
    def add_interaction(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        metadata: Dict = None
    ):
        """
        Add a user-AI interaction to both short-term and long-term memory
        
        Args:
            session_id: Session identifier
            user_message: User's input message
            ai_response: AI's response
            metadata: Optional metadata (recommendation details, etc.)
        """
        if session_id not in self._session_memories:
            logger.warning(f"Session {session_id} not found, creating new one")
            self._session_memories[session_id] = ConversationBufferWindowMemory(
                k=self.window_size,
                return_messages=True,
                memory_key="chat_history"
            )
            self._chat_histories[session_id] = SQLChatMessageHistory(
                session_id=session_id,
                db=self.db
            )
        
        # Add to short-term memory
        self._session_memories[session_id].save_context(
            {"input": user_message},
            {"output": ai_response}
        )
        
        # Add to long-term memory (SQL)
        self._chat_histories[session_id].add_user_message(user_message)
        self._chat_histories[session_id].add_ai_message(ai_response)
        
        # Store interaction metadata if provided
        if metadata:
            self._store_interaction_metadata(session_id, metadata)
        
        logger.debug(f"Added interaction to session {session_id}")
    
    def _store_interaction_metadata(self, session_id: str, metadata: Dict):
        """Store interaction metadata for learning"""
        try:
            from backend.models.database import InteractionMetadata
            
            record = InteractionMetadata(
                session_id=session_id,
                interaction_data=metadata,
                created_at=datetime.utcnow()
            )
            
            self.db.add(record)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error storing metadata: {e}")
            self.db.rollback()
    
    def get_conversation_context(self, session_id: str) -> Dict:
        """
        Get conversation context for current session
        
        Returns both recent messages (short-term) and formatted history
        """
        if session_id not in self._session_memories:
            return {"chat_history": [], "context_summary": ""}
        
        memory = self._session_memories[session_id]
        
        # Get recent messages from short-term memory
        try:
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
        except Exception:
            chat_history = []
        
        # Create context summary
        context_summary = self._create_context_summary(session_id)
        
        return {
            "chat_history": chat_history,
            "context_summary": context_summary,
            "message_count": len(chat_history)
        }
    
    def _create_context_summary(self, session_id: str) -> str:
        """Create a summary of the conversation context"""
        if session_id not in self._chat_histories:
            return ""
        
        messages = self._chat_histories[session_id].messages
        if not messages:
            return ""
        
        # Create brief summary
        user_queries = [m.content for m in messages if isinstance(m, HumanMessage)]
        
        if len(user_queries) <= 1:
            return ""
        
        summary = f"Previous topics discussed: {', '.join(user_queries[-3:][:50])}"
        return summary
    
    def get_full_history(self, session_id: str) -> List[Dict]:
        """Get full conversation history from database"""
        try:
            from backend.models.database import ConversationMessage
            
            # Load directly from database (doesn't require in-memory cache)
            messages = self.db.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id
            ).order_by(ConversationMessage.created_at).all()
            
            return [
                {
                    "role": "user" if msg.role == "human" else "assistant",
                    "content": msg.content,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None
                }
                for msg in messages
            ]
            
        except Exception as e:
            logger.error(f"Error getting full history: {e}")
            return []
    
    def get_customer_sessions(self, customer_id: int, limit: int = 5) -> List[Dict]:
        """Get recent sessions for a customer"""
        try:
            from backend.models.database import CustomerSession
            
            sessions = self.db.query(CustomerSession).filter(
                CustomerSession.customer_id == customer_id
            ).order_by(
                CustomerSession.started_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    "session_id": s.session_id,
                    "started_at": s.started_at.isoformat(),
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                    "feedback": s.feedback
                }
                for s in sessions
            ]
            
        except Exception as e:
            logger.error(f"Error getting customer sessions: {e}")
            return []
    
    def end_session(self, session_id: str, feedback: str = None):
        """End a session and optionally store feedback"""
        try:
            from backend.models.database import CustomerSession
            
            session = self.db.query(CustomerSession).filter(
                CustomerSession.session_id == session_id
            ).first()
            
            if session:
                session.ended_at = datetime.utcnow()
                if feedback:
                    session.feedback = feedback
                self.db.commit()
            
            # Clean up in-memory data
            if session_id in self._session_memories:
                del self._session_memories[session_id]
            
            logger.info(f"Ended session {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            self.db.rollback()
    
    def clear_session_memory(self, session_id: str):
        """Clear short-term memory for a session (keeps long-term history)"""
        if session_id in self._session_memories:
            self._session_memories[session_id].clear()
            logger.info(f"Cleared short-term memory for session {session_id}")
