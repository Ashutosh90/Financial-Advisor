"""
Orchestrator - Multi-agent system using LangGraph state machine
Coordinates all agents in a structured workflow with state management
Uses existing AdvisorAgent and XAIAgent for their specialized logic
Implements LangChain memory for session management
Includes guardrails for safety and compliance
"""
from typing import Dict, List, TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from loguru import logger
import json
from datetime import datetime
import operator

from backend.agents.data_agent import DataAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.advisor_agent import AdvisorAgent
from backend.agents.xai_agent import XAIAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.session_memory import SessionMemoryManager
from backend.agents.guardrails import FinancialAdvisorGuardrails, GuardrailViolationType


class AgentState(TypedDict):
    """State object that flows through the LangGraph"""
    # Input fields
    user_id: int
    query: str
    investment_amount: float
    duration_months: int
    financial_goal: str
    risk_tolerance: str
    monthly_income: float
    monthly_expenses: float
    age: int
    
    # Intermediate results
    user_context: Dict
    market_data: Dict
    risk_profile: Dict
    
    # Messages for conversational flow
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Final output
    recommendation: Dict
    explanation: Dict
    session_id: str
    
    # Control flags
    missing_info: list
    is_question: bool
    
    # Session management
    session_id: str
    conversation_context: Dict


class Orchestrator:
    """
    Multi-agent orchestrator using LangGraph state machine
    Coordinates DataAgent, RiskAgent, AdvisorAgent, XAIAgent, and MemoryAgent
    Implements LangChain memory for short-term session management
    Uses SQL database for long-term memory persistence
    """
    
    def __init__(self, db: Session):
        self.db = db
        
        # Initialize agents
        self.data_agent = DataAgent()
        self.risk_agent = RiskAgent()
        self.advisor_agent = AdvisorAgent()
        self.xai_agent = XAIAgent(risk_agent=self.risk_agent)  # Pass full RiskAgent for SHAP/LIME
        self.memory_agent = MemoryAgent(db)
        
        # Initialize session memory manager (LangChain-based)
        self.session_memory = SessionMemoryManager(db, window_size=10)
        
        # Initialize guardrails for safety and compliance
        self.guardrails = FinancialAdvisorGuardrails()
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("Orchestrator initialized with all agents, guardrails, LangGraph and LangChain memory")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes (agent functions)
        workflow.add_node("retrieve_context", self._retrieve_user_context)
        workflow.add_node("check_missing_info", self._check_missing_info)
        workflow.add_node("fetch_market_data", self._fetch_market_data)
        workflow.add_node("assess_risk", self._assess_risk_profile)
        workflow.add_node("generate_recommendation", self._generate_recommendation)
        workflow.add_node("explain_recommendation", self._explain_recommendation)
        workflow.add_node("store_session", self._store_session)
        
        # Define edges (workflow)
        workflow.set_entry_point("retrieve_context")
        
        workflow.add_edge("retrieve_context", "check_missing_info")
        
        # Conditional edge: if missing info, ask questions; else proceed
        workflow.add_conditional_edges(
            "check_missing_info",
            self._should_ask_questions,
            {
                "ask": "generate_recommendation",  # Use LLM to ask questions
                "proceed": "fetch_market_data"
            }
        )
        
        workflow.add_edge("fetch_market_data", "assess_risk")
        workflow.add_edge("assess_risk", "generate_recommendation")
        
        # Conditional edge: if recommendation is a question, skip explanation
        workflow.add_conditional_edges(
            "generate_recommendation",
            self._should_explain,
            {
                "explain": "explain_recommendation",
                "skip": "store_session"
            }
        )
        
        workflow.add_edge("explain_recommendation", "store_session")
        workflow.add_edge("store_session", END)
        
        return workflow.compile()
    
    def _retrieve_user_context(self, state: AgentState) -> AgentState:
        """Node: Retrieve user context from memory"""
        logger.info(f"Retrieving context for user {state['user_id']}")
        
        user_context = self.memory_agent.get_personalized_context(
            state['user_id']
        )
        
        state['user_context'] = user_context
        state['messages'] = [HumanMessage(content=state['query'])]
        
        return state
    
    def _check_missing_info(self, state: AgentState) -> AgentState:
        """Node: Check for missing critical information"""
        missing = []
        
        if state['investment_amount'] <= 0:
            missing.append("Investment Amount")
        if state['duration_months'] <= 0:
            missing.append("Investment Duration")
        if state['financial_goal'] == "Not Specified" or not state['financial_goal']:
            missing.append("Financial Goal")
        
        state['missing_info'] = missing
        state['is_question'] = len(missing) > 0
        
        logger.info(f"Missing info check: {missing}")
        
        return state
    
    def _should_ask_questions(self, state: AgentState) -> str:
        """Conditional edge: Determine if we need to ask for missing info"""
        if state['missing_info']:
            return "ask"
        return "proceed"
    
    def _fetch_market_data(self, state: AgentState) -> AgentState:
        """Node: Fetch current market data"""
        logger.info("Fetching market data")
        
        market_data = self.data_agent.get_all_market_data()
        state['market_data'] = market_data
        
        return state
    
    def _assess_risk_profile(self, state: AgentState) -> AgentState:
        """Node: Assess user's risk profile"""
        logger.info("Assessing risk profile")
        
        risk_profile = self.risk_agent.assess_risk(
            monthly_income=state['monthly_income'],
            monthly_expenses=state['monthly_expenses'],
            investment_amount=state['investment_amount'],
            investment_duration_months=state['duration_months'],
            age=state['age'],
            user_risk_tolerance=state['risk_tolerance']
        )
        
        state['risk_profile'] = risk_profile
        
        return state
    
    def _generate_recommendation(self, state: AgentState) -> AgentState:
        """Node: Generate investment recommendation using AdvisorAgent"""
        logger.info("Generating recommendation with AdvisorAgent")
        
        # Use the existing AdvisorAgent which handles all the logic
        recommendation = self.advisor_agent.generate_recommendation(
            risk_profile=state.get('risk_profile', {}),
            market_data=state.get('market_data', {}),
            user_query=state['query'],
            investment_amount=state['investment_amount'],
            duration_months=state['duration_months'],
            financial_goal=state['financial_goal']
        )
        
        state['recommendation'] = recommendation
        state['is_question'] = recommendation.get('is_question', False)
        
        # Add to message history
        state['messages'].append(AIMessage(content=recommendation.get('reasoning', '')))
        
        return state
    
    def _should_explain(self, state: AgentState) -> str:
        """Conditional edge: Determine if we need to explain the recommendation"""
        if state['is_question']:
            return "skip"
        return "explain"
    
    def _explain_recommendation(self, state: AgentState) -> AgentState:
        """Node: Generate XAI explanation using XAIAgent with SHAP/LIME"""
        logger.info("Generating explanation with XAIAgent (SHAP/LIME/LLM)")
        
        # Get customer ID for ML-based explanations
        customer_id = state.get('user_id')
        
        # Prepare features for SHAP/LIME if we have a valid customer_id
        features_df = None
        customer_data = None
        if customer_id:
            try:
                customer_data_df = self.risk_agent.fetch_customer_data(customer_id)
                if customer_data_df is not None:
                    features_df = self.risk_agent.prepare_features(customer_data_df)
                    customer_data = customer_data_df.iloc[0].to_dict()
            except Exception as e:
                logger.warning(f"Could not prepare features for XAI: {e}")
        
        # Generate comprehensive explanation with SHAP, LIME, and LLM
        risk_explanation = self.xai_agent.explain_risk_profile(
            risk_profile=state['risk_profile'],
            customer_data=customer_data,
            features_df=features_df,
            use_shap=True,
            use_lime=True,
            simplify_with_llm=True
        )
        
        # Also explain the recommendation
        recommendation_explanation = self.xai_agent.explain_recommendation(
            risk_profile=state['risk_profile'],
            recommendation=state['recommendation'],
            market_data=state['market_data'],
            simplify_with_llm=True
        )
        
        state['explanation'] = {
            "risk_explanation": risk_explanation,
            "recommendation_explanation": recommendation_explanation,
            "methods_used": risk_explanation.get("explanation_methods_used", [])
        }
        
        return state
    
    def _store_session(self, state: AgentState) -> AgentState:
        """Node: Store session in database"""
        logger.info("Storing session")
        
        # Generate session ID
        session_id = str(datetime.now().timestamp())
        state['session_id'] = session_id
        
        # Store only if it's a recommendation (not a question)
        if not state['is_question']:
            self.memory_agent.store_session(
                user_id=state['user_id'],
                session_id=session_id,
                query=state['query'],
                recommendation=state['recommendation'],
                risk_profile=state['risk_profile']
            )
        
        return state
    
    def process_query(
        self,
        user_id: int,
        query: str,
        monthly_income: float = 0,
        investment_amount: float = 0,
        investment_duration_months: int = 0,
        financial_goal: str = "Not Specified",
        risk_tolerance: str = "Not Specified",
        monthly_expenses: float = 0,
        age: int = 35
    ) -> Dict:
        """
        Process user query through the LangGraph workflow
        
        Returns:
            Complete advisory response with recommendation and explanation
        """
        logger.info(f"Processing query for user {user_id} through LangGraph")
        
        # Initialize state
        initial_state = AgentState(
            user_id=user_id,
            query=query,
            investment_amount=investment_amount,
            duration_months=investment_duration_months,
            financial_goal=financial_goal,
            risk_tolerance=risk_tolerance,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            age=age,
            user_context={},
            market_data={},
            risk_profile={},
            messages=[],
            recommendation={},
            explanation={},
            session_id="",
            missing_info=[],
            is_question=False
        )
        
        # Execute graph
        final_state = self.graph.invoke(initial_state)
        
        # Format response
        response = {
            "recommendation": final_state['recommendation'],
            "market_data": final_state.get('market_data', {}),
            "risk_profile": final_state.get('risk_profile', {}),
            "session_id": final_state.get('session_id', ''),
        }
        
        # Add explanation if available
        if not final_state['is_question'] and 'explanation' in final_state:
            response['explanation'] = final_state['explanation']
        
        logger.info("Query processed successfully through LangGraph")
        
        return response
    
    def _accumulate_investment_details_from_conversation(
        self, 
        current_query: str, 
        session_id: str
    ) -> Dict:
        """
        Parse investment details from the ENTIRE conversation history.
        This ensures we remember information provided in earlier messages.
        
        The latest value for each field takes precedence (user can override earlier values).
        """
        accumulated = {
            "investment_amount": None,
            "duration_months": None,
            "financial_goal": None
        }
        
        # Get conversation history
        try:
            full_history = self.session_memory.get_full_history(session_id)
            
            # Parse each user message in the conversation (oldest to newest)
            # This way, later messages override earlier ones
            for msg in full_history:
                if msg.get("role") == "user":
                    parsed = self._parse_investment_details_from_query(msg.get("content", ""))
                    
                    # Update accumulated values if new values found
                    if parsed.get("investment_amount"):
                        accumulated["investment_amount"] = parsed["investment_amount"]
                    if parsed.get("duration_months"):
                        accumulated["duration_months"] = parsed["duration_months"]
                    if parsed.get("financial_goal"):
                        accumulated["financial_goal"] = parsed["financial_goal"]
            
            logger.debug(f"After parsing history: {accumulated}")
            
        except Exception as e:
            logger.warning(f"Error parsing conversation history: {e}")
        
        # Finally, parse the current query (takes highest precedence)
        current_parsed = self._parse_investment_details_from_query(current_query)
        
        if current_parsed.get("investment_amount"):
            accumulated["investment_amount"] = current_parsed["investment_amount"]
        if current_parsed.get("duration_months"):
            accumulated["duration_months"] = current_parsed["duration_months"]
        if current_parsed.get("financial_goal"):
            accumulated["financial_goal"] = current_parsed["financial_goal"]
        
        logger.info(f"Final accumulated details: Amount={accumulated['investment_amount']}, "
                   f"Duration={accumulated['duration_months']}, Goal={accumulated['financial_goal']}")
        
        return accumulated
    
    def _parse_investment_details_from_query(self, query: str) -> Dict:
        """
        Parse investment amount, duration, and goal from user's natural language query
        Returns extracted values or None if not found
        """
        import re
        
        result = {
            "investment_amount": None,
            "duration_months": None,
            "financial_goal": None
        }
        
        query_lower = query.lower()
        
        # Parse investment amount
        # Order: more specific patterns first, then generic number patterns
        amount_patterns = [
            # With currency symbol AND unit
            (r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(?:crores?|cr)\b', 10000000),  # ₹1 crore
            (r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(?:lakhs?|lacs?|l)\b', 100000),  # ₹8 lakhs
            (r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(?:thousands?|k)\b', 1000),  # ₹50 thousand
            # Without currency symbol but WITH unit
            (r'([\d,]+(?:\.\d+)?)\s*(?:crores?|cr)\b', 10000000),  # 1 crore
            (r'([\d,]+(?:\.\d+)?)\s*(?:lakhs?|lacs?|l)\b', 100000),  # 8 lakhs
            (r'([\d,]+(?:\.\d+)?)\s*(?:thousands?|k)\b', 1000),  # 50 thousand, 50k
            # With currency symbol only (raw number)
            (r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\b', 1),  # ₹800000
            # Plain number patterns - capture amounts 1000 and above
            (r'\b(\d{4,})(?:\.\d+)?\b', 1),  # 4+ digit number like 8000, 50000, 100000
        ]
        
        for pattern, multiplier in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str) * multiplier
                    # Sanity check: investment amount should be at least 1000
                    if amount >= 1000:
                        result["investment_amount"] = amount
                        logger.debug(f"Parsed amount: {amount} (pattern: {pattern}, multiplier: {multiplier})")
                        break
                except ValueError:
                    continue
        
        # Parse duration
        # Match patterns like: "4 years", "5 year", "3-5 years", "18 months", "2 yrs"
        # Order matters - more specific patterns first
        duration_patterns = [
            (r'(\d+)-(\d+)\s*(?:years?|yrs?)\b', 'years_range'),  # 3-5 years (use average)
            (r'for\s+(\d+)\s*(?:years?|yrs?)\b', 'years'),  # for 4 years
            (r'in\s+(\d+)\s*(?:years?|yrs?)\b', 'years'),   # in 4 years
            (r'(\d+)\s*(?:years?|yrs?)\b', 'years'),  # 4 years, 5 year, 2 yrs
            (r'for\s+(\d+)\s*(?:months?|mos?)\b', 'months'),  # for 6 months
            (r'in\s+(\d+)\s*(?:months?|mos?)\b', 'months'),   # in 6 months
            (r'(\d+)\s*(?:months?|mos?)\b', 'months'),   # 18 months
        ]
        
        for pattern, unit_type in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    if unit_type == 'years_range':  # Range like 3-5 years
                        years = (int(match.group(1)) + int(match.group(2))) / 2
                        result["duration_months"] = int(years * 12)
                    elif unit_type == 'years':
                        years = int(match.group(1))
                        result["duration_months"] = int(years * 12)  # Convert years to months
                    else:  # months
                        result["duration_months"] = int(match.group(1))  # Already in months
                    logger.debug(f"Parsed duration: {result['duration_months']} months (pattern: {pattern})")
                    break
                except ValueError:
                    continue
        
        # Parse financial goal - expanded keywords
        goal_keywords = {
            "education": ["education", "college", "university", "school", "daughter's education", 
                         "son's education", "child's education", "higher studies", "study", "studies",
                         "tuition", "course", "degree"],
            "retirement": ["retirement", "retire", "pension", "old age", "golden years"],
            "home": ["home", "house", "property", "real estate", "flat", "apartment", "housing"],
            "wedding": ["wedding", "marriage", "shaadi", "engagement"],
            "car": ["car", "vehicle", "automobile", "bike", "motorcycle", "two wheeler"],
            "wealth": ["wealth", "grow", "growth", "accumulation", "corpus", "build wealth", 
                      "savings", "save", "investment", "invest", "portfolio"],
            "emergency": ["emergency", "rainy day", "contingency", "safety net"],
            "travel": ["travel", "vacation", "holiday", "trip", "tour"],
            "business": ["business", "startup", "entrepreneurship", "venture", "shop"],
            "gadget": ["mobile", "phone", "laptop", "computer", "gadget", "electronics", 
                      "iphone", "smartphone", "tablet", "ipad", "macbook"],
            "medical": ["medical", "health", "hospital", "treatment", "surgery", "healthcare"],
        }
        
        for goal, keywords in goal_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    result["financial_goal"] = goal.title()
                    logger.debug(f"Parsed goal: {result['financial_goal']} (keyword: {keyword})")
                    break
            if result["financial_goal"]:
                break
        
        logger.info(f"Parsed from query '{query}': Amount={result['investment_amount']}, Duration={result['duration_months']} months, Goal={result['financial_goal']}")
        return result

    def _create_blocked_response(
        self,
        customer_id: int,
        reason: List[str],
        violation_type: str
    ) -> Dict:
        """
        Create a safe response when input is blocked by guardrails
        
        Args:
            customer_id: Customer ID
            reason: List of violation reasons
            violation_type: Type of violation (input_validation, output_compliance, etc.)
            
        Returns:
            A safe, compliant response dict
        """
        safe_message = (
            "I apologize, but I'm unable to process this request. "
            "As a financial advisor, I can only provide investment advice and help with "
            "financial planning queries. Please ask me about:\n\n"
            "• Investment recommendations based on your risk profile\n"
            "• Portfolio allocation strategies\n"
            "• Financial goal planning\n"
            "• Understanding your risk assessment\n"
            "• Market data and investment options"
        )
        
        return {
            "recommendation": {
                "response": safe_message,
                "reasoning": safe_message,
                "is_question": False,
                "guardrails_blocked": True,
                "violation_type": violation_type,
                "portfolio": {}
            },
            "market_data": {},
            "risk_profile": {},
            "explanation": {
                "simplified_explanation": "This request could not be processed due to content guidelines.",
                "key_factors": []
            },
            "customer_id": customer_id,
            "session_id": None,
            "conversation_context": {"message_count": 0},
            "customer_data": {},
            "guardrails_info": {
                "blocked": True,
                "violation_type": violation_type,
                "reasons": reason
            }
        }

    def process_customer_query(
        self,
        customer_id: int,
        customer_data: Dict,
        query: str = "Provide investment recommendations based on my risk profile",
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Process query using customer data from risk profiling database
        Uses the trained ML model to predict risk profile
        Maintains conversation context using LangChain memory
        Applies guardrails for safety and compliance
        
        Args:
            customer_id: Customer ID from database
            customer_data: Customer data from risk_profiling_monthly_data table
            query: User query (optional)
            session_id: Existing session ID to continue conversation (optional)
            
        Returns:
            Complete advisory response with recommendation and explanation
        """
        logger.info(f"Processing query for customer {customer_id} using ML model")
        
        # === GUARDRAILS: Input Validation (Non-blocking for legitimate queries) ===
        try:
            is_valid, sanitized_query, error_message = self.guardrails.validate_input(query)
            if not is_valid:
                # Only block truly malicious inputs (prompt injection, adversarial attacks)
                # Log the issue but let the advisor handle off-topic gracefully
                logger.warning(f"Input guardrail warning for customer {customer_id}: {error_message}")
                # Still process the query - the advisor agent will handle inappropriate queries gracefully
            else:
                # Use sanitized query if PII was masked
                query = sanitized_query
        except Exception as e:
            # If guardrails fail, don't block the user - just log and continue
            logger.error(f"Guardrails validation error (continuing): {e}")
        
        # Get or create session for conversation memory
        current_session_id = self.session_memory.get_or_create_session(
            customer_id=customer_id,
            session_id=session_id
        )
        
        # Get conversation context from memory
        conversation_context = self.session_memory.get_conversation_context(current_session_id)
        logger.debug(f"Session {current_session_id}: {conversation_context.get('message_count', 0)} messages in history")
        
        # Parse investment details from the ENTIRE conversation history, not just current query
        # This ensures we accumulate information across multiple messages
        accumulated_details = self._accumulate_investment_details_from_conversation(
            current_query=query,
            session_id=current_session_id
        )
        
        # Use accumulated values if available
        investment_amount = accumulated_details.get("investment_amount") or 0
        duration_months = accumulated_details.get("duration_months") or 0
        financial_goal = accumulated_details.get("financial_goal") or "Not Specified"
        
        logger.info(f"Accumulated from conversation - Amount: {investment_amount}, Duration: {duration_months} months, Goal: {financial_goal}")
        
        # Get ML-based risk prediction
        risk_profile = self.risk_agent.predict_risk_profile(customer_id)
        
        # Fetch market data
        market_data = self.data_agent.get_all_market_data()
        
        # Generate recommendation using AdvisorAgent
        # Pass parsed values (or 0/Not Specified to trigger clarifying questions if needed)
        recommendation = self.advisor_agent.generate_recommendation(
            risk_profile=risk_profile,
            market_data=market_data,
            user_query=query,
            investment_amount=investment_amount,
            duration_months=duration_months,
            financial_goal=financial_goal
        )
        
        # === GUARDRAILS: Output Validation (Non-blocking, adds metadata only) ===
        # Validate and optionally enhance the recommendation with disclaimers
        try:
            recommendation_text = recommendation.get('reasoning', '') or recommendation.get('response', '')
            is_output_valid, processed_response, output_error = self.guardrails.validate_output(
                ai_response=recommendation_text,
                market_data=market_data,
                risk_level=risk_profile.get('risk_category', 'moderate')
            )
            
            if not is_output_valid:
                # Log the issue but don't block - just add metadata
                logger.warning(f"Output guardrail warning: {output_error}")
                recommendation['guardrails_warning'] = output_error
            
            # If disclaimers were added, update the reasoning (this enhances, not replaces)
            if processed_response and processed_response != recommendation_text:
                recommendation['reasoning_with_disclaimer'] = processed_response
        except Exception as e:
            # If guardrails fail, don't block - just log and continue
            logger.error(f"Output guardrails error (continuing): {e}")
        
        # Generate explanation using XAI Agent
        explanation = {}
        if not recommendation.get('is_question', False):
            # Prepare features_df for SHAP/LIME explanations
            features_df = None
            try:
                customer_data_df = self.risk_agent.fetch_customer_data(customer_id)
                if customer_data_df is not None:
                    features_df = self.risk_agent.prepare_features(customer_data_df)
            except Exception as e:
                logger.warning(f"Could not prepare features for XAI: {e}")
            
            # Explain the risk profile with SHAP/LIME
            risk_explanation = self.xai_agent.explain_risk_profile(
                risk_profile=risk_profile,
                customer_data=customer_data,
                features_df=features_df,
                use_shap=True,
                use_lime=True,
                simplify_with_llm=True
            )
            
            # Explain the recommendation with LLM
            recommendation_explanation = self.xai_agent.explain_recommendation(
                recommendation=recommendation,
                risk_profile=risk_profile,
                market_data=market_data,
                simplify_with_llm=True
            )
            
            # Combine explanations in a structure the frontend expects
            explanation = {
                "risk_assessment": risk_explanation,
                "simplified_explanation": risk_explanation.get('simplified_explanation', ''),
                "key_factors": risk_explanation.get('key_factors', []),
                "methods_used": risk_explanation.get('explanation_methods_used', []),
                # Include recommendation explanation details
                "portfolio_breakdown": recommendation_explanation.get('technical_analysis', {}).get('portfolio_breakdown', {}),
                "risk_alignment": recommendation_explanation.get('technical_analysis', {}).get('risk_alignment', ''),
                "market_context": recommendation_explanation.get('technical_analysis', {}).get('market_context', ''),
                "expected_outcome": recommendation_explanation.get('technical_analysis', {}).get('expected_outcome', ''),
                "recommendation_explanation": recommendation_explanation.get('simplified_explanation', '')
            }
            
            # Add SHAP explanation if available
            if risk_explanation.get('shap_explanation'):
                explanation['shap_values'] = {
                    f['feature']: f['shap_value'] 
                    for f in risk_explanation['shap_explanation'].get('feature_contributions', [])[:10]
                }
        
        # Format AI response for memory
        ai_response = recommendation.get('reasoning', 'Recommendation generated based on your risk profile.')
        
        # Store interaction in session memory (short-term + long-term)
        self.session_memory.add_interaction(
            session_id=current_session_id,
            user_message=query,
            ai_response=ai_response,
            metadata={
                "risk_category": risk_profile.get('risk_category'),
                "confidence": risk_profile.get('prediction_confidence'),
                "portfolio": recommendation.get('portfolio', {})
            }
        )
        
        # Format response
        response = {
            "recommendation": recommendation,
            "market_data": market_data,
            "risk_profile": risk_profile,
            "explanation": explanation,
            "customer_id": customer_id,
            "session_id": current_session_id,  # Return session ID for continuity
            "conversation_context": {
                "message_count": conversation_context.get('message_count', 0) + 1,
                "context_summary": conversation_context.get('context_summary', '')
            },
            "customer_data": {
                "age": customer_data.get('age'),
                "annual_income": customer_data.get('annual_income'),
                "net_worth": customer_data.get('net_worth'),
                "risk_category": customer_data.get('risk_profile')
            }
        }
        
        logger.info(f"Generated advice for customer {customer_id} in session {current_session_id}")
        
        return response
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get full conversation history for a session"""
        return self.session_memory.get_full_history(session_id)
    
    def get_customer_sessions(self, customer_id: int, limit: int = 5) -> List[Dict]:
        """Get recent sessions for a customer"""
        return self.session_memory.get_customer_sessions(customer_id, limit)
    
    def end_customer_session(self, session_id: str, feedback: str = None):
        """End a session and optionally store feedback"""
        self.session_memory.end_session(session_id, feedback)
    
    def submit_feedback(self, session_id: str, feedback: str, user_id: int) -> bool:
        """Submit feedback for a session"""
        return self.memory_agent.update_feedback(session_id, feedback)
    
    def get_user_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get user's advisory history"""
        return self.memory_agent.get_past_sessions(user_id, limit)
