"""
Advisor Agent - Generates personalized investment recommendations using LLM
Responsibilities: Synthesize data, risk profile, and market conditions into actionable advice
"""
from openai import OpenAI
from typing import Dict, List
from loguru import logger
from backend.config import settings
import json


class AdvisorAgent:
    """Agent responsible for generating personalized investment advice"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o-mini"  # Best balance: cheaper & better than gpt-3.5-turbo
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the advisor"""
        return """You are an expert financial advisor specializing in Indian retail investments with a conversational and agentic approach.

Your role is to:
- Understand user's financial questions in natural language
- Act as an intelligent agent that identifies missing or conflicting information
- Ask clarifying questions when critical information is missing or ambiguous
- Provide personalized, practical investment recommendations only when you have sufficient information
- Consider their financial profile, risk tolerance, and current market conditions
- Recommend Indian investment instruments (Fixed Deposits, Mutual Funds, ETFs, Gold, PPF, NPS, etc.)
- Explain complex concepts in simple terms

Key principles:
1. **ALWAYS check for conflicts AND missing information first** - If you see conflicting information (e.g., "⚠️ IMPORTANT - Please clarify") OR if critical details are missing, DO NOT provide recommendations. Instead, ask the user friendly clarifying questions.
2. **Ask, don't assume** - If critical information is missing (investment amount, duration, or goal), ask the user to provide it. Never make assumptions about these core details.
3. **Critical information required**: Investment Amount, Duration, and Financial Goal. If any of these show as 0, "Not Specified", or are clearly placeholder values, ASK for them.
4. Answer conversationally - address their specific question first
5. Provide specific portfolio allocation with percentages only when you have complete information
6. Give realistic expected returns
7. Explain reasoning in clear, simple language
8. Highlight risks and considerations
9. Suggest actionable next steps

**When information is missing:**
- Be friendly: "I'd love to help you! To provide the best investment advice, I need a few more details..."
- Ask specific questions: "Could you tell me: 1) How much are you planning to invest? 2) For how long? 3) What's your investment goal?"
- Don't proceed with generic recommendations - wait for user input

**When you detect conflicts:**
- Acknowledge the conflict politely: "I noticed you mentioned X in your question but selected Y in the form..."
- Ask which value is correct: "Could you please confirm which amount you'd like to invest?"
- Wait for clarification before proceeding with recommendations
- Be friendly and helpful, not accusatory

Be friendly, practical, and focused on Indian investment options. Act as an intelligent agent that ensures all information is clear before providing advice."""
    
    def _create_user_prompt(
        self,
        risk_profile: Dict,
        market_data: Dict,
        user_query: str,
        investment_amount: float,
        duration_months: int,
        financial_goal: str
    ) -> str:
        """Create detailed user prompt with all context"""
        
        # Check for missing critical information
        missing_info = []
        if investment_amount <= 0:
            missing_info.append("Investment Amount")
        if duration_months <= 0:
            missing_info.append("Investment Duration")
        if financial_goal == "Not Specified" or not financial_goal:
            missing_info.append("Financial Goal")
        
        prompt = f"""
USER FINANCIAL PROFILE:
- Investment Amount: {f"₹{investment_amount:,.0f}" if investment_amount > 0 else "NOT PROVIDED"}
- Investment Duration: {f"{duration_months} months ({duration_months/12:.1f} years)" if duration_months > 0 else "NOT PROVIDED"}
- Financial Goal: {financial_goal if financial_goal != "Not Specified" and financial_goal else "NOT PROVIDED"}
- Risk Category: {risk_profile['risk_category']}
- Confidence: {risk_profile.get('prediction_confidence', risk_profile.get('risk_score', 0)):.1%}
- Risk Score: {risk_profile.get('risk_score', 0):.2f}

CURRENT MARKET CONDITIONS:
- Fixed Deposit Rates (1 year): {market_data['fd_rates'].get('regular', {}).get('1_year', market_data['fd_rates'].get('1_year', 6.5)):.2f}%
- Repo Rate: {market_data['repo_rate']:.2f}%
- Inflation Rate: {market_data['inflation_rate']:.2f}%
- Debt MF Returns (avg): {market_data['mf_returns']['debt']['short_term']:.2f}%
- Equity MF Returns (avg): {market_data['mf_returns']['equity']['large_cap']:.2f}%
- Gold Price: ₹{market_data['gold_rates']['price_per_10_gram_24k']:,.0f} per 10g

USER QUERY: {user_query}

{f"⚠️ CRITICAL: The following information is MISSING: {', '.join(missing_info)}. You MUST ask the user for ONLY these missing details. Do NOT ask for information that is already provided above (marked as provided, not 'NOT PROVIDED'). Be specific about what's missing." if missing_info else ""}

{f"✅ IMPORTANT: You have all required information (Investment Amount, Duration, and Goal are all provided). Please proceed with generating a complete investment recommendation." if not missing_info else ""}

Please provide a personalized investment recommendation with:
1. Portfolio allocation (specific percentages for each asset class)
2. Expected returns (realistic annual return %)
3. Specific instrument recommendations
4. Clear reasoning
5. Risk considerations

Return your response as a valid JSON object with this structure:
{{
    "portfolio": {{"asset_class_name": percentage, ...}},
    "expected_return_annual": float,
    "risk_level": "Conservative/Moderate/Aggressive",
    "reasoning": "detailed explanation",
    "asset_allocation": {{
        "asset_class": {{
            "instrument": "specific instrument name",
            "allocation": percentage,
            "expected_return": float,
            "reason": "why this is recommended"
        }}
    }},
    "action_steps": ["step 1", "step 2", ...]
}}
"""
        return prompt
    
    def generate_recommendation(
        self,
        risk_profile: Dict,
        market_data: Dict,
        user_query: str,
        investment_amount: float,
        duration_months: int,
        financial_goal: str
    ) -> Dict:
        """
        Generate personalized investment recommendation using LLM
        """
        try:
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(
                risk_profile,
                market_data,
                user_query,
                investment_amount,
                duration_months,
                financial_goal
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            recommendation_text = response.choices[0].message.content
            
            # Check if AI is asking questions instead of providing recommendations
            is_asking_question = any(phrase in recommendation_text.lower() for phrase in [
                'could you tell me', 'could you provide', 'please clarify', 
                'i need to know', "i'd need", 'please confirm', 'can you tell',
                'which amount', 'how much', 'how long', 'what is your'
            ])
            
            if is_asking_question:
                # AI is asking for clarification - return as conversational response
                logger.info("AI is asking clarifying questions")
                return {
                    "type": "question",
                    "message": recommendation_text,
                    "portfolio": {},
                    "expected_return_annual": 0,
                    "risk_level": risk_profile.get('risk_category', 'Moderate'),
                    "reasoning": recommendation_text,
                    "action_steps": ["Please provide the requested information above"],
                    "is_question": True
                }
            
            # Try to parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                if "```json" in recommendation_text:
                    json_start = recommendation_text.find("```json") + 7
                    json_end = recommendation_text.find("```", json_start)
                    recommendation_text = recommendation_text[json_start:json_end].strip()
                elif "```" in recommendation_text:
                    json_start = recommendation_text.find("```") + 3
                    json_end = recommendation_text.find("```", json_start)
                    recommendation_text = recommendation_text[json_start:json_end].strip()
                
                recommendation = json.loads(recommendation_text)
                recommendation["is_question"] = False
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response
                recommendation = self._create_fallback_recommendation(
                    risk_profile,
                    market_data,
                    recommendation_text,
                    duration_months
                )
                recommendation["is_question"] = False
            
            logger.info("Generated investment recommendation successfully")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return self._create_fallback_recommendation(
                risk_profile,
                market_data,
                "Unable to generate detailed recommendation. Using rule-based approach.",
                duration_months
            )
    
    def _create_fallback_recommendation(
        self,
        risk_profile: Dict,
        market_data: Dict,
        reasoning: str,
        duration_months: int
    ) -> Dict:
        """
        Create rule-based recommendation when LLM fails
        """
        risk_category = risk_profile['risk_category']
        
        # Define allocations based on risk profile
        if risk_category == "Conservative":
            portfolio = {
                "Fixed Deposits": 50,
                "Debt Mutual Funds": 30,
                "Liquid Funds": 15,
                "Gold": 5
            }
            expected_return = 7.0
            asset_details = {
                "Fixed Deposits": {
                    "instrument": "Bank FD (3-year)",
                    "allocation": 50,
                    "expected_return": 7.0,
                    "reason": "Safe, guaranteed returns"
                },
                "Debt Mutual Funds": {
                    "instrument": "Short-term Debt Fund",
                    "allocation": 30,
                    "expected_return": 6.8,
                    "reason": "Better than FD with moderate safety"
                }
            }
        elif risk_category == "Moderate":
            portfolio = {
                "Debt Mutual Funds": 40,
                "Equity Mutual Funds": 35,
                "Fixed Deposits": 20,
                "Gold": 5
            }
            expected_return = 9.0
            asset_details = {
                "Debt Mutual Funds": {
                    "instrument": "Short-term Debt Fund",
                    "allocation": 40,
                    "expected_return": 7.0,
                    "reason": "Stable returns with liquidity"
                },
                "Equity Mutual Funds": {
                    "instrument": "Large-cap Index Fund",
                    "allocation": 35,
                    "expected_return": 12.0,
                    "reason": "Growth potential with managed risk"
                }
            }
        else:  # Aggressive
            portfolio = {
                "Equity Mutual Funds": 60,
                "Debt Mutual Funds": 25,
                "Gold": 10,
                "International Funds": 5
            }
            expected_return = 12.0
            asset_details = {
                "Equity Mutual Funds": {
                    "instrument": "Multi-cap Equity Fund",
                    "allocation": 60,
                    "expected_return": 14.0,
                    "reason": "High growth potential for long-term"
                },
                "Debt Mutual Funds": {
                    "instrument": "Short-term Debt Fund",
                    "allocation": 25,
                    "expected_return": 7.0,
                    "reason": "Balancing portfolio risk"
                }
            }
        
        return {
            "portfolio": portfolio,
            "expected_return_annual": expected_return,
            "risk_level": risk_category,
            "reasoning": reasoning + f"\n\nBased on {risk_category} risk profile and {duration_months} month duration.",
            "asset_allocation": asset_details,
            "action_steps": [
                "Open investment account if not already available",
                "Complete KYC verification",
                "Set up systematic investment if applicable",
                "Review portfolio quarterly"
            ]
        }
