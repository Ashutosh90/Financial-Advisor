"""
Guardrails Module - Safety and Compliance Layer for Financial Advisory System

Implements comprehensive guardrails:
1. Input Validation - Detect adversarial prompts, prompt injections
2. Output Filtering - Ensure compliant financial advice
3. Hallucination Detection - Verify factual accuracy of responses
4. PII Protection - Detect and mask sensitive personal information

Based on NeMo Guardrails concepts with custom implementation for financial domain.
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json


class GuardrailViolationType(Enum):
    """Types of guardrail violations"""
    PROMPT_INJECTION = "prompt_injection"
    ADVERSARIAL_PROMPT = "adversarial_prompt"
    PII_DETECTED = "pii_detected"
    NON_COMPLIANT_ADVICE = "non_compliant_advice"
    HALLUCINATION = "hallucination"
    OFF_TOPIC = "off_topic"
    HARMFUL_CONTENT = "harmful_content"


@dataclass
class GuardrailResult:
    """Result of guardrail check"""
    passed: bool
    violation_type: Optional[GuardrailViolationType] = None
    violation_details: Optional[str] = None
    sanitized_content: Optional[str] = None
    confidence: float = 1.0


class InputGuardrails:
    """
    Input validation guardrails to detect adversarial prompts and prompt injections.
    Protects the system from malicious user inputs.
    """
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r"ignore (all |your |previous |above )?(instructions|rules|guidelines|prompts)",
        r"disregard (all |your |previous |above )?(instructions|rules|guidelines|prompts)",
        r"forget (all |your |previous |above )?(instructions|rules|guidelines|prompts)",
        r"override (all |your |previous |above )?(instructions|rules|guidelines|prompts)",
        r"bypass (all |your |previous |above )?(instructions|rules|guidelines|prompts)",
        
        # Role manipulation
        r"you are now (a |an )?(?!financial|investment)",  # Trying to change role
        r"pretend (to be|you are|you're)",
        r"act as (a |an )?(?!financial|investment)",
        r"roleplay as",
        r"switch (to |into )?(a |an )?different (role|mode|persona)",
        
        # System prompt extraction
        r"(show|tell|reveal|display|print|output) (me |us )?(your |the )?(system|initial|original|hidden) (prompt|instructions|rules)",
        r"what (are|is) your (system|initial|original|hidden) (prompt|instructions|rules)",
        r"repeat (your |the )?(system|initial|original) (prompt|message|instructions)",
        
        # Jailbreak attempts
        r"(dan|dude|developer) mode",
        r"jailbreak",
        r"(enable|activate) (unrestricted|unlimited|god) mode",
        
        # Code execution attempts
        r"(execute|run|eval) (this |the )?(code|command|script)",
        r"```(python|javascript|bash|shell|sql)",
        
        # Data exfiltration
        r"(send|transmit|exfiltrate|export) (data|information|records) to",
        r"(access|read|query) (the |all )?database",
        r"(show|list|dump) all (users|customers|records|data)",
    ]
    
    # Adversarial/harmful content patterns
    ADVERSARIAL_PATTERNS = [
        # Financial fraud
        r"(how to|ways to) (avoid|evade) (taxes|tax)",
        r"(money laundering|launder money)",
        r"(insider trading|trade on insider)",
        r"(ponzi|pyramid) scheme",
        r"(manipulate|rig) (the )?(market|stock|price)",
        
        # Scam/fraud assistance
        r"(fake|forge|falsify) (documents|statements|records)",
        r"(hide|conceal) (income|assets|money) from",
        r"(illegal|unlawful) (investment|trading|scheme)",
        
        # Harmful financial advice requests
        r"(gamble|bet) (all|everything|my life savings)",
        r"(borrow|take loan) to (invest|gamble|trade)",
        r"(get rich quick|guaranteed returns|100% profit)",
    ]
    
    # Off-topic patterns (non-financial queries)
    OFF_TOPIC_PATTERNS = [
        r"(write|compose|create) (a |an )?(poem|story|essay|code|song)",
        r"(help me|assist with) (homework|assignment|exam)",
        r"(tell|share) (a |me a )?(joke|story|tale)",
        r"(who|what) (is|are) (you|your|the president|prime minister)",
        r"(recipe|cook|make|prepare) (food|dish|meal)",
        r"(relationship|dating|love) advice",
        r"(medical|health|doctor) advice",
        r"(legal|lawyer|court) advice",
    ]
    
    @classmethod
    def validate_input(cls, user_input: str) -> GuardrailResult:
        """
        Validate user input for potential threats.
        
        Args:
            user_input: The raw user query
            
        Returns:
            GuardrailResult with validation status
        """
        if not user_input or not user_input.strip():
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailViolationType.ADVERSARIAL_PROMPT,
                violation_details="Empty input provided"
            )
        
        input_lower = user_input.lower()
        
        # Check for prompt injection
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                logger.warning(f"Prompt injection detected: {pattern}")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.PROMPT_INJECTION,
                    violation_details=f"Detected potential prompt injection attempt",
                    confidence=0.9
                )
        
        # Check for adversarial content
        for pattern in cls.ADVERSARIAL_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                logger.warning(f"Adversarial prompt detected: {pattern}")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.ADVERSARIAL_PROMPT,
                    violation_details="Request involves potentially illegal or harmful financial activities",
                    confidence=0.85
                )
        
        # Check for off-topic queries (warning only, don't block)
        for pattern in cls.OFF_TOPIC_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                logger.info(f"Off-topic query detected: {pattern}")
                return GuardrailResult(
                    passed=True,  # Allow but flag
                    violation_type=GuardrailViolationType.OFF_TOPIC,
                    violation_details="Query appears to be off-topic for financial advice",
                    confidence=0.7
                )
        
        return GuardrailResult(passed=True, confidence=1.0)


class PIIGuardrails:
    """
    PII (Personally Identifiable Information) protection.
    Detects and masks sensitive information in inputs and outputs.
    """
    
    # PII patterns with named groups for detection
    PII_PATTERNS = {
        "aadhaar": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Aadhaar: 1234 5678 9012
        "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",  # PAN: ABCDE1234F
        "phone": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",  # Indian phone
        "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "bank_account": r"\b\d{9,18}\b",  # Bank account numbers (9-18 digits)
        "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",  # IFSC code
        "credit_card": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",  # Credit card
        "passport": r"\b[A-Z]\d{7}\b",  # Indian passport
        "voter_id": r"\b[A-Z]{3}\d{7}\b",  # Voter ID
    }
    
    # Masking templates
    MASK_TEMPLATES = {
        "aadhaar": "XXXX-XXXX-XXXX",
        "pan": "XXXXX0000X",
        "phone": "+91-XXXXXX0000",
        "email": "***@***.***",
        "bank_account": "XXXXXX0000",
        "ifsc": "XXXX0XXXXXX",
        "credit_card": "XXXX-XXXX-XXXX-XXXX",
        "passport": "X0000000",
        "voter_id": "XXX0000000",
    }
    
    @classmethod
    def detect_pii(cls, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of detected PII with type and position
        """
        detected = []
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return detected
    
    @classmethod
    def mask_pii(cls, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask all PII in text.
        
        Args:
            text: Text containing potential PII
            
        Returns:
            Tuple of (masked_text, list of masked items)
        """
        masked_items = []
        result = text
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            
            # Process in reverse order to maintain positions
            for match in reversed(matches):
                original = match.group()
                mask = cls.MASK_TEMPLATES.get(pii_type, "***MASKED***")
                
                masked_items.append({
                    "type": pii_type,
                    "original_length": len(original),
                    "masked_with": mask
                })
                
                result = result[:match.start()] + mask + result[match.end():]
        
        return result, masked_items
    
    @classmethod
    def validate_no_pii(cls, text: str) -> GuardrailResult:
        """
        Validate that text doesn't contain sensitive PII.
        
        Args:
            text: Text to validate
            
        Returns:
            GuardrailResult with validation status
        """
        detected = cls.detect_pii(text)
        
        # Filter out common false positives (e.g., amounts that look like account numbers)
        sensitive_pii = [d for d in detected if d["type"] in ["aadhaar", "pan", "credit_card", "passport"]]
        
        if sensitive_pii:
            masked_text, _ = cls.mask_pii(text)
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailViolationType.PII_DETECTED,
                violation_details=f"Detected {len(sensitive_pii)} sensitive PII item(s): {[d['type'] for d in sensitive_pii]}",
                sanitized_content=masked_text,
                confidence=0.9
            )
        
        return GuardrailResult(passed=True)


class OutputGuardrails:
    """
    Output filtering guardrails for compliant financial advice.
    Ensures AI responses meet regulatory and safety standards.
    """
    
    # Non-compliant advice patterns (SEBI/RBI regulations)
    NON_COMPLIANT_PATTERNS = [
        # Guaranteed returns claims
        (r"guaranteed? (returns?|profit|income)", "Claims of guaranteed returns are not permitted"),
        (r"100%\s*(safe|secure|guaranteed)", "No investment is 100% guaranteed"),
        (r"(risk[- ]?free|no[- ]?risk)\s+(investment|returns)", "All investments carry some risk"),
        
        # Unrealistic promises
        (r"(double|triple|10x|100x)\s+(your\s+)?money", "Unrealistic return promises"),
        (r"(get rich|become millionaire)\s+(quick|fast|overnight)", "Get rich quick schemes"),
        
        # Specific stock recommendations without disclaimer
        (r"(buy|sell|invest in)\s+[A-Z]{2,10}\s+(stock|shares?)", "Individual stock recommendations require disclaimers"),
        
        # Tax evasion (not avoidance)
        (r"(evade|hide|conceal)\s+(tax|income)", "Tax evasion advice is illegal"),
    ]
    
    # Required disclaimers for financial advice
    FINANCIAL_DISCLAIMERS = [
        "past performance does not guarantee future results",
        "investments are subject to market risks",
        "consult a certified financial advisor",
        "this is for educational purposes",
    ]
    
    # Harmful content patterns
    HARMFUL_PATTERNS = [
        r"(suicide|self[- ]?harm|kill yourself)",
        r"(hate|violence|discriminat)",
        r"(terrorist|terrorism|extremis)",
    ]
    
    @classmethod
    def validate_output(cls, ai_response: str) -> GuardrailResult:
        """
        Validate AI output for compliance and safety.
        
        Args:
            ai_response: The AI-generated response
            
        Returns:
            GuardrailResult with validation status
        """
        if not ai_response:
            return GuardrailResult(passed=True)
        
        response_lower = ai_response.lower()
        
        # Check for harmful content first
        for pattern in cls.HARMFUL_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.error(f"Harmful content detected in output")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.HARMFUL_CONTENT,
                    violation_details="Response contains potentially harmful content",
                    confidence=0.95
                )
        
        # Check for non-compliant financial advice
        for pattern, reason in cls.NON_COMPLIANT_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                logger.warning(f"Non-compliant advice detected: {reason}")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.NON_COMPLIANT_ADVICE,
                    violation_details=reason,
                    confidence=0.8
                )
        
        return GuardrailResult(passed=True)
    
    @classmethod
    def add_disclaimers(cls, response: str, risk_level: str = "moderate") -> str:
        """
        Add appropriate disclaimers to financial advice.
        
        Args:
            response: The AI response
            risk_level: Risk level of the advice (conservative, moderate, aggressive)
            
        Returns:
            Response with disclaimers added
        """
        # Check if disclaimers already exist
        response_lower = response.lower()
        has_disclaimer = any(d in response_lower for d in cls.FINANCIAL_DISCLAIMERS)
        
        if not has_disclaimer:
            disclaimer = "\n\n---\n**Disclaimer:** "
            
            if risk_level == "aggressive":
                disclaimer += "This investment advice involves higher risk assets. "
            
            disclaimer += "Investments are subject to market risks. Past performance does not guarantee future results. Please consult a SEBI-registered financial advisor before making investment decisions."
            
            response += disclaimer
        
        return response


class HallucinationGuardrails:
    """
    Hallucination detection for financial advice.
    Verifies factual accuracy of AI responses.
    """
    
    # Current market data bounds (approximate, should be updated periodically)
    MARKET_BOUNDS = {
        "fd_rate": (3.0, 9.5),  # FD rates typically 3-9.5%
        "repo_rate": (3.0, 8.0),  # RBI repo rate range
        "inflation": (2.0, 12.0),  # Inflation typically 2-12%
        "equity_returns": (-30, 50),  # Annual equity returns
        "debt_returns": (4.0, 12.0),  # Debt fund returns
        "gold_price": (40000, 80000),  # Gold per 10g in INR
        "nifty": (15000, 30000),  # NIFTY 50 range
    }
    
    # Factual claims that need verification
    VERIFIABLE_PATTERNS = [
        (r"(current|today'?s?)\s+(fd|fixed deposit)\s+rate\s*[:\s]+(\d+\.?\d*)\s*%", "fd_rate"),
        (r"repo\s+rate\s*[:\s]+(\d+\.?\d*)\s*%", "repo_rate"),
        (r"inflation\s*[:\s]+(\d+\.?\d*)\s*%", "inflation"),
        (r"nifty\s*[:\s]+(\d+[,\d]*)", "nifty"),
        (r"gold\s+price\s*[:\s]+[₹rs\s]*(\d+[,\d]*)", "gold_price"),
    ]
    
    @classmethod
    def detect_hallucinations(
        cls,
        response: str,
        actual_market_data: Optional[Dict] = None
    ) -> GuardrailResult:
        """
        Detect potential hallucinations in the response.
        
        Args:
            response: AI response to check
            actual_market_data: Actual market data for comparison
            
        Returns:
            GuardrailResult with detection status
        """
        issues = []
        
        response_lower = response.lower()
        
        # Check for claims with verifiable data
        for pattern, data_type in cls.VERIFIABLE_PATTERNS:
            matches = re.finditer(pattern, response_lower)
            for match in matches:
                try:
                    # Extract the claimed value
                    value_str = match.group(len(match.groups()))
                    value = float(value_str.replace(",", ""))
                    
                    # Check against bounds
                    if data_type in cls.MARKET_BOUNDS:
                        min_val, max_val = cls.MARKET_BOUNDS[data_type]
                        if value < min_val or value > max_val:
                            issues.append(f"{data_type}: claimed {value}, expected {min_val}-{max_val}")
                    
                    # Check against actual data if provided
                    if actual_market_data:
                        actual = cls._get_actual_value(actual_market_data, data_type)
                        if actual and abs(value - actual) / actual > 0.2:  # 20% tolerance
                            issues.append(f"{data_type}: claimed {value}, actual {actual}")
                            
                except (ValueError, IndexError):
                    continue
        
        if issues:
            logger.warning(f"Potential hallucinations detected: {issues}")
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailViolationType.HALLUCINATION,
                violation_details=f"Factual inconsistencies: {'; '.join(issues)}",
                confidence=0.75
            )
        
        return GuardrailResult(passed=True)
    
    @classmethod
    def _get_actual_value(cls, market_data: Dict, data_type: str) -> Optional[float]:
        """Extract actual value from market data"""
        try:
            if data_type == "fd_rate":
                return market_data.get("fd_rates", {}).get("regular", {}).get("1_year", 
                       market_data.get("fd_rates", {}).get("1_year"))
            elif data_type == "repo_rate":
                return market_data.get("repo_rate")
            elif data_type == "inflation":
                return market_data.get("inflation_rate")
            elif data_type == "gold_price":
                return market_data.get("gold_rates", {}).get("price_per_10_gram_24k")
            elif data_type == "nifty":
                return market_data.get("nifty_data", {}).get("current_price")
        except Exception:
            pass
        return None


class FinancialAdvisorGuardrails:
    """
    Main guardrails class that orchestrates all guardrail checks.
    Entry point for the guardrails system.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize guardrails.
        
        Args:
            strict_mode: If True, block on warnings; if False, allow with warnings
        """
        self.strict_mode = strict_mode
        self.input_guardrails = InputGuardrails()
        self.pii_guardrails = PIIGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.hallucination_guardrails = HallucinationGuardrails()
        
        logger.info(f"Financial Advisor Guardrails initialized (strict_mode={strict_mode})")
    
    def validate_input(self, user_query: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate user input before processing.
        
        Args:
            user_query: Raw user query
            
        Returns:
            Tuple of (is_valid, sanitized_query, error_message)
        """
        # Check for adversarial prompts
        input_result = InputGuardrails.validate_input(user_query)
        if not input_result.passed:
            if input_result.violation_type == GuardrailViolationType.OFF_TOPIC and not self.strict_mode:
                # Allow off-topic with warning
                logger.info("Off-topic query allowed with warning")
            else:
                error_msg = self._get_safe_error_message(input_result)
                return False, user_query, error_msg
        
        # Check for PII and mask if found
        pii_result = PIIGuardrails.validate_no_pii(user_query)
        if not pii_result.passed:
            # Mask PII and continue
            sanitized = pii_result.sanitized_content or user_query
            logger.warning(f"PII detected and masked: {pii_result.violation_details}")
            return True, sanitized, None
        
        return True, user_query, None
    
    def validate_output(
        self,
        ai_response: str,
        market_data: Optional[Dict] = None,
        risk_level: str = "moderate"
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate AI output before sending to user.
        
        Args:
            ai_response: AI-generated response
            market_data: Actual market data for hallucination check
            risk_level: Risk level for disclaimer
            
        Returns:
            Tuple of (is_valid, processed_response, error_message)
        """
        # Check for harmful/non-compliant content
        output_result = OutputGuardrails.validate_output(ai_response)
        if not output_result.passed:
            error_msg = self._get_safe_error_message(output_result)
            return False, ai_response, error_msg
        
        # Check for hallucinations
        hallucination_result = HallucinationGuardrails.detect_hallucinations(
            ai_response, 
            market_data
        )
        if not hallucination_result.passed:
            if self.strict_mode:
                error_msg = self._get_safe_error_message(hallucination_result)
                return False, ai_response, error_msg
            else:
                logger.warning(f"Potential hallucination allowed: {hallucination_result.violation_details}")
        
        # Check for PII in output
        pii_result = PIIGuardrails.validate_no_pii(ai_response)
        if not pii_result.passed:
            ai_response = pii_result.sanitized_content or ai_response
            logger.warning("PII masked in AI response")
        
        # Add disclaimers
        ai_response = OutputGuardrails.add_disclaimers(ai_response, risk_level)
        
        return True, ai_response, None
    
    def _get_safe_error_message(self, result: GuardrailResult) -> str:
        """Get a user-friendly error message without revealing system details"""
        messages = {
            GuardrailViolationType.PROMPT_INJECTION: 
                "I'm sorry, but I can only help with financial planning and investment questions. Could you please rephrase your query?",
            GuardrailViolationType.ADVERSARIAL_PROMPT:
                "I'm unable to provide advice on this topic as it may involve activities that are not in your best interest or may not be legally compliant. Let me help you with legitimate investment planning instead.",
            GuardrailViolationType.OFF_TOPIC:
                "I'm a financial advisor assistant. I'd be happy to help with investment planning, risk assessment, portfolio allocation, or other financial questions. What would you like to know?",
            GuardrailViolationType.NON_COMPLIANT_ADVICE:
                "I apologize, but I cannot provide that type of advice as it may not comply with financial regulations. Let me offer you some compliant alternatives.",
            GuardrailViolationType.HALLUCINATION:
                "I need to verify some information to ensure accuracy. Let me provide you with verified market data.",
            GuardrailViolationType.HARMFUL_CONTENT:
                "I'm sorry, but I cannot engage with this type of content. I'm here to help with financial planning.",
            GuardrailViolationType.PII_DETECTED:
                "For your security, please avoid sharing sensitive personal information like Aadhaar, PAN, or bank account numbers in our conversation.",
        }
        
        return messages.get(result.violation_type, 
            "I encountered an issue processing your request. Please try rephrasing your question.")
    
    def get_guardrail_status(self) -> Dict:
        """Get current guardrails configuration status"""
        return {
            "enabled": True,
            "strict_mode": self.strict_mode,
            "components": {
                "input_validation": True,
                "pii_protection": True,
                "output_filtering": True,
                "hallucination_detection": True
            },
            "version": "1.0.0"
        }
