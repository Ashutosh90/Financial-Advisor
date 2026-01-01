"""
Example: Using the Updated Risk Agent

This example shows how to integrate the updated risk_agent.py
with the Streamlit UI and Advisor Agent.
"""

import sys
sys.path.append('./backend')

from agents.risk_agent import RiskAgent

def example_streamlit_integration():
    """
    Example of how to use risk_agent in Streamlit UI
    
    In your streamlit_app.py, you can use:
    """
    print("\n" + "="*80)
    print("STREAMLIT INTEGRATION EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize the risk agent (do this once, maybe in session_state)
    risk_agent = RiskAgent()
    
    # User enters customer ID in UI
    customer_id = 100002  # This would come from st.number_input()
    
    # Get risk profile
    result = risk_agent.predict_risk_profile(customer_id)
    
    # Display in Streamlit
    print(f"Customer ID: {result['customer_id']}")
    print(f"Age: {result['age']}")
    print(f"Annual Income: ${result['annual_income']:,.2f}")
    print(f"Net Worth: ${result['net_worth']:,.2f}")
    print(f"\n🎯 Risk Profile: {result['risk_category']}")
    print(f"📊 Confidence: {result['prediction_confidence']:.1%}")
    print(f"\nProbabilities:")
    for category, prob in result['risk_probabilities'].items():
        print(f"  {category}: {prob:.1%}")
    
    return result


def example_advisor_agent_integration(customer_id: int):
    """
    Example of how advisor_agent can use risk predictions
    
    In your advisor_agent.py:
    """
    print("\n" + "="*80)
    print("ADVISOR AGENT INTEGRATION EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize risk agent
    risk_agent = RiskAgent()
    
    # Get risk profile
    risk_result = risk_agent.predict_risk_profile(customer_id)
    
    # Use the risk category to tailor recommendations
    risk_category = risk_result['risk_category']
    confidence = risk_result['prediction_confidence']
    
    print(f"Customer {customer_id} Risk Profile: {risk_category} ({confidence:.1%} confidence)")
    
    # Generate personalized recommendations based on risk
    if risk_category == "Aggressive":
        print("\n💼 Investment Recommendation: Aggressive Portfolio")
        print("  - 70% Equities (Growth stocks, Mid-caps)")
        print("  - 20% Alternative Investments (REITs, Commodities)")
        print("  - 10% Bonds (High-yield corporate)")
        print("\n  Expected Return: 12-15% annually")
        print("  Risk Level: High volatility expected")
        
    elif risk_category == "Conservative":
        print("\n💼 Investment Recommendation: Conservative Portfolio")
        print("  - 60% Bonds (Government, Investment-grade corporate)")
        print("  - 30% Equities (Blue-chip, Dividend stocks)")
        print("  - 10% Cash/Money Market")
        print("\n  Expected Return: 6-8% annually")
        print("  Risk Level: Low to moderate volatility")
    
    return {
        "risk_profile": risk_result,
        "recommendations": "Generated based on risk category"
    }


def example_batch_processing():
    """
    Example of processing multiple customers
    """
    print("\n" + "="*80)
    print("BATCH PROCESSING EXAMPLE")
    print("="*80 + "\n")
    
    risk_agent = RiskAgent()
    
    # Process multiple customers
    customer_ids = [100000, 100002, 100003, 100004]
    
    results = []
    for cust_id in customer_ids:
        result = risk_agent.predict_risk_profile(cust_id)
        results.append(result)
        
        print(f"Customer {cust_id}: {result['risk_category']:12} "
              f"(Confidence: {result['prediction_confidence']:5.1%}, "
              f"Income: ${result['annual_income']:>10,.0f})")
    
    # Summary statistics
    aggressive_count = sum(1 for r in results if r['risk_category'] == 'Aggressive')
    conservative_count = sum(1 for r in results if r['risk_category'] == 'Conservative')
    
    print(f"\n📊 Summary:")
    print(f"  Aggressive: {aggressive_count}/{len(results)} ({aggressive_count/len(results):.1%})")
    print(f"  Conservative: {conservative_count}/{len(results)} ({conservative_count/len(results):.1%})")
    
    return results


if __name__ == "__main__":
    # Run examples
    
    # 1. Streamlit integration example
    streamlit_result = example_streamlit_integration()
    
    # 2. Advisor agent integration example
    advisor_result = example_advisor_agent_integration(100003)
    
    # 3. Batch processing example
    batch_results = example_batch_processing()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80 + "\n")
