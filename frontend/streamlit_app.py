"""
Streamlit Frontend for Financial Advisory System
Interactive dashboard for users to get investment advice
"""
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Advisor AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .risk-card h3 {
        color: #000000;
        font-weight: bold;
    }
    .risk-card p {
        color: #1a1a1a;
        font-weight: 500;
    }
    .conservative {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .moderate {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .aggressive {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=15)
            return response.status_code == 200
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue  # Retry
            st.warning("⏳ Backend is slow to respond. It might be processing other requests.")
            return False
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to backend - server might not be running")
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            st.error(f"🔍 Debug: Backend connection issue - {str(e)}")
            return False
    return False


def get_user_by_id(user_id):
    """Get user by ID"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/users/{user_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def get_user_by_email(email):
    """Get user by email"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/users/email/{email}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def get_customer_data(customer_id):
    """Get customer data from risk profiling database"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/customer/{customer_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def search_customer_ids(prefix):
    """Search for customer IDs matching the given prefix"""
    try:
        if not prefix or len(prefix) < 1:
            return []
        response = requests.get(f"{API_BASE_URL}/api/customer/search/{prefix}")
        if response.status_code == 200:
            data = response.json()
            return data.get('customer_ids', [])
        return []
    except Exception as e:
        return []


def get_user_history(user_id, limit=10):
    """Get user's advisory history"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/advisory/history/{user_id}", params={"limit": limit})
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        return []


def create_user(name, email, age, occupation):
    """Create a new user or return existing user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/users/",
            json={
                "name": name,
                "email": email,
                "age": age,
                "occupation": occupation
            }
        )
        if response.status_code == 201:
            return {"user": response.json(), "is_new": True}
        elif response.status_code == 400 and "already exists" in response.json().get("detail", ""):
            # User already exists, this is handled in the UI now
            return {"user": None, "is_new": False, "error": "User already exists"}
        return None
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return None


def get_customer_investment_advice(customer_id, query="Provide investment recommendations based on my risk profile", session_id=None):
    """Get investment advice using customer_id from risk profiling database"""
    try:
        params = {"query": query}
        if session_id:
            params["session_id"] = session_id
            
        response = requests.post(
            f"{API_BASE_URL}/api/advisory/customer/{customer_id}",
            params=params,
            timeout=120  # 2 minute timeout for AI processing
        )
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The AI is taking longer than expected. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error getting advice: {e}")
        return None


def get_investment_advice(user_id, query, financial_data):
    """Get investment advice from API (legacy method)"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/advisory/query",
            json={
                "user_id": user_id,
                "query": query,
                **financial_data
            }
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error getting advice: {e}")
        return None


def submit_feedback(session_id, feedback):
    """Submit feedback for a session"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/advisory/feedback",
            json={
                "session_id": session_id,
                "feedback": feedback
            }
        )
        return response.status_code == 200
    except:
        return False


def plot_portfolio_allocation(portfolio):
    """Create pie chart for portfolio allocation"""
    fig = go.Figure(data=[go.Pie(
        labels=list(portfolio.keys()),
        values=list(portfolio.values()),
        hole=.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title="Recommended Portfolio Allocation",
        height=400
    )
    return fig


def plot_risk_profile(risk_probabilities):
    """Create bar chart for risk probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(risk_probabilities.keys()),
            y=list(risk_probabilities.values()),
            marker_color=['#4caf50', '#ff9800', '#f44336']
        )
    ])
    
    fig.update_layout(
        title="Risk Profile Assessment",
        xaxis_title="Risk Category",
        yaxis_title="Probability",
        height=350
    )
    return fig


def plot_shap_values(shap_values):
    """Create bar chart for SHAP feature importance"""
    features = list(shap_values.keys())
    values = list(shap_values.values())
    
    # Sort by absolute value
    sorted_data = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
    features, values = zip(*sorted_data)
    
    colors = ['#f44336' if v < 0 else '#4caf50' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=values,
            orientation='h',
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title="Feature Importance (SHAP Values)",
        xaxis_title="Impact on Risk Assessment",
        yaxis_title="Feature",
        height=400
    )
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<p class="main-header">💰 AI Financial Advisor</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check API health
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ Backend API is not running or not reachable.")
        st.info(f"**Expected API URL:** {API_BASE_URL}")
        st.info("**How to start the backend:**\n```bash\ncd backend && python main.py\n```")
        st.info("**Verify backend is running:** Open http://localhost:8000/docs in your browser")
        
        if st.button("🔄 Retry Connection"):
            st.rerun()
        return
    
    # Sidebar - Customer Information (SINGLE INSTANCE)
    with st.sidebar:
        st.header("👤 Customer Lookup")
        
        # Initialize session state
        if 'customer_id' not in st.session_state:
            st.session_state.customer_id = None
        if 'customer_data' not in st.session_state:
            st.session_state.customer_data = None
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        
        if st.session_state.customer_id is None:
            st.subheader("Enter Customer ID")
            
            # Text input for customer ID with autocomplete
            customer_id_input = st.text_input(
                "Customer ID",
                placeholder="Type 6-digit customer ID (e.g., 100000)",
                help="Type customer ID to search - matching IDs will appear below",
                key="customer_id_input_main"
            )
            
            # Show matching suggestions if user has typed something
            if customer_id_input and customer_id_input.isdigit() and len(customer_id_input) >= 1:
                # Fetch matching customer IDs
                matching_ids = search_customer_ids(customer_id_input)
                
                if matching_ids:
                    st.caption(f"🔍 Found {len(matching_ids)} matching customer ID(s):")
                    
                    # Display matching IDs as clickable options (limited to first 10)
                    display_ids = matching_ids[:10]
                    
                    # Create a nice formatted list
                    for cid in display_ids:
                        st.markdown(f"- `{cid}`")
                    
                    if len(matching_ids) > 10:
                        st.caption(f"... and {len(matching_ids) - 10} more. Keep typing to narrow down.")
                else:
                    st.warning("No matching customer IDs found")
            
            if st.button("Load Customer Data", type="primary", key="load_customer_btn_main"):
                if customer_id_input and customer_id_input.isdigit():
                    customer_id = int(customer_id_input)
                    customer_data = get_customer_data(customer_id)
                    if customer_data:
                        st.session_state.customer_id = customer_id
                        st.session_state.customer_data = customer_data
                        st.session_state.search_results = []
                        # Clear any previous results
                        if 'last_result' in st.session_state:
                            del st.session_state.last_result
                        st.rerun()
                    else:
                        st.error(f"❌ Customer ID {customer_id} not found in database.")
                        st.info("Please check the customer ID and try again.")
                else:
                    st.warning("Please enter a valid customer ID (numbers only)")
        else:
            # Show customer info with consistent formatting
            if st.session_state.customer_data:
                data = st.session_state.customer_data
                st.success(f"✅ Customer ID: {st.session_state.customer_id}")
                
                # Display info with consistent font size
                st.markdown(f"""
                <div style="background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <p style="margin: 5px 0; font-size: 14px;"><b>👤 Age:</b> {data.get('age', 'N/A')}</p>
                    <p style="margin: 5px 0; font-size: 14px;"><b>💰 Annual Income:</b> ₹{data.get('annual_income', 0):,.0f}</p>
                    <p style="margin: 5px 0; font-size: 14px;"><b>💎 Net Worth:</b> ₹{data.get('net_worth', 0):,.0f}</p>
                    <p style="margin: 5px 0; font-size: 14px;"><b>📊 Risk Profile:</b> {data.get('risk_profile', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ Customer ID: {st.session_state.customer_id}")
            
            if st.button("Switch Customer", type="secondary", key="switch_customer_btn_sidebar"):
                st.session_state.customer_id = None
                st.session_state.customer_data = None
                st.session_state.search_results = []
                if 'last_result' in st.session_state:
                    del st.session_state.last_result
                if 'chat_history' in st.session_state:
                    st.session_state.chat_history = []
                if 'current_session_id' in st.session_state:
                    del st.session_state.current_session_id
                st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This AI-powered financial advisor uses multiple intelligent agents to provide personalized investment recommendations.")
    
    # Main content area
    if st.session_state.customer_id is None:
        st.info("👈 Please enter a Customer ID to get started")
        return
    
    # Create tabs with full functionality
    tab1, tab2, tab3 = st.tabs(["💡 Get Advice", "📊 Market Data", "📜 History"])
    
    with tab1:
        st.header("🤖 AI Financial Advisor")
        st.markdown(f"### Personalized Advice for Customer #{st.session_state.customer_id}")
        
        st.markdown("---")
        
        # CHAT MODE
        st.subheader("💬 Chat with AI Financial Advisor")
        
        # Show example queries
        with st.expander("💡 Example Questions", expanded=False):
            st.markdown("""
            **Investment Planning:**
            - "I want to invest ₹8 lakhs in 4 years for my daughter's higher education"
            - "Need to invest ₹10 lakhs for 5 years for my child's education with moderate risk"
            - "I want to save ₹5 lakhs in 2 years for a car with conservative risk tolerance"
            
            **Portfolio Questions:**
            - "What's the best portfolio allocation for my risk profile?"
            - "How should I diversify my investments?"
            - "What are the recommended investment instruments for me?"
            
            **General Advice:**
            - "Provide investment recommendations based on my risk profile"
            - "What investment strategy suits my financial situation?"
            - "How can I maximize returns while managing risk?"
            """)
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background-color: #1f1f1f; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <b>👤 You:</b><br>{message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #0e4c92; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <b>🤖 AI Advisor:</b><br>{message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input field and send button
        col_input, col_button = st.columns([4, 1])
        
        with col_input:
            user_message = st.text_input(
                "Type your message...",
                placeholder="Type your investment query here...",
                key="chat_input_field",
                label_visibility="collapsed"
            )
        
        with col_button:
            send_button = st.button("Send 📤", use_container_width=True, type="primary", key="send_chat_btn")
        
        # Clear chat button
        if len(st.session_state.chat_history) > 0:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_btn"):
                st.session_state.chat_history = []
                if 'last_result' in st.session_state:
                    del st.session_state['last_result']
                if 'current_session_id' in st.session_state:
                    del st.session_state['current_session_id']
                st.rerun()
        
        if send_button and user_message:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_message
            })
            
            # Get session_id from previous interactions (for conversation continuity)
            session_id = st.session_state.get('current_session_id', None)
            
            # Get AI response
            with st.spinner("🤖 AI is thinking..."):
                result = get_customer_investment_advice(
                    st.session_state.customer_id, 
                    user_message,
                    session_id=session_id
                )
                
                if result and result.get('recommendation'):
                    # Store session_id for future requests (conversation continuity)
                    if result.get('session_id'):
                        st.session_state.current_session_id = result['session_id']
                    
                    recommendation = result['recommendation']
                    # Get the reasoning/recommendation text
                    # Handle case where recommendation might be a string (raw JSON) instead of dict
                    if isinstance(recommendation, str):
                        try:
                            recommendation = json.loads(recommendation)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    
                    if isinstance(recommendation, dict):
                        ai_response = recommendation.get('reasoning', recommendation.get('recommendation', recommendation.get('message', 'I apologize, I could not generate a recommendation.')))
                    else:
                        ai_response = str(recommendation)
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': ai_response
                    })
                    
                    # Store the full result for display
                    st.session_state.last_result = result
                else:
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': "I apologize, I encountered an error. Could you please rephrase your question?"
                    })
            
            st.rerun()
        
        if not user_message and send_button:
            st.warning("⚠️ Please type a message before sending")
        
        # Display results
        if 'last_result' in st.session_state and st.session_state.last_result:
            result = st.session_state.last_result
            st.markdown("---")
            
            # Display Risk Profile
            if 'risk_profile' in result and result['risk_profile']:
                st.subheader("📊 Your Risk Profile")
                risk_data = result['risk_profile']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Risk Category",
                        risk_data.get('risk_category', 'Unknown'),
                        help="Your assessed risk tolerance"
                    )
                with col2:
                    confidence = risk_data.get('prediction_confidence', 0)
                    st.metric(
                        "Confidence",
                        f"{confidence:.1%}",
                        help="Model confidence in risk assessment"
                    )
                with col3:
                    age = risk_data.get('age', 'N/A')
                    st.metric(
                        "Age",
                        age,
                        help="Your current age"
                    )
            
            # Display Recommendation - only show when AI provides actual recommendations (not questions)
            if 'recommendation' in result and result['recommendation']:
                rec_data = result['recommendation']
                
                # Check if it's a question (AI asking for more info)
                # Also check if reasoning contains question indicators
                is_question = rec_data.get('is_question', False)
                reasoning_text = rec_data.get('reasoning') or rec_data.get('recommendation', '')
                
                # Additional check: if reasoning contains question patterns, treat as question
                if not is_question and reasoning_text:
                    question_indicators = ['how much', 'how long', 'what is your', 'could you', 'please provide', 'need a few more details', 'need more details']
                    if any(indicator in reasoning_text.lower() for indicator in question_indicators):
                        is_question = True
                
                # Only show Investment Recommendations section if NOT a question
                if is_question:
                    # Don't show anything here - the question is already displayed in chat history
                    pass
                else:
                    # Show the Investment Recommendations section header
                    st.subheader("💡 Investment Recommendations")
                    
                    # Main recommendation text (check both 'reasoning' and 'recommendation' keys)
                    reasoning_text = rec_data.get('reasoning') or rec_data.get('recommendation', '')
                    if reasoning_text:
                        st.markdown(f"""
                        <div style="background-color: #0e4c92; padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                            <h3 style="color: #ffffff; margin-top: 0;">🎯 Recommended Strategy</h3>
                            <p style="color: #ffffff; font-size: 16px; line-height: 1.6;">
                                {reasoning_text}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Portfolio allocation (check both 'portfolio' and 'portfolio_allocation' keys)
                    portfolio = rec_data.get('portfolio') or rec_data.get('portfolio_allocation', {})
                    if portfolio:
                        st.markdown("#### 📈 Suggested Portfolio Allocation")
                        fig_portfolio = plot_portfolio_allocation(portfolio)
                        st.plotly_chart(fig_portfolio, use_container_width=True)
                    
                    # Asset allocation details
                    if 'asset_allocation' in rec_data and rec_data['asset_allocation']:
                        st.markdown("#### 💼 Asset Allocation Details")
                        for asset, details in rec_data['asset_allocation'].items():
                            if isinstance(details, dict):
                                with st.expander(f"📌 {asset} ({details.get('allocation', 0)}%)"):
                                    st.write(f"**Instrument:** {details.get('instrument', 'N/A')}")
                                    st.write(f"**Expected Return:** {details.get('expected_return', 0):.1f}%")
                                    st.write(f"**Reason:** {details.get('reason', 'N/A')}")
                    
                    # Action steps (check both 'action_steps' and 'specific_recommendations' keys)
                    action_steps = rec_data.get('action_steps') or rec_data.get('specific_recommendations', [])
                    if action_steps:
                        st.markdown("#### 📝 Action Steps")
                        for i, step in enumerate(action_steps, 1):
                            st.markdown(f"**{i}.** {step}")
                    
                    # Expected return
                    if rec_data.get('expected_return_annual', 0) > 0:
                        st.success(f"📈 **Expected Annual Return:** {rec_data['expected_return_annual']:.1f}%")
            
            # Display Explanation (XAI)
            if 'explanation' in result and result['explanation']:
                with st.expander("🔍 Why These Recommendations? (AI Explanation)", expanded=True):
                    exp_data = result['explanation']
                    
                    # Display simplified LLM explanation first (most user-friendly)
                    if exp_data.get('simplified_explanation'):
                        st.markdown("### 🤖 AI Explanation")
                        st.markdown(exp_data['simplified_explanation'])
                        st.markdown("---")
                    
                    # Display recommendation explanation if available
                    if exp_data.get('recommendation_explanation'):
                        st.markdown("### 💼 Investment Strategy Explanation")
                        st.markdown(exp_data['recommendation_explanation'])
                        st.markdown("---")
                    
                    # Display key factors from SHAP/LIME analysis
                    if exp_data.get('key_factors') and len(exp_data['key_factors']) > 0:
                        st.markdown("### 📊 Key Factors Influencing Your Risk Profile")
                        for factor in exp_data['key_factors']:
                            if isinstance(factor, dict):
                                factor_name = factor.get('factor', '').replace('_', ' ').title()
                                value = factor.get('value', '')
                                impact = factor.get('impact', 0)
                                direction = factor.get('direction', 'neutral')
                                
                                # Format value and impact properly
                                value_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                                impact_str = f"{impact:.4f}" if isinstance(impact, (int, float)) else str(impact)
                                
                                if direction == 'positive':
                                    st.success(f"✅ **{factor_name}**: {value_str} (Impact: {impact_str})")
                                else:
                                    st.warning(f"⚠️ **{factor_name}**: {value_str} (Impact: {impact_str})")
                            else:
                                st.write(f"• {factor}")
                        st.markdown("---")
                    
                    # Display SHAP values chart if available
                    if exp_data.get('shap_values') and len(exp_data['shap_values']) > 0:
                        st.markdown("### 📈 Feature Importance (SHAP Analysis)")
                        fig_shap = plot_shap_values(exp_data['shap_values'])
                        st.plotly_chart(fig_shap, use_container_width=True)
                        st.markdown("---")
                    
                    # Display risk assessment explanation if available
                    if 'risk_assessment' in exp_data and isinstance(exp_data['risk_assessment'], dict):
                        risk_exp = exp_data['risk_assessment']
                        if risk_exp.get('confidence'):
                            st.markdown(f"**Model Confidence:** {risk_exp['confidence']:.1%}")
                        if risk_exp.get('explanation_methods_used'):
                            st.markdown(f"**Analysis Methods Used:** {', '.join(risk_exp['explanation_methods_used'])}")
                    
                    # Display portfolio explanation
                    if 'portfolio_breakdown' in exp_data and exp_data['portfolio_breakdown']:
                        st.markdown("### 💼 Portfolio Analysis")
                        breakdown = exp_data['portfolio_breakdown']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Equity Exposure", f"{breakdown.get('equity_exposure', 0)}%")
                        with col2:
                            st.metric("Debt Exposure", f"{breakdown.get('debt_exposure', 0)}%")
                        with col3:
                            st.metric("Other Assets", f"{breakdown.get('other_exposure', 0)}%")
                    
                    if exp_data.get('risk_alignment'):
                        st.info(f"**Risk Alignment:** {exp_data['risk_alignment']}")
                    
                    if exp_data.get('market_context'):
                        st.info(f"**Market Context:** {exp_data['market_context']}")
                    
                    if exp_data.get('expected_outcome'):
                        st.success(f"**Expected Outcome:** {exp_data['expected_outcome']}")
    
    with tab2:
        st.header("📊 Current Market Data")
        
        try:
            response = requests.get(f"{API_BASE_URL}/api/market/data")
            if response.status_code == 200:
                market_data = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Repo Rate", f"{market_data['repo_rate']:.2f}%")
                    st.metric("Inflation Rate", f"{market_data['inflation_rate']:.2f}%")
                
                with col2:
                    st.metric("Gold 24K (10g)", f"₹{market_data['gold_rates']['price_per_10_gram_24k']:,.0f}")
                    if market_data.get('nifty_data'):
                        nifty = market_data['nifty_data']
                        st.metric(
                            "NIFTY 50",
                            f"{nifty['current_price']:,.0f}",
                            delta=f"{nifty['monthly_return']:.2f}%"
                        )
                
                with col3:
                    pass  # Reserved for future metrics
                
                st.markdown("---")
                
                # Display Fixed Deposit Rates
                st.subheader("🏦 Fixed Deposit Interest Rates")
                
                fd_data = []
                tenure_labels = {
                    "1_month": "1 Month",
                    "2_months": "2 Months",
                    "3_months": "3 Months",
                    "6_months": "6 Months",
                    "9_months": "9 Months",
                    "1_year": "1 Year",
                    "15_months": "15 Months",
                    "18_months": "18 Months",
                    "2_years": "2 Years",
                    "3_years": "3 Years",
                    "5_years": "5 Years",
                    "10_years": "10 Years"
                }
                
                if 'regular' in market_data['fd_rates']:
                    regular_rates = market_data['fd_rates']['regular']
                    senior_rates = market_data['fd_rates']['senior_citizen']
                    
                    for tenure_key, label in tenure_labels.items():
                        if tenure_key in regular_rates:
                            fd_data.append({
                                "Tenure": label,
                                "Regular Rate (%)": f"{regular_rates[tenure_key]:.2f}",
                                "Senior Citizen Rate (%)": f"{senior_rates.get(tenure_key, regular_rates[tenure_key]):.2f}"
                            })
                else:
                    for tenure_key, label in tenure_labels.items():
                        if tenure_key in market_data['fd_rates']:
                            fd_data.append({
                                "Tenure": label,
                                "Interest Rate (%)": f"{market_data['fd_rates'][tenure_key]:.2f}"
                            })
                
                if fd_data:
                    fd_df = pd.DataFrame(fd_data)
                    st.dataframe(fd_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Display mutual fund returns
                st.subheader("📈 Mutual Fund Returns (1-Year Average)")
                
                mf_data = []
                for category, funds in market_data['mf_returns'].items():
                    for fund_type, returns in funds.items():
                        mf_data.append({
                            "Category": category.capitalize(),
                            "Fund Type": fund_type.replace("_", " ").title(),
                            "Return (%)": returns
                        })
                
                df = pd.DataFrame(mf_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Error loading market data: {e}")
    
    with tab3:
        st.header("📜 Your Advisory History")
        
        st.info("💡 This feature tracks your past advisory sessions and learned preferences")
        
        # Note: The history API uses user_id, but we're using customer_id system
        # For now, showing a placeholder. Can be enhanced to track customer advisory history
        
        st.markdown("### 🧠 Your Risk Profile")
        if st.session_state.customer_data:
            data = st.session_state.customer_data
            
            # Get ML-predicted risk profile if available
            predicted_risk = "Not yet assessed"
            confidence = 0
            if 'last_result' in st.session_state and st.session_state.last_result:
                risk_data = st.session_state.last_result.get('risk_profile', {})
                predicted_risk = risk_data.get('risk_category', 'Not yet assessed')
                confidence = risk_data.get('prediction_confidence', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if confidence > 0:
                    st.metric("Predicted Risk Profile", predicted_risk, delta=f"{confidence:.1%} confidence")
                else:
                    st.metric("Predicted Risk Profile", predicted_risk)
            with col2:
                st.metric("Annual Income", f"₹{data.get('annual_income', 0):,.0f}")
            with col3:
                st.metric("Net Worth", f"₹{data.get('net_worth', 0):,.0f}")
        
        st.markdown("---")
        st.markdown("### 📊 Customer Financial Overview")
        
        if st.session_state.customer_data:
            data = st.session_state.customer_data
            
            # Financial Overview
            st.subheader("💰 Financial Snapshot")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Age", data.get('age', 'N/A'))
                st.metric("Dependents", data.get('dependents', 'N/A'))
                st.metric("Customer Segment", data.get('customer_segment', 'N/A'))
            
            with col2:
                st.metric("Education", data.get('education', 'N/A'))
                st.metric("Total Debt", f"₹{data.get('total_debt', 0):,.0f}")
                st.metric("Investment Portfolio", f"₹{data.get('investment_portfolio_value', 0):,.0f}")
            
            st.markdown("---")
            st.subheader("📈 Investment Performance")
            
            annual_return = data.get('annual_investment_return', 0)
            portfolio_value = data.get('investment_portfolio_value', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Annual Investment Return", f"₹{annual_return:,.0f}")
            with col2:
                if portfolio_value > 0:
                    return_rate = (annual_return / portfolio_value) * 100
                    st.metric("Return Rate", f"{return_rate:.2f}%")
                else:
                    st.metric("Return Rate", "N/A")
        
        st.markdown("---")
        st.markdown("### 💡 Recommendations Summary")
        
        if 'last_result' in st.session_state and st.session_state.last_result:
            result = st.session_state.last_result
            
            st.success("✅ Latest recommendation available")
            
            if result.get('risk_profile'):
                risk_data = result['risk_profile']
                st.info(f"**Predicted Risk Category:** {risk_data.get('risk_category', 'Unknown')} (Confidence: {risk_data.get('prediction_confidence', 0):.1%})")
            
            if result.get('recommendation'):
                with st.expander("View Latest Recommendation", expanded=True):
                    rec_data = result['recommendation']
                    
                    # Check if it's a question
                    if rec_data.get('is_question', False):
                        st.info(f"🤖 **AI asked:** {rec_data.get('message') or rec_data.get('reasoning', '')}")
                    else:
                        # Display main recommendation (check both 'reasoning' and 'recommendation' keys)
                        reasoning_text = rec_data.get('reasoning') or rec_data.get('recommendation', '')
                        if reasoning_text:
                            st.markdown("### 🎯 Investment Recommendation")
                            st.markdown(reasoning_text)
                        
                        # Display portfolio allocation (check both 'portfolio' and 'portfolio_allocation' keys)
                        portfolio = rec_data.get('portfolio') or rec_data.get('portfolio_allocation', {})
                        if portfolio:
                            st.markdown("### 📊 Portfolio Allocation")
                            for asset, percentage in portfolio.items():
                                st.write(f"- **{asset}**: {percentage}%")
                        
                        # Display action steps (check both 'action_steps' and 'specific_recommendations' keys)
                        action_steps = rec_data.get('action_steps') or rec_data.get('specific_recommendations', [])
                        if action_steps:
                            st.markdown("### 📝 Action Steps")
                            for i, step in enumerate(action_steps, 1):
                                st.write(f"{i}. {step}")
                        
                        # Expected return
                        if rec_data.get('expected_return_annual', 0) > 0:
                            st.success(f"📈 **Expected Annual Return:** {rec_data['expected_return_annual']:.1f}%")
            
            # Display explanation summary
            if result.get('explanation'):
                with st.expander("View Explanation Summary", expanded=False):
                    exp_data = result['explanation']
                    
                    if exp_data.get('simplified_explanation'):
                        st.markdown("### 🤖 AI Explanation")
                        st.markdown(exp_data['simplified_explanation'])
                    
                    if exp_data.get('methods_used'):
                        st.markdown(f"**Analysis Methods:** {', '.join(exp_data['methods_used'])}")
        else:
            st.warning("No recommendations generated yet. Go to 'Get Advice' tab to get started!")


if __name__ == "__main__":
    main()
