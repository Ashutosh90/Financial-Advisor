# System Architecture and Implementation Guide

## Multi-Agent Financial Advisory System - Technical Documentation

---

## 1. System Overview

### 1.1 Purpose
The Multi-Agent Financial Advisory System is designed to provide personalized investment recommendations to Indian retail investors using a coordinated network of AI agents, each specializing in specific tasks.

### 1.2 Key Components

```
User Interface (Streamlit)
         ↓
    FastAPI Backend
         ↓
   Orchestrator (Agent Coordinator)
         ↓
    ┌────┴────┬────────┬──────────┬─────────┐
    ↓         ↓        ↓          ↓         ↓
  Memory   Data     Risk      Advisor    XAI
  Agent    Agent    Agent     Agent     Agent
```

---

## 2. Agent Architecture

### 2.1 Memory Agent
**Purpose**: Store and recall user preferences, learn from feedback

**Key Features**:
- Session storage in SQLite
- User preference learning
- Feedback processing
- Personalization engine

**Technologies**:
- SQLAlchemy ORM
- SQLite database
- Pattern recognition algorithms

**Data Stored**:
```python
- User profiles
- Financial profiles
- Session history
- User preferences (learned)
- Feedback data
```

### 2.2 Data Agent
**Purpose**: Fetch and process real-time market data

**Key Features**:
- Fixed Deposit rates from major banks
- RBI repo rate
- Inflation data (CPI)
- Mutual fund returns (debt, equity, hybrid)
- Stock index data (NIFTY 50)
- Gold prices

**Technologies**:
- yfinance for market data
- Web scraping capabilities
- Data caching mechanism

**Data Sources**:
```python
- yfinance: Stock market data, ETFs, gold
- RBI (simulated): Repo rate, inflation
- Bank APIs (simulated): FD rates
- AMFI (simulated): Mutual fund NAVs
```

### 2.3 Risk Agent
**Purpose**: Assess user's risk profile using machine learning

**Key Features**:
- XGBoost classification model
- Risk categorization (Conservative, Moderate, Aggressive)
- Feature engineering
- Probability distribution

**Technologies**:
- XGBoost
- Scikit-learn
- Pandas for data processing

**Risk Assessment Features**:
```python
1. Monthly Income (normalized)
2. Savings Ratio (income - expenses / income)
3. Investment to Income Ratio
4. Age Factor (risk capacity decreases with age)
5. Investment Duration (longer = more risk capacity)
```

**Model Training**:
```python
# Synthetic data generation for initial model
- 1000 samples with varied profiles
- Features: income, savings, duration, age
- Labels: Risk categories (0, 1, 2)
- Model: XGBoost with 100 estimators
- Saved as: models/risk_model.json
```

### 2.4 Advisor Agent
**Purpose**: Generate personalized recommendations using LLM

**Key Features**:
- OpenAI GPT4 integration
- Context-aware prompting
- Portfolio allocation
- Expected returns calculation
- Natural language explanations

**Technologies**:
- OpenAI API
- LangChain (framework)
- Prompt engineering

**Recommendation Process**:
```python
1. Receive: Risk profile + Market data + User query
2. Generate: System prompt (financial advisor role)
3. Create: User prompt with all context
4. Call: OpenAI API
5. Parse: JSON response with portfolio
6. Fallback: Rule-based if LLM fails
```

**Output Structure**:
```json
{
  "portfolio": {"FD": 50, "MF": 30, ...},
  "expected_return_annual": 8.5,
  "risk_level": "Moderate",
  "reasoning": "...",
  "asset_allocation": {...},
  "action_steps": [...]
}
```

### 2.5 XAI Agent
**Purpose**: Provide explainability and transparency

**Key Features**:
- SHAP values for feature importance
- LIME explanations
- Natural language interpretation
- Visual explanations

**Technologies**:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Plotly for visualizations

**Explainability Layers**:
```python
1. Risk Profile Explanation
   - Feature importance
   - SHAP values
   - Why this risk category?

2. Recommendation Explanation
   - Portfolio composition reasoning
   - Market context
   - Risk alignment

3. Visual Explanations
   - SHAP bar charts
   - Feature impact plots
```

### 2.6 Orchestrator
**Purpose**: Coordinate all agents in the workflow

**Key Features**:
- Agent lifecycle management
- Request routing
- Response aggregation
- Error handling

**Workflow**:
```python
def process_query():
    1. Get user context (Memory Agent)
    2. Fetch market data (Data Agent)
    3. Assess risk (Risk Agent)
    4. Generate recommendation (Advisor Agent)
    5. Explain decision (XAI Agent)
    6. Store session (Memory Agent)
    7. Return aggregated response
```

---

## 3. Technical Stack

### 3.1 Backend
```yaml
Framework: FastAPI
Language: Python 3.10+
Database: SQLite (SQLAlchemy ORM)
API Documentation: Swagger/OpenAPI
Logging: Loguru
```

### 3.2 Machine Learning
```yaml
Risk Model: XGBoost Classifier
Explainability: SHAP, LIME
Data Processing: Pandas, NumPy
Feature Engineering: Custom transformations
```

### 3.3 LLM Integration
```yaml
Provider: OpenAI
Model: GPT-3.5-turbo
Framework: LangChain
Fallback: Rule-based system
```

### 3.4 Frontend
```yaml
Framework: Streamlit
Visualization: Plotly
Charts: Pie, Bar, Line
Interactive: Real-time updates
```

### 3.5 Data Sources
```yaml
Market Data: yfinance
Financial Data: Simulated RBI, bank APIs
Caching: In-memory + database
```

---

## 4. Database Schema

### 4.1 Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    age INTEGER,
    occupation VARCHAR(100),
    created_at DATETIME,
    updated_at DATETIME
);
```

### 4.2 Financial Profiles Table
```sql
CREATE TABLE financial_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY,
    monthly_income FLOAT,
    monthly_expenses FLOAT,
    current_savings FLOAT,
    investment_amount FLOAT,
    investment_duration_months INTEGER,
    financial_goal VARCHAR(200),
    risk_tolerance VARCHAR(50),
    created_at DATETIME,
    updated_at DATETIME
);
```

### 4.3 Sessions Table
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY,
    session_id VARCHAR(100) UNIQUE,
    query TEXT,
    recommendation JSON,
    risk_profile JSON,
    feedback VARCHAR(50),
    created_at DATETIME
);
```

### 4.4 User Preferences Table
```sql
CREATE TABLE user_preferences (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY,
    preference_key VARCHAR(100),
    preference_value TEXT,
    confidence_score FLOAT,
    created_at DATETIME,
    updated_at DATETIME
);
```

---

## 5. API Endpoints

### 5.1 User Management
```
POST   /api/users/                  Create user
GET    /api/users/{user_id}         Get user details
POST   /api/financial-profile/      Create financial profile
```

### 5.2 Advisory Services
```
POST   /api/advisory/query          Get investment advice
POST   /api/advisory/feedback       Submit feedback
GET    /api/advisory/history/{id}   Get user history
```

### 5.3 Market Data
```
GET    /api/market/data             Get all market data
GET    /api/market/fd-rates         Get FD rates
```

### 5.4 Risk Assessment
```
POST   /api/risk/assess             Standalone risk assessment
```

---

## 6. Machine Learning Implementation

### 6.1 XGBoost Risk Model

**Training Data Generation**:
```python
Features:
- monthly_income: 20K - 200K range
- savings_ratio: 0.05 - 0.50 range
- investment_to_income_ratio: 0.05 - 0.40
- age_factor: 0.3 - 1.0 (normalized)
- investment_duration: 12 - 120 months

Labels:
- 0: Conservative (risk_score < 0.35)
- 1: Moderate (0.35 <= risk_score < 0.65)
- 2: Aggressive (risk_score >= 0.65)
```

**Model Configuration**:
```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    objective='multi:softmax',
    num_class=3
)
```

**Feature Importance**:
Extracted using XGBoost's `feature_importances_` attribute

### 6.2 SHAP Explainability

**Implementation**:
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# Extract importance for predicted class
feature_importance = dict(zip(
    feature_names,
    shap_values[predicted_class]
))
```

**Visualization**:
- Bar charts showing feature impact
- Positive values push toward higher risk
- Negative values push toward lower risk

---

## 7. LLM Integration

### 7.1 Prompt Engineering

**System Prompt**:
```
Role: Expert financial advisor
Specialization: Indian retail investments
Output: Structured JSON recommendations
Focus: Practical, actionable advice
```

**User Prompt Structure**:
```
1. User Financial Profile
   - Income, expenses, savings
   - Investment amount and duration
   - Financial goals
   - Risk profile

2. Market Conditions
   - FD rates, repo rate, inflation
   - Mutual fund returns
   - Gold prices
   - Stock indices

3. Request
   - Portfolio allocation
   - Expected returns
   - Detailed reasoning
   - Specific instruments
```

### 7.2 Response Handling

**JSON Parsing**:
```python
1. Extract JSON from markdown code blocks
2. Parse into structured dictionary
3. Fallback to rule-based if parsing fails
```

**Validation**:
```python
- Portfolio percentages sum to 100
- Expected returns are realistic
- Asset allocation is complete
```

---

## 8. Deployment Architecture

### 8.1 Local Development
```
Terminal 1: FastAPI (port 8000)
Terminal 2: Streamlit (port 8501)
Database: SQLite file in data/
Models: Stored in models/
Logs: Written to logs/
```

### 8.2 Docker Deployment
```yaml
Services:
  - backend (FastAPI)
  - frontend (Streamlit)

Networks:
  - financial-network (bridge)

Volumes:
  - data (persistent)
  - logs (persistent)
  - models (persistent)
```

---

## 9. Security Considerations

### 9.1 API Key Management
```
- Stored in .env file
- Never committed to version control
- Environment variables in Docker
```

### 9.2 Data Privacy
```
- Local SQLite database
- No external data transmission
- User data encrypted at rest (future)
```

### 9.3 Input Validation
```
- Pydantic schemas for all inputs
- Type checking and validation
- Range validation for numeric inputs
```

---

## 10. Testing Strategy

### 10.1 Unit Tests
```python
- Agent functionality
- Database operations
- API endpoints
- Model predictions
```

### 10.2 Integration Tests
```python
- End-to-end workflow
- Agent coordination
- API response validation
```

### 10.3 Manual Testing
```python
- UI functionality
- Different user profiles
- Edge cases
- Error handling
```

---

## 11. Performance Optimization

### 11.1 Caching
```python
- Market data (6-hour TTL)
- Model predictions
- Database queries
```

### 11.2 Asynchronous Operations
```python
- Async FastAPI endpoints
- Concurrent agent execution
- Non-blocking I/O
```

### 11.3 Database Optimization
```python
- Indexed fields (email, session_id)
- Query optimization
- Connection pooling
```

---

## 12. Future Enhancements

### 12.1 Technical
```
- Real-time RBI data integration
- Advanced portfolio optimization (Modern Portfolio Theory)
- Multi-language support
- Mobile app (React Native)
- GraphQL API
```

### 12.2 Features
```
- Backtesting capabilities
- Goal-based planning
- Tax optimization
- Automated rebalancing
- Integration with trading platforms
```

### 12.3 ML/AI
```
- Reinforcement learning for portfolio optimization
- Sentiment analysis from news
- Time-series forecasting
- Personalized risk models
- Advanced NLP for user queries
```

---

## 13. Dissertation Context

### 13.1 Research Contributions
```
1. Multi-agent architecture for finance
2. XGBoost-based risk profiling
3. LLM integration for personalization
4. Explainability framework
5. Memory and learning system
```

### 13.2 Novel Aspects
```
- Coordinated AI agents
- Context-aware recommendations
- Transparent decision-making
- Feedback learning loop
```

### 13.3 Evaluation Metrics
```
- Recommendation accuracy
- User satisfaction
- System performance
- Explainability quality
```

---

## 14. Code Quality

### 14.1 Standards
```
- PEP 8 compliance
- Type hints
- Docstrings
- Error handling
```

### 14.2 Documentation
```
- Inline comments
- API documentation
- Architecture diagrams
- User guides
```

### 14.3 Maintainability
```
- Modular design
- Separation of concerns
- Configuration management
- Logging and monitoring
```

---

## 15. References

### 15.1 Technologies
- FastAPI: https://fastapi.tiangolo.com/
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Streamlit: https://streamlit.io/
- OpenAI: https://platform.openai.com/

### 15.2 Research Papers
- XGBoost: "XGBoost: A Scalable Tree Boosting System"
- SHAP: "A Unified Approach to Interpreting Model Predictions"
- Multi-Agent Systems: Survey papers on agent coordination

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Author**: Ashutosh Kumar
