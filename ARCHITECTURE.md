# 🏗️ Financial Advisory Multi-Agent System - Architecture Documentation

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Multi-Agent Framework](#multi-agent-framework)
4. [LangGraph Orchestration](#langgraph-orchestration)
5. [LangChain Memory System](#langchain-memory-system)
6. [Dataset & Data Dictionary](#dataset--data-dictionary)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [XAI (Explainable AI) Implementation](#xai-explainable-ai-implementation)
9. [CI/CD Pipeline & Model Monitoring](#cicd-pipeline--model-monitoring)
10. [Database Architecture](#database-architecture)
11. [API Layer (FastAPI)](#api-layer-fastapi)
12. [Frontend (Streamlit)](#frontend-streamlit)
13. [MLflow Integration](#mlflow-integration)
14. [Data Flow](#data-flow)
15. [Technology Stack](#technology-stack)
16. [Deployment](#deployment)

---

## Executive Summary

The **Financial Advisory Multi-Agent System** is an AI-powered investment recommendation platform designed for Indian retail investors. It combines:

- **Multi-Agent Architecture**: Specialized agents for different tasks (Risk Assessment, Data Collection, Advisory, Explainability)
- **ML-Based Risk Profiling**: XGBoost classifier with 97.88% accuracy for predicting Conservative/Aggressive risk profiles
- **Explainable AI (XAI)**: SHAP and LIME for transparent model explanations, simplified by GPT-4o-mini
- **LangGraph Orchestration**: State-machine based workflow coordination
- **LangChain Memory**: Short-term (session) and long-term (SQL) memory persistence
- **Real-time Market Data**: Live FD rates, mutual fund returns, gold prices, and NIFTY data
- **CI/CD Pipeline**: Automated model monitoring with PSI/CSI drift detection and retraining

### Key Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy | 97.88% |
| ROC-AUC Score | 99.74% |
| Total Features | 112 (raw), 39 (selected) |
| Dataset Size | 125,000 customers |
| Risk Classes | Conservative, Aggressive |
| PSI Threshold | 0.25 (retrain trigger) |
| CSI Threshold | 0.25 (per feature) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FINANCIAL ADVISOR SYSTEM                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────┐         ┌─────────────────────────────────────────────┐   │
│   │   FRONTEND      │         │              BACKEND (FastAPI)              │   │
│   │   (Streamlit)   │ ◄─────► │                                             │   │
│   │   Port: 8501    │   REST  │   Port: 8000                                │   │
│   └─────────────────┘   API   │                                             │   │
│                               │   ┌─────────────────────────────────────┐   │   │
│                               │   │         ORCHESTRATOR                │   │   │
│                               │   │         (LangGraph)                 │   │   │
│                               │   │                                     │   │   │
│                               │   │   ┌───────┐ ┌───────┐ ┌───────┐     │   │   │
│                               │   │   │ Data  │ │ Risk  │ │Advisor│     │   │   │
│                               │   │   │ Agent │ │ Agent │ │ Agent │     │   │   │
│                               │   │   └───┬───┘ └───┬───┘ └───┬───┘     │   │   │
│                               │   │       │         │         │         │   │   │
│                               │   │   ┌───┴───┐ ┌───┴───┐ ┌───┴───┐     │   │   │
│                               │   │   │  XAI  │ │Memory │ │Session│     │   │   │
│                               │   │   │ Agent │ │ Agent │ │Memory │     │   │   │
│                               │   │   └───────┘ └───────┘ └───────┘     │   │   │
│                               │   └─────────────────────────────────────┘   │   │
│                               └─────────────────────────────────────────────┘   │
│                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                          DATABASES                                     │    │
│   │   ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐  │    │
│   │   │ risk_profiling.db│    │financial_advisor │    │   MLflow        │  │    │
│   │   │ (Customer Data)  │    │.db (Sessions)    │    │   (Tracking)    │  │    │
│   │   └──────────────────┘    └──────────────────┘    └─────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                       EXTERNAL SERVICES                                │    │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │    │
│   │   │ OpenAI   │  │ HDFC Bank│  │ yFinance │  │  Groww   │              │    │
│   │   │ GPT-4o   │  │ FD Rates │  │  NIFTY   │  │Gold Price│              │    │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘              │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Agent Framework

The system employs a **specialized multi-agent architecture** where each agent has distinct responsibilities:

### 1. 📊 Data Agent (`data_agent.py`)

**Purpose**: Fetches and aggregates real-time market data from various sources.

| Data Source | Information | Update Frequency |
|-------------|-------------|------------------|
| HDFC Bank Website | Fixed Deposit Rates (Regular & Senior) | 6 hours |
| yFinance API | NIFTY 50 Index Data | Real-time |
| Groww Website | Gold Prices (24K) | 6 hours |
| RBI Website | Repo Rate | Daily |
| Static Data | Mutual Fund Returns, Inflation | Configurable |

**Key Features**:
- Web scraping with BeautifulSoup
- Caching mechanism (6-hour TTL)
- Fallback to static data on failure
- Comprehensive error handling

### 2. 🎯 Risk Agent (`risk_agent.py`)

**Purpose**: Predicts customer risk profile (Conservative/Aggressive) using trained XGBoost model.

**Capabilities**:
- Loads trained model from `models/risk_model.json`
- Fetches customer data from SQLite database
- Prepares 39 selected features for prediction
- Uses artifact-based feature engineering
- Returns risk category with confidence scores

**Output Structure**:
```json
{
  "customer_id": 100001,
  "risk_category": "Conservative",
  "risk_score": 0.7279,
  "risk_probabilities": {
    "Conservative": 0.7279,
    "Aggressive": 0.2721
  },
  "prediction_confidence": 0.7279,
  "features_used": 39
}
```

### 3. 💡 Advisor Agent (`advisor_agent.py`)

**Purpose**: Generates personalized investment recommendations using GPT-4o-mini.

**Key Features**:
- Contextual prompt engineering with customer profile
- Portfolio allocation with percentages
- Asset-specific recommendations (FD, MF, Gold, etc.)
- Clarifying question generation for missing information
- Action steps for implementation

**Recommendation Structure**:
```json
{
  "portfolio": {
    "Fixed Deposits": 50,
    "Debt Mutual Funds": 30,
    "Gold": 10,
    "Equity Mutual Funds": 10
  },
  "expected_return_annual": 7.25,
  "risk_level": "Conservative",
  "reasoning": "...",
  "asset_allocation": {...},
  "action_steps": [...]
}
```

### 4. 🔍 XAI Agent (`xai_agent.py`)

**Purpose**: Provides explainability for ML predictions using SHAP, LIME, and LLM simplification.

**Explainability Methods**:

| Method | Type | Description |
|--------|------|-------------|
| SHAP (TreeExplainer) | Global & Local | Shapley values for feature attribution |
| LIME | Local | Instance-level decision rules |
| GPT-4o-mini | Simplification | User-friendly natural language explanations |

**Output Includes**:
- Feature contributions (sorted by impact)
- Top positive/negative factors
- SHAP values for visualization
- Simplified explanation for end users

### 5. 🧠 Memory Agent (`memory_agent.py`)

**Purpose**: Manages user preferences and session history for personalization.

**Capabilities**:
- Stores learned preferences
- Retrieves personalized context
- Tracks session history
- Manages feedback collection

### 6. 💾 Session Memory Manager (`session_memory.py`)

**Purpose**: LangChain-based memory management for conversational context.

**Memory Types**:

| Type | Implementation | Persistence |
|------|----------------|-------------|
| Short-term | `ConversationBufferWindowMemory` | In-memory (10 messages) |
| Long-term | `SQLChatMessageHistory` | SQLite database |

---

## LangGraph Orchestration

The **Orchestrator** uses LangGraph's `StateGraph` for workflow coordination.

### State Machine Workflow

```
                    ┌─────────────────────┐
                    │   ENTRY POINT       │
                    │  retrieve_context   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  check_missing_info │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        [missing info]   [proceed]              │
              │                │                │
              │     ┌──────────▼──────────┐     │
              │     │  fetch_market_data  │     │
              │     └──────────┬──────────┘     │
              │                │                │
              │     ┌──────────▼──────────┐     │
              │     │    assess_risk      │     │
              │     └──────────┬──────────┘     │
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │generate_recommendation│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        [is_question]    [explain]              │
              │                │                │
              │     ┌──────────▼──────────┐     │
              │     │explain_recommendation│     │
              │     └──────────┬──────────┘     │
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    store_session    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │        END          │
                    └─────────────────────┘
```

### AgentState TypedDict

```python
class AgentState(TypedDict):
    # Input fields
    user_id: int
    query: str
    investment_amount: float
    duration_months: int
    financial_goal: str
    risk_tolerance: str
    
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
    conversation_context: Dict
```

### Query Parsing

The orchestrator includes intelligent natural language parsing for investment queries:

```python
def _parse_investment_query(self, query: str) -> dict:
    """
    Parses natural language investment queries.
    Examples:
    - "I want to invest ₹8 lakhs in 4 years for my daughter's education"
    - "Invest 50000 for 6 months"
    
    Extracts: investment_amount, duration_months, financial_goal
    """
```

**Supported Patterns**:
- Amounts: "₹8 lakhs", "50000", "10L", "2 crores"
- Duration: "4 years", "6 months", "3 yrs"
- Goals: Education, Retirement, Marriage, Home, Emergency, Travel, Wealth

---

## LangChain Memory System

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SESSION MEMORY MANAGER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐  │
│  │   SHORT-TERM MEMORY     │   │     LONG-TERM MEMORY        │  │
│  │  (LangChain)            │   │     (SQL Database)          │  │
│  │                         │   │                             │  │
│  │  ConversationBuffer     │   │   SQLChatMessageHistory     │  │
│  │  WindowMemory           │   │                             │  │
│  │  (k=10 messages)        │   │   customer_sessions         │  │
│  │                         │   │   conversation_messages     │  │
│  │  - Fast access          │   │   interaction_metadata      │  │
│  │  - Session scoped       │   │   customer_preferences      │  │
│  │  - Automatic cleanup    │   │                             │  │
│  └─────────────────────────┘   └─────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  CAPABILITIES                               ││
│  │  • Create/restore sessions by session_id                    ││
│  │  • Store human and AI messages                              ││
│  │  • Get conversation context summary                         ││
│  │  • Persist to SQLite for durability                         ││
│  │  • Session end with feedback collection                     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Session ID Format
```
session_{customer_id}_{YYYYMMDD}_{HHMMSS}_{uuid4_prefix}
Example: session_100001_20251221_120516_f49544b2
```

---

## Dataset & Data Dictionary

### 📁 Data Source Overview

| Property | Value |
|----------|-------|
| **Database** | `data/risk_profiling.db` (SQLite) |
| **Table Name** | `risk_profiling_monthly_data` |
| **Total Records** | 125,000 customers |
| **Total Features** | 112 columns |
| **Snapshot Date** | Monthly snapshots (versioned) |
| **Customer ID Range** | 100000 - 224999 |

### 📊 Target Variable

| Feature | Type | Values | Distribution |
|---------|------|--------|--------------|
| `risk_profile` | Categorical | Conservative, Aggressive | 13% / 87% (Imbalanced) |

### 🔢 Feature Categories

The 112 features are organized into logical groups based on their domain:

---

#### 1️⃣ **Demographic Features** (7 features)

Basic customer information and life stage indicators.

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `customer_id` | Integer | Unique identifier | 100000, 100001 |
| `age` | Integer | Customer age in years | 25-65 |
| `gender` | Text | Gender (PII - removed) | Male, Female |
| `marital_status` | Text | Marital status (PII - removed) | Single, Married |
| `dependents` | Integer | Number of dependents | 0-6 |
| `education` | Text | Education level | Graduate, Post Graduate, Professional |
| `city_tier` | Text | City classification | Metro, Tier 1, Tier 2, Tier 3 |

---

#### 2️⃣ **Customer Segmentation Features** (3 features)

Banking relationship and customer classification.

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `customer_segment` | Text | Bank customer segment | Mass Market, Mass Affluent, Emerging Affluent, HNI, UHNI |
| `employment_status` | Text | Employment type | Salaried, Self-Employed, Business |
| `occupation_sector` | Text | Industry sector | IT, Banking, Healthcare, Other |

---

#### 3️⃣ **Income Features** (12 features)

All income-related metrics and income patterns.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `annual_income` | Float | Total annual income | INR |
| `monthly_income` | Float | Monthly income | INR |
| `quarterly_income` | Float | Quarterly income | INR |
| `semi_annual_income` | Float | Half-yearly income | INR |
| `total_annual_income` | Float | Comprehensive annual income | INR |
| `bonus_income` | Float | Annual bonus amount | INR |
| `other_income_sources` | Float | Additional income sources | INR |
| `income_per_family_member` | Float | Income per dependent | INR |
| `income_growth_rate` | Float | YoY income growth | Percentage |
| `income_volatility` | Float | Income stability measure | Score 0-1 |
| `disposable_income` | Float | After-tax available income | INR |
| `debt_free_income` | Float | Income after debt payments | INR |

---

#### 4️⃣ **Expense Features** (8 features)

Spending patterns and financial obligations.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `annual_expenses` | Float | Total yearly expenses | INR |
| `monthly_expenses` | Float | Monthly expenses | INR |
| `quarterly_expenses` | Float | Quarterly expenses | INR |
| `essential_expenses` | Float | Non-discretionary expenses | INR |
| `discretionary_spending` | Float | Lifestyle expenses | INR |
| `expense_to_income_ratio` | Float | Expenses as % of income | Ratio 0-1 |
| `insurance_premium_annual` | Float | Annual insurance cost | INR |
| `insurance_to_income_ratio` | Float | Insurance as % of income | Ratio 0-1 |

---

#### 5️⃣ **Savings Features** (5 features)

Savings patterns and adequacy metrics.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `annual_savings` | Float | Total annual savings | INR |
| `savings_per_month` | Float | Monthly savings amount | INR |
| `total_savings` | Float | Cumulative savings | INR |
| `savings_to_income_ratio` | Float | Savings rate | Ratio 0-1 |
| `emergency_fund_months` | Integer | Emergency fund coverage | Months |

---

#### 6️⃣ **Debt Features** (14 features)

Liability profile and debt management.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `total_debt` | Float | Total outstanding debt | INR |
| `home_loan_amount` | Float | Home loan outstanding | INR |
| `auto_loan` | Float | Vehicle loan outstanding | INR |
| `personal_loan` | Float | Personal loan outstanding | INR |
| `credit_card_outstanding` | Float | Credit card balance | INR |
| `secured_debt` | Float | Collateral-backed debt | INR |
| `unsecured_debt` | Float | Non-collateral debt | INR |
| `monthly_debt_payment` | Float | Monthly EMI outflow | INR |
| `debt_to_income_ratio` | Float | Debt as % of income | Ratio |
| `debt_to_asset_ratio` | Float | Debt as % of assets | Ratio |
| `debt_service_ratio` | Float | Debt payments vs income | Ratio |
| `debt_burden_score` | Float | Overall debt burden | Score 0-100 |
| `avg_debt_age_years` | Integer | Average age of debts | Years |
| `loan_to_value_ratio` | Float | LTV for secured loans | Ratio |

---

#### 7️⃣ **Investment Features** (14 features)

Investment portfolio and returns.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `investment_portfolio_value` | Float | Total investment value | INR |
| `equity_investments` | Float | Stocks and equity MFs | INR |
| `debt_investments` | Float | Bonds and debt MFs | INR |
| `mutual_fund_investments` | Float | Total MF holdings | INR |
| `fixed_deposits` | Float | FD holdings | INR |
| `gold_investments` | Float | Gold and gold bonds | INR |
| `real_estate_investments` | Float | Property investments | INR |
| `tax_saving_investments` | Float | 80C investments | INR |
| `monthly_sip_amount` | Float | SIP contribution | INR |
| `annual_investment_return` | Float | Yearly returns | INR |
| `investment_to_income_ratio` | Float | Investment as % income | Ratio |
| `investment_experience_years` | Integer | Years of investing | Years |
| `investment_horizon_years` | Integer | Target horizon | Years |
| `num_existing_investments` | Integer | Count of investments | Count |

---

#### 8️⃣ **Net Worth & Asset Features** (3 features)

Overall wealth metrics.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `net_worth` | Float | Assets minus liabilities | INR |
| `liquid_cash` | Float | Cash and equivalents | INR |
| `emergency_fund_adequacy` | Float | Emergency fund score | Score 0-100 |

---

#### 9️⃣ **Credit Health Features** (6 features)

Credit history and creditworthiness.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `credit_health_score` | Float | Credit score proxy | Score 0-100 |
| `credit_utilization` | Float | Credit card utilization | Percentage |
| `credit_card_payment_history_score` | Float | Payment history | Score 0-100 |
| `loan_repayment_track_record` | Float | Loan repayment score | Score 0-100 |
| `num_credit_cards` | Integer | Active credit cards | Count |
| `overdraft_usage_times` | Integer | Overdraft frequency | Count |

---

#### 🔟 **Banking Behavior Features** (12 features)

Transaction patterns and banking engagement.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `num_bank_accounts` | Integer | Active bank accounts | Count |
| `avg_monthly_balance` | Float | Average balance | INR |
| `avg_min_balance_maintained` | Float | Minimum balance | INR |
| `avg_transaction_value` | Float | Average transaction | INR |
| `num_monthly_transactions` | Integer | Transaction count | Count |
| `num_digital_transactions` | Integer | Online transactions | Count |
| `digital_banking_usage_pct` | Float | Digital adoption | Percentage |
| `online_payment_frequency` | Integer | Online payment count | Count |
| `bill_payment_frequency` | Integer | Bill payments | Count |
| `cash_withdrawal_frequency` | Integer | ATM withdrawals | Count |
| `international_transaction_count` | Integer | Forex transactions | Count |
| `relationship_vintage_years` | Integer | Years with bank | Years |

---

#### 1️⃣1️⃣ **Cash Flow Features** (6 features)

Monthly money movement patterns.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `monthly_credit_amount_1m` | Float | Last month credits | INR |
| `monthly_debit_amount_1m` | Float | Last month debits | INR |
| `avg_monthly_credit_6m` | Float | 6-month avg credits | INR |
| `avg_monthly_debit_6m` | Float | 6-month avg debits | INR |
| `debit_to_credit_ratio_1m` | Float | D/C ratio (1 month) | Ratio |
| `debit_to_credit_ratio_6m` | Float | D/C ratio (6 months) | Ratio |

---

#### 1️⃣2️⃣ **Insurance Features** (5 features)

Insurance coverage adequacy.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `life_insurance_coverage` | Float | Life cover amount | INR |
| `health_insurance_coverage` | Float | Health cover amount | INR |
| `property_insurance_coverage` | Float | Property cover | INR |
| `total_insurance_coverage` | Float | Total coverage | INR |
| `insurance_adequacy_ratio` | Float | Coverage adequacy | Ratio |

---

#### 1️⃣3️⃣ **Risk & Behavior Scores** (8 features)

Behavioral and risk indicators.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `risk_appetite_score` | Float | Risk tolerance level | Score 0-100 |
| `investment_risk_score` | Float | Portfolio risk level | Score 0-100 |
| `financial_literacy_score` | Float | Financial knowledge | Score 0-100 |
| `financial_planning_score` | Float | Planning discipline | Score 0-100 |
| `financial_stress_score` | Float | Financial stress level | Score 0-100 |
| `budgeting_discipline_score` | Float | Budgeting habits | Score 0-100 |
| `retirement_planning_score` | Float | Retirement readiness | Score 0-100 |
| `tax_planning_efficiency` | Float | Tax optimization | Score 0-100 |

---

#### 1️⃣4️⃣ **Portfolio Metrics** (4 features)

Investment portfolio characteristics.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `portfolio_diversification_score` | Float | Diversification level | Score 0-100 |
| `portfolio_turnover_ratio` | Float | Trading frequency | Ratio |
| `avg_holding_period_months` | Integer | Avg holding time | Months |
| `years_of_market_experience` | Integer | Market experience | Years |

---

#### 1️⃣5️⃣ **Financial Goals Features** (2 features)

Goal tracking.

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `num_financial_goals` | Integer | Active goals count | Count |
| `years_employed` | Integer | Employment tenure | Years |

---

#### 1️⃣6️⃣ **Data Management Features** (2 features)

Dataset versioning and split tracking.

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `data_split` | Text | Dataset partition | Train, Validation, Test |
| `snapshot_date` | Text | Data version date | YYYY-MM-DD |

---

### 📈 Data Split Distribution

| Split | Records | Percentage |
|-------|---------|------------|
| Training | 87,500 | 70% |
| Validation | 18,750 | 15% |
| Test (OOT) | 18,750 | 15% |

### 🎯 Risk Profile Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Aggressive | 108,865 | 87.09% |
| Conservative | 16,135 | 12.91% |

> **Note**: The dataset is imbalanced. The ML pipeline uses stratified sampling and ROC-AUC as the primary metric to handle this.

---

## Machine Learning Pipeline

### 📊 Data Exploration (EDA)

Key insights from exploratory data analysis:

| Observation | Details |
|-------------|---------|
| Missing Values | None (clean dataset) |
| Data Types | 22 Integer, 80 Float, 10 Text |
| Outliers | Detected in income and debt features |
| Correlations | High multicollinearity in income variants |
| Class Imbalance | 87% Aggressive vs 13% Conservative |

### 🧹 Data Preprocessing

#### 1. PII Removal
```python
# Features removed to prevent bias
pii_features = ['gender', 'marital_status']
df = df.drop(columns=pii_features)
```

#### 2. Multicollinearity Check
Features with correlation > 0.9 flagged:
- `monthly_income` ↔ `annual_income` (0.99)
- `quarterly_income` ↔ `annual_income` (0.99)

#### 3. Outlier Detection & Removal
```python
# Negative outliers removed
# Z-score method for extreme values
# Capping applied for income features
```

### 🔧 Feature Engineering

#### Derived Features Created

| Feature | Formula | Description | Type |
|---------|---------|-------------|------|
| `income_per_dependent` | `annual_income / (dependents + 1)` | Income per family member | Ratio |
| `debt_coverage_ratio` | `annual_income / (total_debt + 1)` | Debt payment capability | Ratio |
| `investment_efficiency` | `annual_investment_return / (investment_portfolio_value + 1)` | Return on investment | Ratio |

#### Artifact-Based Feature Engineering

All transformations are stored in `models/feature_engineering_config.json`:

```json
{
  "version": "1.0",
  "derived_features": [
    {
      "name": "income_per_dependent",
      "formula": "annual_income / (dependents + 1)",
      "required_features": ["annual_income", "dependents"],
      "type": "ratio"
    }
  ],
  "feature_transformations": {
    "income_per_dependent": {
      "numerator": "annual_income",
      "denominator": "dependents",
      "offset": 1,
      "operation": "divide"
    }
  }
}
```

#### Categorical Encoding

| Feature | Encoding Type | Classes |
|---------|---------------|---------|
| `customer_segment` | LabelEncoder | Emerging Affluent, HNI, Mass Affluent, Mass Market, UHNI |
| `education` | LabelEncoder | Graduate, Others, Post Graduate, Professional |

Encoders saved to `models/categorical_encoders.pkl` for consistent inference.

### 📈 Feature Selection

**Method Used**: Univariate Selection (ANOVA F-test)

| Method Compared | Features Selected | Validation Accuracy | ROC-AUC |
|-----------------|-------------------|---------------------|---------|
| **Univariate (F-test)** | 39 | **97.87%** ✓ | **0.9973** |
| Mutual Information | 39 | 97.65% | 0.9968 |
| RFE (Recursive) | 39 | 97.45% | 0.9962 |
| Tree-based | 39 | 97.72% | 0.9970 |

#### Selected Features (39 Total)

The model uses these 39 features selected through univariate feature selection:

```
age, annual_income, annual_investment_return, annual_savings, 
avg_holding_period_months, budgeting_discipline_score, 
credit_utilization, customer_segment, debt_burden_score, 
debt_to_income_ratio, disposable_income, education, 
emergency_fund_adequacy, financial_literacy_score, 
financial_planning_score, financial_stress_score, 
income_per_dependent, investment_efficiency, 
investment_experience_years, investment_horizon_years, 
investment_portfolio_value, investment_risk_score, 
investment_to_income_ratio, monthly_sip_amount, net_worth, 
num_existing_investments, other_income_sources, 
portfolio_diversification_score, portfolio_turnover_ratio, 
retirement_planning_score, risk_appetite_score, 
savings_to_income_ratio, tax_planning_efficiency, 
total_debt, total_savings, years_employed, 
years_of_market_experience, ... (+ derived features)
```

### 🎛️ Hyperparameter Tuning

**Method**: RandomizedSearchCV with 5-fold Stratified Cross-Validation

**Search Strategy**: 50 combinations with early stopping

#### Hyperparameter Search Space

| Parameter | Search Range | Best Value |
|-----------|--------------|------------|
| `learning_rate` | [0.01, 0.1, 0.2, 0.3] | **0.3** |
| `max_depth` | [3, 5, 7, 10] | **3** |
| `n_estimators` | [100, 200, 300, 500] | **278** |
| `subsample` | [0.7, 0.8, 0.9, 1.0] | **0.9** |
| `colsample_bytree` | [0.7, 0.8, 0.9, 1.0] | **0.7** |
| `min_child_weight` | [1, 3, 5] | **1** |
| `gamma` | [0, 0.1, 0.2] | **0.2** |
| `reg_alpha` | [0, 0.1, 0.5, 1.0] | **0.1** |
| `reg_lambda` | [0, 0.1, 0.5, 1.0] | **0** |
| `scale_pos_weight` | [1, 3, 5] | **1** |

**Early Stopping**: 25 rounds on validation ROC-AUC

### 🎯 Model Performance

#### Classification Metrics

| Metric | Training | Validation | Test (OOT) |
|--------|----------|------------|------------|
| **Accuracy** | 99.11% | 97.87% | **97.88%** |
| **Precision** | 99.04% | 97.84% | 97.86% |
| **Recall** | 99.05% | 97.87% | 97.88% |
| **F1-Score** | 99.04% | 97.83% | 97.84% |
| **ROC-AUC** | 99.95% | 99.73% | **99.74%** |

#### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Aggressive | 98.47% | 99.36% | 98.91% | 16,330 |
| Conservative | 95.42% | 89.59% | 92.41% | 2,420 |

#### Confusion Matrix (Test Set)

|  | Predicted Aggressive | Predicted Conservative |
|--|---------------------|----------------------|
| **Actual Aggressive** | 16,225 (TN) | 105 (FP) |
| **Actual Conservative** | 252 (FN) | 2,168 (TP) |

### 🏆 Feature Importance (Top 20)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `debt_burden_score` | 14.27% | Debt |
| 2 | `net_worth` | 9.76% | Net Worth |
| 3 | `annual_income` | 8.42% | Income |
| 4 | `customer_segment` | 6.28% | Segmentation |
| 5 | `risk_appetite_score` | 6.11% | Risk Score |
| 6 | `age` | 5.91% | Demographic |
| 7 | `financial_literacy_score` | 5.77% | Risk Score |
| 8 | `investment_horizon_years` | 5.76% | Investment |
| 9 | `investment_portfolio_value` | 5.42% | Investment |
| 10 | `investment_experience_years` | 4.08% | Investment |
| 11 | `education` | 3.77% | Demographic |
| 12 | `debt_to_income_ratio` | 3.51% | Debt |
| 13 | `other_income_sources` | 3.50% | Income |
| 14 | `years_employed` | 2.75% | Goals |
| 15 | `investment_to_income_ratio` | 2.06% | Investment |
| 16 | `savings_to_income_ratio` | 1.92% | Savings |
| 17 | `emergency_fund_adequacy` | 1.85% | Savings |
| 18 | `financial_planning_score` | 1.72% | Risk Score |
| 19 | `portfolio_diversification_score` | 1.58% | Portfolio |
| 20 | `total_debt` | 1.42% | Debt |

### 📦 Model Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| `risk_model.json` | `models/` | XGBoost model (JSON format) |
| `risk_profiling_model.pkl` | `models/` | Complete model (Pickle) |
| `selected_features.json` | `models/` | List of 39 selected features |
| `label_encoder.pkl` | `models/` | Target variable encoder |
| `categorical_encoders.pkl` | `models/` | Categorical feature encoders |
| `best_params.json` | `models/` | Optimized hyperparameters |
| `evaluation_metrics.json` | `models/` | All performance metrics |
| `feature_engineering_config.json` | `models/` | Feature transformation config |
| `model_metadata.json` | `models/` | Training metadata |

---

## XAI (Explainable AI) Implementation

### SHAP Analysis

```
┌──────────────────────────────────────────────────────────────┐
│                    SHAP TreeExplainer                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   XGBoost Model ───► TreeExplainer ───► SHAP Values         │
│                                                              │
│   For each prediction:                                       │
│   • Base value (expected prediction)                         │
│   • Feature contributions (positive/negative)                │
│   • Sorted by impact magnitude                               │
│                                                              │
│   Output:                                                    │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ {                                                      │ │
│   │   "base_value": -2.289,                                │ │
│   │   "feature_contributions": [                           │ │
│   │     {"feature": "age", "shap_value": -5.17, ...},     │ │
│   │     {"feature": "financial_literacy_score", ...}       │ │
│   │   ],                                                   │ │
│   │   "top_positive_factors": [...],                       │ │
│   │   "top_negative_factors": [...]                        │ │
│   │ }                                                      │ │
│   └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### LIME Analysis

```
┌──────────────────────────────────────────────────────────────┐
│              LIME TabularExplainer                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Instance ───► Local Surrogate Model ───► Decision Rules   │
│                                                              │
│   For each prediction:                                       │
│   • Generates interpretable rules                            │
│   • Shows feature thresholds                                 │
│   • Weights indicate support/opposition                      │
│                                                              │
│   Output:                                                    │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ {                                                      │ │
│   │   "predicted_class": "Conservative",                   │ │
│   │   "prediction_probability": 0.7279,                    │ │
│   │   "feature_rules": [                                   │ │
│   │     {"rule": "age <= 32.00", "weight": 0.15, ...},    │ │
│   │     {"rule": "income > 500000", "weight": -0.08, ...} │ │
│   │   ]                                                    │ │
│   │ }                                                      │ │
│   └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### LLM Simplification

```
┌──────────────────────────────────────────────────────────────┐
│                 GPT-4o-mini Simplification                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Technical Explanation ───► GPT-4o-mini ───► User-Friendly │
│                                                              │
│   Input:                                                     │
│   • SHAP top factors                                         │
│   • LIME decision rules                                      │
│   • Customer profile                                         │
│   • Risk prediction                                          │
│                                                              │
│   Output Example:                                            │
│   ┌────────────────────────────────────────────────────────┐ │
│   │ "Based on your profile, you've been classified as      │ │
│   │  Conservative. Here's why:                              │ │
│   │  • Your debt burden is higher than average              │ │
│   │  • Your age (32) suggests caution for long-term goals   │ │
│   │  • Your investment horizon of 7 years is favorable      │ │
│   │  This means we'll focus on safer investments..."        │ │
│   └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### XAI Data Flow

```
Customer ID
     │
     ▼
┌─────────────────┐
│   RiskAgent     │──► ML Prediction (Conservative/Aggressive)
│  predict()      │         │
└─────────────────┘         │
                            ▼
              ┌─────────────────────────────┐
              │         XAI Agent           │
              │                             │
              │  ┌─────────────────────┐    │
              │  │ SHAP TreeExplainer  │────┼──► Feature Contributions
              │  └─────────────────────┘    │
              │                             │
              │  ┌─────────────────────┐    │
              │  │ LIME Explainer      │────┼──► Decision Rules
              │  └─────────────────────┘    │
              │                             │
              │  ┌─────────────────────┐    │
              │  │ GPT-4o-mini         │────┼──► Simplified Explanation
              │  └─────────────────────┘    │
              └─────────────────────────────┘
                            │
                            ▼
              Combined Human-Readable Explanation
```

---

## CI/CD Pipeline & Model Monitoring

The system implements a comprehensive CI/CD pipeline for automated model monitoring, drift detection, and retraining.

### 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CI/CD PIPELINE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        SCHEDULED TRIGGER                                │   │
│   │                   (Monthly - Default | On-demand)                       │   │
│   └────────────────────────────┬────────────────────────────────────────────┘   │
│                                │                                                │
│                                ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  STAGE 1: DRIFT DETECTION                                               │   │
│   │  ┌──────────────────────┐    ┌──────────────────────┐                   │   │
│   │  │   PSI Analysis       │    │   CSI Analysis       │                   │   │
│   │  │   (Population        │    │   (Characteristic    │                   │   │
│   │  │   Stability Index)   │    │   Stability Index)   │                   │   │
│   │  │                      │    │                      │                   │   │
│   │  │   Threshold: 0.25    │    │   Threshold: 0.25    │                   │   │
│   │  └──────────────────────┘    └──────────────────────┘                   │   │
│   └────────────────────────────┬────────────────────────────────────────────┘   │
│                                │                                                │
│              ┌─────────────────┼─────────────────┐                              │
│              │                 │                 │                              │
│         [No Drift]       [PSI ≥ 0.25]    [CSI ≥ 0.25 on                        │
│              │           [Major Drift]    3+ features]                          │
│              ▼                 │                 │                              │
│        ┌───────────┐          └────────┬────────┘                              │
│        │  STABLE   │                   │                                        │
│        │  No Action│                   ▼                                        │
│        └───────────┘   ┌────────────────────────────────────────────────────┐   │
│                        │  STAGE 2: AUTOMATED RETRAINING                     │   │
│                        │  • Load recent data from database                  │   │
│                        │  • Apply feature engineering artifacts             │   │
│                        │  • Train new XGBoost model                         │   │
│                        │  • Evaluate on validation/test sets                │   │
│                        └────────────────────────┬───────────────────────────┘   │
│                                                 │                               │
│                                                 ▼                               │
│                        ┌────────────────────────────────────────────────────┐   │
│                        │  STAGE 3: MODEL VALIDATION                         │   │
│                        │  • Compare with current production model           │   │
│                        │  • Check accuracy >= current - 1%                  │   │
│                        │  • Check ROC-AUC >= current - 1%                   │   │
│                        └────────────────────────┬───────────────────────────┘   │
│                                                 │                               │
│              ┌──────────────────────────────────┼───────────────────────┐       │
│              │                                  │                       │       │
│         [Validation Failed]              [Validation Passed]            │       │
│              │                                  │                       │       │
│              ▼                                  ▼                       │       │
│        ┌───────────┐               ┌────────────────────────────────┐   │       │
│        │  REJECT   │               │  STAGE 4: DEPLOYMENT           │   │       │
│        │  Keep     │               │  • Backup current model        │   │       │
│        │  Current  │               │  • Promote new model           │   │       │
│        └───────────┘               │  • Update artifacts            │   │       │
│                                    │  • Log to MLflow               │   │       │
│                                    │  • Create GitHub release       │   │       │
│                                    └────────────────┬───────────────┘   │       │
│                                                     │                   │       │
│                                                     ▼                   │       │
│                                    ┌────────────────────────────────┐   │       │
│                                    │  STAGE 5: NOTIFICATION         │   │       │
│                                    │  • Slack/Email alerts          │   │       │
│                                    │  • GitHub Actions summary      │   │       │
│                                    │  • Audit logging               │   │       │
│                                    └────────────────────────────────┘   │       │
│                                                                         │       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 📊 Population Stability Index (PSI)

PSI measures the shift in the distribution of a key variable between baseline (training) and current (production) data.

**Formula**:
$$PSI = \sum_{i=1}^{n} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$

Where:
- $A_i$ = Actual percentage in bin $i$
- $E_i$ = Expected (baseline) percentage in bin $i$

**Interpretation**:

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant drift | ✅ No action required |
| 0.10 - 0.25 | Moderate drift | ⚠️ Monitor closely |
| ≥ 0.25 | Significant drift | 🚨 Retrain model |

### 📈 Characteristic Stability Index (CSI)

CSI applies PSI analysis to each individual input feature to identify which features have drifted.

**Calculation**: Same formula as PSI, applied per feature.

**Trigger Conditions**:
- Any single feature with CSI ≥ 0.25
- Three or more features with CSI ≥ 0.10

**Top Monitored Features** (by importance):

| Feature | Category | Monitoring Priority |
|---------|----------|---------------------|
| `debt_burden_score` | Debt | High |
| `net_worth` | Net Worth | High |
| `annual_income` | Income | High |
| `risk_appetite_score` | Risk Score | High |
| `age` | Demographic | Medium |
| `investment_portfolio_value` | Investment | Medium |

### 🔧 MLOps Module Structure

```
mlops/
├── __init__.py                 # Module exports
├── drift_detector.py           # PSI/CSI calculation
├── model_retrainer.py          # Automated retraining pipeline
├── monitoring_scheduler.py     # Scheduled monitoring
└── mlops_cli.py               # Command-line interface
```

### 📄 Module Details

#### 1. Drift Detector (`drift_detector.py`)

```python
class DriftDetector:
    """
    Detects model drift using PSI and CSI metrics.
    
    Thresholds:
    - PSI_THRESHOLD_STABLE = 0.10
    - PSI_THRESHOLD_MODERATE = 0.25
    - CSI_THRESHOLD = 0.25
    """
    
    def detect_drift(
        self,
        baseline_snapshot: str = None,
        current_snapshot: str = None
    ) -> DriftReport:
        """
        Returns DriftReport with:
        - psi_score
        - psi_status ('stable', 'moderate', 'drift')
        - csi_scores (per feature)
        - csi_drifted_features
        - overall_drift_detected
        - recommendation
        """
```

**Output Example**:
```json
{
  "timestamp": "2025-12-21T18:30:00",
  "psi_score": 0.2847,
  "psi_status": "drift",
  "csi_drifted_features": ["debt_burden_score", "annual_income", "net_worth"],
  "overall_drift_detected": true,
  "recommendation": "URGENT: Significant drift detected. Triggering retraining."
}
```

#### 2. Model Retrainer (`model_retrainer.py`)

```python
class ModelRetrainer:
    """
    Automated model retraining pipeline.
    
    Steps:
    1. Load recent data from database
    2. Apply feature engineering from artifacts
    3. Train XGBoost with saved hyperparameters
    4. Evaluate on train/val/test sets
    5. Compare with current production model
    6. Promote if performance acceptable
    7. Log to MLflow
    """
    
    def retrain(
        self,
        snapshot_date: str = None,
        force: bool = False
    ) -> Dict:
        """
        Returns:
        - status: 'success', 'completed_no_promotion', 'failed'
        - metrics: train/val/test performance
        - comparison: accuracy_change, roc_auc_change
        - model_promoted: bool
        """
```

#### 3. Monitoring Scheduler (`monitoring_scheduler.py`)

```python
class MonitoringScheduler:
    """
    Orchestrates scheduled drift detection and retraining.
    
    Features:
    - Configurable intervals (daily, weekly, monthly)
    - Automatic retraining trigger
    - Notification callbacks
    - Comprehensive logging
    """
    
    def schedule_monitoring(
        self,
        interval: str = "monthly",
        time_of_day: str = "02:00"
    ):
        """Schedule regular drift checks."""
```

### 🖥️ CLI Usage

```bash
# Run drift detection
python mlops/mlops_cli.py drift-check
python mlops/mlops_cli.py drift-check --baseline 2025-11-30 --current 2025-12-31

# Trigger model retraining
python mlops/mlops_cli.py retrain
python mlops/mlops_cli.py retrain --force

# Run full CI/CD pipeline
python mlops/mlops_cli.py pipeline
python mlops/mlops_cli.py pipeline --force

# Start scheduled monitoring
python mlops/mlops_cli.py schedule --interval monthly --time 02:00

# Check monitoring status
python mlops/mlops_cli.py status
```

### 🔄 GitHub Actions Workflow

The pipeline is automated via GitHub Actions (`.github/workflows/model-cicd.yml`):

**Triggers**:
- **Scheduled**: 1st of every month at 2 AM UTC
- **Manual**: Via workflow_dispatch
- **On Push**: When `data/risk_profiling.db` or `mlops/**` changes

**Stages**:

| Stage | Description | Condition |
|-------|-------------|-----------|
| `drift-detection` | Calculate PSI/CSI | Always runs |
| `model-retraining` | Train new model | If drift detected or forced |
| `model-validation` | Validate performance | If model trained |
| `deploy-model` | Commit & release | If validation passed |
| `notify` | Send notifications | Always (summary) |

**Artifacts Produced**:
- Drift reports (30-day retention)
- Model artifacts (90-day retention)
- Retraining reports (30-day retention)
- GitHub release with model version

### 📊 Monitoring Dashboard

Drift reports are saved to `logs/drift_reports/` and include:

```json
{
  "timestamp": "2025-12-21T18:30:00",
  "psi_score": 0.1234,
  "psi_status": "moderate",
  "csi_scores": {
    "debt_burden_score": 0.0892,
    "net_worth": 0.0654,
    "annual_income": 0.0543,
    ...
  },
  "csi_drifted_features": [],
  "overall_drift_detected": false,
  "recommendation": "Monitor closely. No immediate action required.",
  "baseline_snapshot": "2025-11-30",
  "current_snapshot": "2025-12-31",
  "baseline_records": 87500,
  "current_records": 125000
}
```

### 🔔 Notification Integration

Supports multiple notification channels:

| Channel | Configuration |
|---------|--------------|
| Slack | Set `SLACK_WEBHOOK_URL` secret |
| Email | Configure SMTP settings |
| GitHub | Automatic job summaries |

### 📈 Retraining Validation Thresholds

New models are promoted only if they meet these criteria:

| Metric | Minimum Threshold |
|--------|-------------------|
| Accuracy | >= current_accuracy - 1% |
| ROC-AUC | >= current_roc_auc - 1% |
| Validation Accuracy | >= 90% |
| Validation ROC-AUC | >= 95% |

### 📁 Generated Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| Drift Reports | `logs/drift_reports/` | JSON drift analysis |
| Retraining Reports | `logs/retraining_reports/` | Training results |
| Model Backups | `models/backup_YYYYMMDD/` | Previous model versions |
| Pipeline Logs | `logs/cicd_pipeline.jsonl` | Pipeline execution log |

---

## Database Architecture

### Database 1: Risk Profiling (`data/risk_profiling.db`)

**Purpose**: Customer financial data for ML predictions

| Table | Description | Records |
|-------|-------------|---------|
| `risk_profiling_monthly_data` | Customer financial snapshots | 125,000 |

**Schema Overview**:
- 112 columns total
- Primary Key: `customer_id`
- Partition Key: `data_split`
- Version Key: `snapshot_date`

### Database 2: Financial Advisor (`data/financial_advisor.db`)

**Purpose**: Session management, user preferences, market data cache

| Table | Description |
|-------|-------------|
| `users` | User profiles |
| `financial_profiles` | User financial information |
| `sessions` | User interaction sessions |
| `user_preferences` | Learned preferences |
| `market_data` | Cached market data |
| `customer_sessions` | Customer advisory sessions |
| `conversation_messages` | Chat history storage |
| `interaction_metadata` | Learning data |
| `customer_preferences` | Customer-specific preferences |

### Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐
│     users       │       │ customer_sessions│
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ name            │       │ customer_id     │
│ email           │       │ session_id (UK) │
│ age             │       │ started_at      │
│ occupation      │       │ ended_at        │
└────────┬────────┘       │ feedback        │
         │                │ risk_profile    │
         │                └────────┬────────┘
         │                         │
    ┌────▼────────────┐    ┌───────▼───────────┐
    │financial_profiles│    │conversation_msgs │
    ├─────────────────┤    ├───────────────────┤
    │ id (PK)         │    │ id (PK)           │
    │ user_id (FK)    │    │ session_id (FK)   │
    │ monthly_income  │    │ role              │
    │ investment_amt  │    │ content           │
    │ risk_tolerance  │    │ message_metadata  │
    └─────────────────┘    └───────────────────┘
```

---

## API Layer (FastAPI)

### Base Configuration

| Setting | Value |
|---------|-------|
| Framework | FastAPI |
| Host | 0.0.0.0 |
| Port | 8000 |
| Auto-reload | Enabled (dev) |
| Documentation | `/docs` (Swagger), `/redoc` |

### API Endpoints

#### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with system info |
| `/health` | GET | Health check |

#### Customer Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/customer/search/{prefix}` | GET | Search customer IDs |
| `/api/customer/{customer_id}` | GET | Get customer data |

#### Advisory Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/advisory/customer/{customer_id}` | POST | Get investment advice |
| `/api/risk-prediction/{customer_id}` | GET | Get ML risk prediction |
| `/api/risk-explain/{customer_id}` | GET | Get XAI explanation |

#### Session Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/session/{session_id}/history` | GET | Get session history |
| `/api/session/customer/{customer_id}` | GET | Get customer sessions |
| `/api/session/{session_id}/end` | POST | End session with feedback |

#### Market Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/market/data` | GET | Get all market data |

---

## Frontend (Streamlit)

### Configuration

| Setting | Value |
|---------|-------|
| Framework | Streamlit |
| Port | 8501 |
| Layout | Wide |
| Theme | Custom CSS |

### UI Components

#### Tab 1: 💡 Get Advice

- **Customer ID Selector**: Search and select customers
- **Chat Interface**: Natural language query input
- **Form Mode**: Structured investment details
- **Results Display**:
  - Risk profile with confidence
  - Portfolio allocation chart
  - Asset allocation details
  - Action steps
  - XAI explanation panel

#### Tab 2: 📊 Market Data

- **Real-time Metrics**: Repo rate, inflation, gold price, NIFTY
- **FD Rate Table**: Tenure-wise rates (Regular & Senior)
- **Mutual Fund Returns**: Debt, Equity, Hybrid categories
- **Market Indicators**: 52-week high/low

#### Tab 3: 📜 History

- **Customer Profile Summary**: Risk profile, income, net worth
- **Financial Snapshot**: Age, dependents, debt, portfolio
- **Latest Recommendation**: Portfolio, reasoning, action steps
- **XAI Summary**: Simplified explanation, methods used

### Visualizations

| Chart Type | Library | Use Case |
|------------|---------|----------|
| Pie Chart | Plotly | Portfolio allocation |
| Bar Chart | Plotly | Risk probabilities |
| Waterfall | Plotly | SHAP feature importance |
| Metrics | Streamlit | Key financial indicators |

---

## MLflow Integration

### Configuration

```python
# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment
mlflow.set_experiment("risk_profiling_ml_pipeline")
```

### Tracked Experiments

| Run Type | Description | Key Metrics |
|----------|-------------|-------------|
| `feature_selection_comparison` | Compare 4 FS methods | val_accuracy, val_roc_auc |
| `hyperparameter_tuning` | 50 trial combinations | val_roc_auc, val_f1_score |
| `final_model_evaluation` | Train/Val/Test metrics | All classification metrics |
| `model_registration` | Production model | Model artifacts |

### Model Registry

| Property | Value |
|----------|-------|
| Model Name | `RiskProfilingXGBoost` |
| Model Type | XGBoost Classifier |
| Artifact Path | `xgboost_model` |
| Tags | feature_selection, test_accuracy, test_roc_auc |

### Viewing MLflow UI

```bash
cd /Users/ashutosh/BITS/Financial-Advisor
mlflow ui
# Access: http://localhost:5000
```

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                               │
└──────────────────────────────────────────────────────────────────────────────┘

User Query (Frontend)
        │
        ▼
┌───────────────────┐
│   FastAPI Route   │ ◄───── Session Memory Check
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Orchestrator    │ ◄───── LangGraph State Machine
└─────────┬─────────┘        ◄───── Query Parsing (amount, duration, goal)
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│  Data   │ │  Risk   │ ◄───── SQLite: risk_profiling.db
│  Agent  │ │  Agent  │ ◄───── XGBoost Model (39 features)
└────┬────┘ └────┬────┘
     │           │
     │           ▼
     │    ┌─────────────┐
     │    │  XAI Agent  │ ◄───── SHAP + LIME + GPT-4o-mini
     │    └──────┬──────┘
     │           │
     ▼           ▼
┌──────────────────────┐
│    Advisor Agent     │ ◄───── GPT-4o-mini
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Response Assembly  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Session Storage    │ ◄───── SQLite: financial_advisor.db
└──────────┬───────────┘
           │
           ▼
      JSON Response ───► Frontend Display
```

---

## Technology Stack

### Backend

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | FastAPI | 0.121.3 |
| Server | Uvicorn | 0.38.0 |
| Validation | Pydantic | 2.12.4 |
| ORM | SQLAlchemy | 2.0.44 |
| ML Framework | XGBoost | 3.1.2 |
| ML Utilities | Scikit-learn | 1.7.2 |
| XAI - SHAP | SHAP | 0.50.0 |
| XAI - LIME | LIME | 0.2.0.1 |
| LLM Integration | OpenAI | ≥1.58.1 |
| Orchestration | LangGraph | 0.2.59 |
| Memory | LangChain | 0.3.13 |
| Data Fetching | yfinance | 0.2.66 |
| Web Scraping | BeautifulSoup4 | 4.14.2 |
| Logging | Loguru | 0.7.3 |

### Frontend

| Component | Technology | Version |
|-----------|------------|---------|
| Framework | Streamlit | 1.51.0 |
| Charts | Plotly | 6.5.0 |
| Data Processing | Pandas | 2.3.3 |
| HTTP Client | Requests | 2.32.5 |

### MLOps

| Component | Technology | Version |
|-----------|------------|---------|
| Experiment Tracking | MLflow | Latest |
| Model Registry | MLflow Model Registry | - |
| Drift Detection | PSI/CSI (Custom) | - |
| CI/CD | GitHub Actions | - |
| Scheduling | schedule (Python) | 1.2.0 |

### Infrastructure

| Component | Technology |
|-----------|------------|
| Containerization | Docker |
| Orchestration | Docker Compose |
| Database | SQLite (local) |
| Environment | Python 3.11+ |
| CI/CD | GitHub Actions |

---

## Deployment

### Local Development

```bash
# Clone and setup
git clone <repository>
cd Financial-Advisor

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Start backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (new terminal)
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini | Yes |
| `DATABASE_URL` | SQLite connection string | No (default provided) |
| `LOG_LEVEL` | Logging level (INFO, DEBUG) | No |

---

## 📚 Related Documentation

| Document | Description |
|----------|-------------|
| `README.md` | Project overview and quick start |
| `QUICKSTART.md` | Detailed setup instructions |

---

## 🔗 References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

*Last Updated: December 21, 2025*
*Version: 2.0.0*
