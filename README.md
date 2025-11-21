# Multi-Agent Financial Advisory System

## 🎯 Project Overview

An intelligent, multi-agent AI system that provides personalized financial investment advice for Indian retail investors. The system uses specialized AI agents for risk assessment, market data analysis, recommendation generation, and explainability.

### Key Features

- **Multi-Agent Architecture**: Coordinated agents for different tasks
- **Personalized Recommendations**: Based on user profile and risk assessment
- **XGBoost Risk Profiling**: ML-based risk categorization
- **LLM-Powered Advice**: OpenAI GPT for natural language recommendations
- **Explainability**: SHAP and LIME for transparent decision-making
- **Memory & Learning**: Learns from user feedback to improve recommendations
- **Real-time Market Data**: Integration with financial data sources
- **Interactive Dashboard**: Streamlit-based user interface

## 🏗️ Architecture

### Agents

1. **Memory Agent**: Stores user preferences and past interactions
2. **Data Agent**: Fetches market data (FD rates, mutual funds, gold prices)
3. **Risk Agent**: XGBoost-based risk profiling
4. **Advisor Agent**: LLM-powered recommendation generation
5. **XAI Agent**: Explainability using SHAP/LIME
6. **Orchestrator**: Coordinates all agents

### Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **ML**: XGBoost, Scikit-learn
- **Explainability**: SHAP, LIME
- **LLM**: OpenAI GPT-3.5
- **Database**: SQLite
- **Data Sources**: yfinance, RBI (simulated)

## 📁 Project Structure

```
financial-advisor/
├── backend/
│   ├── agents/
│   │   ├── data_agent.py          # Market data fetching
│   │   ├── risk_agent.py          # XGBoost risk profiling
│   │   ├── advisor_agent.py       # LLM recommendations
│   │   ├── xai_agent.py           # Explainability
│   │   ├── memory_agent.py        # User preferences
│   │   └── orchestrator.py        # Agent coordination
│   ├── models/
│   │   ├── database.py            # SQLAlchemy models
│   │   ├── db_manager.py          # Database connection
│   │   └── schemas.py             # Pydantic schemas
│   ├── api/
│   ├── utils/
│   ├── config.py                  # Configuration
│   └── main.py                    # FastAPI app
├── frontend/
│   └── streamlit_app.py           # Streamlit dashboard
├── data/                          # SQLite database
├── logs/                          # Application logs
├── models/                        # Trained ML models
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image
├── docker-compose.yml             # Docker orchestration
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key
- Virtual environment (venv)

### Installation (macOS)

1. **Clone and Navigate**

```bash
cd /Users/ashutosh/BITS/Financial-Advisor
```

2. **Activate Virtual Environment**

```bash
# If you already created venv
source venv/bin/activate

# If not created yet
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Environment**

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

Update `.env` with your credentials:
```
OPENAI_API_KEY=your_actual_openai_key_here
DATABASE_URL=sqlite:///./data/financial_advisor.db
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

5. **Create Required Directories**

```bash
mkdir -p data logs models
```

## 🏃 Running the Application

### Method 1: Local Development (Recommended for Development)

#### Start Backend (Terminal 1)

```bash
# From project root
cd /Users/ashutosh/BITS/Financial-Advisor
source venv/bin/activate
python backend/main.py
```

Backend will start at: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Start Frontend (Terminal 2)

```bash
# From project root
cd /Users/ashutosh/BITS/Financial-Advisor
source venv/bin/activate
streamlit run frontend/streamlit_app.py
```

Frontend will start at: `http://localhost:8501`

### Method 2: Docker (Recommended for Deployment)

1. **Build and Start**

```bash
# Create .env file with your OpenAI key
echo "OPENAI_API_KEY=your_key_here" > .env

# Build and run
docker-compose up --build
```

2. **Access Services**
- Backend API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

3. **Stop Services**

```bash
docker-compose down
```

## 📊 Using the Application

### 1. Create User Profile

- Open Streamlit UI at `http://localhost:8501`
- Fill in user information (name, email, age, occupation)
- Click "Create Profile"

### 2. Get Investment Advice

- Navigate to "Get Advice" tab
- Enter financial information:
  - Monthly income
  - Monthly expenses
  - Investment amount
  - Investment duration
  - Financial goal
  - Risk tolerance
- Click "Get Personalized Advice"

### 3. Review Recommendations

The system will provide:
- Risk profile assessment
- Portfolio allocation
- Expected returns
- Detailed reasoning
- Explainability charts (SHAP values)

### 4. Provide Feedback

- Click "Useful" or "Not useful"
- System learns from your feedback

### 5. View History

- Check "History" tab for past sessions
- See learned preferences

## 🧪 Testing with Swagger UI

Visit `http://localhost:8000/docs` to test all API endpoints:

### Key Endpoints

1. **Create User**: `POST /api/users/`
2. **Get Investment Advice**: `POST /api/advisory/query`
3. **Submit Feedback**: `POST /api/advisory/feedback`
4. **Get Market Data**: `GET /api/market/data`
5. **Risk Assessment**: `POST /api/risk/assess`

### Example API Call

```bash
curl -X POST "http://localhost:8000/api/advisory/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "query": "I want to invest 50000 for 3 years",
    "monthly_income": 50000,
    "investment_amount": 50000,
    "investment_duration_months": 36,
    "financial_goal": "Wealth Creation",
    "risk_tolerance": "Moderate"
  }'
```

## 📈 Features Demonstration

### 1. Risk Profiling

The XGBoost model assesses risk based on:
- Monthly income
- Savings ratio
- Investment to income ratio
- Age factor
- Investment duration

### 2. Personalized Recommendations

LLM generates recommendations considering:
- User's risk profile
- Current market conditions
- Financial goals
- Investment duration

### 3. Explainability

SHAP values show:
- Feature importance
- Why specific risk category was assigned
- Impact of each factor

### 4. Memory & Learning

System learns:
- Preferred asset classes
- Risk tolerance patterns
- Feedback patterns

## 🐛 Troubleshooting

### Backend Not Starting

```bash
# Check if port 8000 is available
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Restart backend
python backend/main.py
```

### Frontend Not Connecting

- Ensure backend is running first
- Check API_BASE_URL in `streamlit_app.py`
- Verify no firewall blocking

### Database Issues

```bash
# Remove database and restart
rm -rf data/financial_advisor.db
python backend/main.py
```

### OpenAI API Errors

- Verify API key in `.env`
- Check API key validity
- Ensure sufficient credits

## 🧰 Development

### Running Tests

```bash
pytest
```

### Code Structure

- **Agents**: Modular, independent components
- **API**: RESTful endpoints with FastAPI
- **Database**: SQLAlchemy ORM with SQLite
- **Frontend**: Streamlit for rapid UI development

### Adding New Features

1. Create new agent in `backend/agents/`
2. Add to orchestrator
3. Create API endpoint
4. Update UI in Streamlit

## 📝 API Documentation

Full API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔒 Security Notes

- Never commit `.env` file
- Keep API keys secure
- Use environment variables for sensitive data
- In production, use proper authentication

## 🎓 Academic Context

**Dissertation**: Multi-Agent System for Personalized Financial Advisory
**Institution**: BITS Pilani
**Program**: M. Tech in AIML
**Course**: AIMLCZG628T - Dissertation

## 📚 References

- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/

## 🤝 Contributing

This is an academic project. For suggestions:
1. Document issues
2. Propose improvements
3. Test thoroughly

## 📧 Support

For issues or questions related to this dissertation project, contact the developer.

## 🎯 Future Enhancements

- [ ] Real-time RBI data integration
- [ ] More sophisticated portfolio optimization
- [ ] Multi-language support
- [ ] Mobile app
- [ ] Advanced backtesting
- [ ] Integration with trading platforms
- [ ] More financial instruments (commodities, bonds)

---

**Note**: This system is for educational and demonstration purposes. Always consult with certified financial advisors for actual investment decisions.
