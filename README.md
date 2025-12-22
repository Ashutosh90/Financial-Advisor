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
- **ML Tracking**: MLflow (experiment tracking, model registry)
- **Explainability**: SHAP, LIME, GPT-4o-mini
- **LLM**: OpenAI GPT-4o-mini
- **Orchestration**: LangGraph/LangChain
- **Database**: SQLite (user data + ML training data)
- **CI/CD**: GitHub Actions (drift detection, auto-retraining)
- **Monitoring**: PSI/CSI drift detection
- **Data Sources**: yfinance, RBI (simulated)

## 📁 Project Structure

```
financial-advisor/
├── backend/
│   ├── agents/
│   │   ├── data_agent.py          # Market data fetching
│   │   ├── risk_agent.py          # XGBoost risk profiling
│   │   ├── advisor_agent.py       # LLM recommendations
│   │   ├── xai_agent.py           # Explainability (SHAP/LIME)
│   │   ├── memory_agent.py        # User preferences
│   │   └── orchestrator.py        # Agent coordination (LangGraph)
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
├── mlops/                         # CI/CD & Model Monitoring
│   ├── drift_detector.py          # PSI/CSI drift detection
│   ├── model_retrainer.py         # Automated retraining
│   ├── monitoring_scheduler.py    # Scheduled monitoring
│   └── mlops_cli.py               # CLI tool
├── data/
│   ├── financial_advisor.db       # User data & sessions
│   └── risk_profiling.db          # ML training data
├── logs/                          # Application logs
├── models/                        # Trained ML models
│   ├── risk_model.json            # XGBoost model
│   ├── risk_profiling_model.pkl   # Model pickle
│   ├── label_encoder.pkl          # Target encoder
│   ├── selected_features.json     # Feature list
│   ├── best_params.json           # Hyperparameters
│   ├── evaluation_metrics.json    # Model metrics
│   ├── feature_importance.csv     # Feature importance
│   └── model_metadata.json        # Training metadata
├── mlruns/                        # MLflow tracking data
├── .github/workflows/
│   └── model-cicd.yml             # GitHub Actions CI/CD
├── data_loader_to_db.ipynb        # Load CSV to SQLite
├── risk_profiling_ml_pipeline.ipynb  # Complete ML pipeline
├── end_to_end_ml_pipeline.py      # Standalone ML pipeline script
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image
├── docker-compose.yml             # Docker orchestration
├── ARCHITECTURE.md                # Comprehensive documentation
├── QUICKSTART.md                  # Quick setup guide
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## 🧠 ML Pipeline & Model Training

### Risk Profiling Model

The risk profiling system uses XGBoost trained on historical financial data:

#### Data Pipeline

1. **Data Loading** (`data_loader_to_db.ipynb`)
   - Loads CSV data into SQLite database
   - Creates `risk_profiling_monthly_data` table
   - Schema: 107 features including demographics, financials, behavioral metrics
   - Database: `data/risk_profiling.db`

2. **ML Pipeline** (`risk_profiling_ml_pipeline.ipynb`)
   - **Data Loading**: Queries SQLite with time-based splits
   - **Preprocessing**: Handle missing values, encode categoricals
   - **Feature Engineering**: PII removal, correlation analysis
   - **Feature Selection**: RFE, SelectFromModel, SelectKBest
   - **Model Training**: XGBoost with hyperparameter tuning
   - **Evaluation**: Training, Validation, Test splits
   - **MLflow Tracking**: All experiments logged
   - **Model Persistence**: Pickle + MLflow Model Registry

#### Key Features

- **Database-First Approach**: All data stored in SQLite, no CSV dependencies
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Comprehensive Artifacts**: Model, encoder, features, metrics all saved
- **Production Ready**: Trained models in `models/` directory
- **Performance**: ~97-98% accuracy across all datasets

#### Model Files

- `risk_profiling_model.pkl`: Trained XGBoost classifier
- `label_encoder.pkl`: Target variable encoder
- `selected_features.json`: List of features used
- `best_params.json`: Optimized hyperparameters
- `evaluation_metrics.json`: Performance metrics
- `feature_importance.csv`: Feature importance scores

#### MLflow Tracking

All experiments tracked in `mlruns/`:
- Hyperparameter tuning results
- Model performance metrics
- Feature importance
- Model registry with versioning

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key
- Virtual environment (venv)

### Installation (macOS)

1. **Clone and Navigate**

```bash
cd /Users/ashutosh/Financial-Advisor
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
mkdir -p data logs models mlruns
```

6. **Setup Databases** (Optional - for ML training)

The pre-trained model is included, but to retrain:

```bash
# 1. Load data into SQLite (if you have the CSV)
jupyter notebook data_loader_to_db.ipynb

# 2. Run complete ML pipeline
jupyter notebook risk_profiling_ml_pipeline.ipynb
```

This creates:
- `data/risk_profiling.db`: Training data
- `models/`: Trained model artifacts
- `mlruns/`: MLflow tracking data

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

### MLflow UI (Optional)

To view experiment tracking and model registry:

```bash
# From project root
mlflow ui

# Access at: http://localhost:5000
```

This shows:
- All training runs with metrics
- Hyperparameter comparison
- Model registry with versions
- Artifact storage

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



## 📚 Documentation

For comprehensive documentation, see:

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete system architecture, data dictionary, ML pipeline, and technical details |
| [QUICKSTART.md](QUICKSTART.md) | Quick setup and installation guide |

### External References

- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime
- LangGraph: https://python.langchain.com/docs/langgraph
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/
- MLflow: https://mlflow.org/docs/latest/

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

## 🔄 CI/CD Pipeline & Model Monitoring

### Automated Drift Detection & Retraining

The system includes a production-ready CI/CD pipeline for automated model monitoring and retraining:

#### Overview

- **Drift Detection**: Monthly checks using PSI (Population Stability Index) and CSI (Characteristic Stability Index)
- **Automatic Retraining**: Triggered when PSI > 0.25 or multiple features have CSI > 0.25
- **Model Validation**: New models must exceed performance thresholds before deployment
- **GitHub Actions**: Automated workflow for scheduled and triggered runs

#### Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| PSI | > 0.25 | Triggers retraining |
| CSI per feature | > 0.25 | Flags feature drift |
| Accuracy | > 90% | Required for deployment |
| ROC-AUC | > 95% | Required for deployment |

#### MLOps CLI Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Check drift status
python mlops/mlops_cli.py drift-check

# Force retrain model
python mlops/mlops_cli.py retrain --force

# Run full pipeline
python mlops/mlops_cli.py pipeline

# Check system status
python mlops/mlops_cli.py status
```

#### GitHub Actions Workflow

The CI/CD pipeline is defined in `.github/workflows/model-cicd.yml`:

- **Schedule**: Runs on 1st of every month at 2 AM UTC
- **Manual Trigger**: Can be run manually with optional force retrain
- **Push Trigger**: Triggered on changes to data or mlops files

#### Model Artifacts

All model artifacts are stored in `models/`:
- `risk_model.json` - XGBoost model (JSON format)
- `risk_profiling_model.pkl` - Model pickle backup
- `selected_features.json` - List of selected features
- `label_encoder.pkl` - Target variable encoder
- `best_params.json` - Optimized hyperparameters
- `evaluation_metrics.json` - Model performance metrics
- `model_metadata.json` - Training metadata
- `feature_importance.csv` - Feature importance scores

---

**Note**: This system is for educational and demonstration purposes. Always consult with certified financial advisors for actual investment decisions.
