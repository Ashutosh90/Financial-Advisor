# 🚀 QUICK START GUIDE

Get the Financial Advisory System running in under 5 minutes!

## 📋 Prerequisites

- macOS (already have)
- Python 3.10+ installed
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## ⚡ Quick Setup (3 steps)

### Step 1: Navigate to Project

```bash
cd /Users/ashutosh/Financial-Advisor
```

### Step 2: Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create virtual environment
- Install all dependencies
- Create necessary directories
- Setup `.env` file

### Step 3: Add OpenAI API Key

```bash
nano .env
```

Replace `your_openai_api_key_here` with your actual API key, then save (Ctrl+X, Y, Enter).

## 🎯 Running the Application

### Option A: Local Development (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd /Users/ashutosh/Financial-Advisor
source venv/bin/activate
./run.sh backend
```

**Terminal 2 - Frontend:**
```bash
cd /Users/ashutosh/Financial-Advisor
source venv/bin/activate
./run.sh frontend
```

**Access:**
- Backend API: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

### Option B: Docker (Single Command)

```bash
cd /Users/ashutosh/BITS/Financial-Advisor
./run.sh docker
```

**Access:**
- Backend API: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

## 🎮 Using the System

### 1. Open Browser
Go to: http://localhost:8501

### 2. Create Profile
- Enter name, email, age, occupation
- Click "Create Profile"

### 3. Get Investment Advice
- Fill in financial details
- Click "Get Personalized Advice"

### 4. Review Recommendations
- View risk profile
- See portfolio allocation
- Read explanations
- Provide feedback

## 🧪 Testing API

Visit: http://localhost:8000/docs

Try this example:
1. Click "POST /api/advisory/query"
2. Click "Try it out"
3. Use this sample data:

```json
{
  "user_id": 1,
  "query": "I want to invest ₹50,000 for 3 years",
  "monthly_income": 50000,
  "investment_amount": 50000,
  "investment_duration_months": 36,
  "financial_goal": "Wealth Creation",
  "risk_tolerance": "Moderate"
}
```

## ❓ Common Issues

### "Backend not running" in Streamlit

**Solution:** Start backend first in separate terminal

```bash
# Terminal 1
./run.sh backend

# Wait for "Uvicorn running on http://0.0.0.0:8000"
# Then in Terminal 2
./run.sh frontend
```

### "Import Error" or "Module not found"

**Solution:** Ensure virtual environment is activated

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "OpenAI API Error"

**Solution:** Check your API key in `.env`

```bash
cat .env  # Verify OPENAI_API_KEY is set correctly
```

### Port Already in Use

**Solution:** Kill existing process

```bash
# For port 8000 (backend)
lsof -i :8000
kill -9 <PID>

# For port 8501 (frontend)
lsof -i :8501
kill -9 <PID>
```

## 📊 Demo Workflow

### Example User Journey

1. **Profile**: John, 30 years, Software Engineer, ₹80,000/month
2. **Goal**: Invest ₹1,00,000 for 5 years for retirement
3. **Risk**: Moderate

**System Response:**
- Risk Profile: Moderate (Score: 0.72)
- Portfolio: 40% Equity MF, 35% Debt MF, 20% FD, 5% Gold
- Expected Return: 9.5% annually
- Future Value: ₹1,55,000

## 🎓 For Dissertation Demo

### Demo Script

```bash
# 1. Start system
./run.sh backend  # Terminal 1
./run.sh frontend # Terminal 2

# 2. Show architecture
open http://localhost:8501

# 3. Create sample user
# Name: Demo User, Email: demo@bits.edu, Age: 30

# 4. Test conservative investor
Monthly Income: ₹40,000
Investment: ₹50,000
Duration: 24 months
Goal: Capital Protection
Risk: Conservative

# 5. Test aggressive investor
Monthly Income: ₹1,00,000
Investment: ₹2,00,000
Duration: 60 months
Goal: Wealth Creation
Risk: Aggressive

# 6. Show explainability
# SHAP values, feature importance

# 7. Test learning
# Provide feedback, show preferences
```

## 📞 Quick Help

**Project directory:**
```
/Users/ashutosh/BITS/Financial-Advisor
```

**Key files:**
- `backend/main.py` - FastAPI application
- `frontend/streamlit_app.py` - Streamlit UI
- `.env` - Configuration (add your API key here!)
- `requirements.txt` - Dependencies

**Logs:**
- Application: `logs/app.log`
- Errors: Check terminal output

## 🎯 Next Steps

After getting it running:

1. **Read [ARCHITECTURE.md](ARCHITECTURE.md)** for comprehensive documentation including:
   - System architecture and agent details
   - Complete data dictionary with 112 features grouped by category
   - ML pipeline, feature engineering, and model performance
   - XAI (SHAP, LIME, GPT-4o-mini) implementation
   - Database schema and API documentation
2. **Explore API** at http://localhost:8000/docs
3. **Test different scenarios** in Streamlit
4. **View MLflow UI** (optional):
   ```bash
   # In a new terminal
   cd /Users/ashutosh/BITS/Financial-Advisor
   source venv/bin/activate
   mlflow ui
   # Access at: http://localhost:5000
   ```
   - View ML experiments
   - Compare model versions
   - Check training metrics

## 🧠 ML Model Training (Optional)

If you want to retrain the risk profiling model:

```bash
# 1. Load data into database
jupyter notebook data_loader_to_db.ipynb

# 2. Run complete ML pipeline
jupyter notebook risk_profiling_ml_pipeline.ipynb

# Or use the Python script:
python end_to_end_ml_pipeline.py
```

This will:
- Create `data/risk_profiling.db` database
- Train XGBoost model with feature selection
- Save artifacts to `models/` directory
- Log experiments to MLflow (`mlruns/`)
- Achieve ~98% accuracy

**Pre-trained model included**, so this is optional!

---

## 🔄 MLOps & CI/CD Commands

The system includes automated model monitoring and retraining:

### Check Model Status
```bash
source venv/bin/activate
python mlops/mlops_cli.py status
```

### Run Drift Detection
```bash
python mlops/mlops_cli.py drift-check
```

### Force Retrain Model
```bash
python mlops/mlops_cli.py retrain --force
```

### Run Full Pipeline
```bash
python mlops/mlops_cli.py pipeline
```

### Drift Detection Thresholds
- **PSI > 0.25**: Indicates significant population shift → triggers retraining
- **CSI > 0.25 per feature**: Indicates feature drift

---

**Need help?** Check README.md or logs/app.log for detailed information.
