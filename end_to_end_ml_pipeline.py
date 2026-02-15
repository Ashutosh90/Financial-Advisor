"""
================================================================================
END-TO-END ML PIPELINE FOR INDIAN BANKING RISK PROFILING
================================================================================

This script implements a complete ML pipeline with:
1. Data Loading & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Feature Selection (Multiple Methods)
5. Hyperparameter Tuning (Grid Search & Random Search)
6. Model Training (XGBoost)
7. Model Evaluation
8. MLOps Tracking (MLflow)
9. Model Persistence

Author: ML Team
Date: December 2024
Problem: Multi-class classification (3 classes: Conservative, Moderate, Aggressive)
Dataset: 120,000 records (70% Train, 15% Val, 15% Test)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import json
import os
import sqlite3
from sqlalchemy import create_engine, text
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# MLOps - MLflow for experiment tracking
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print("INDIAN BANKING CUSTOMER RISK PROFILING - END-TO-END ML PIPELINE")
print("="*100)
print(f"\nPipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ================================================================================
# 1. DATA LOADING & PREPROCESSING
# ================================================================================

def get_last_month_end_date():
    """
    Calculate the last day of the previous month.
    
    Returns:
        str: Date in 'YYYY-MM-DD' format
    """
    today = datetime.now()
    first_day_current_month = today.replace(day=1)
    last_day_previous_month = first_day_current_month - timedelta(days=1)
    return last_day_previous_month.strftime('%Y-%m-%d')


def load_data_from_database(db_path, table_name, snapshot_date=None):
    """
    Load dataset from SQLite database and separate by data_split column.
    
    Args:
        db_path: Path to SQLite database file
        table_name: Name of the table
        snapshot_date: Optional - filter by snapshot date
        
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames
    """
    print("\n" + "="*100)
    print("STEP 1: DATA LOADING FROM SQLITE DATABASE")
    print("="*100)
    
    print(f"\n✓ Connecting to SQLite database:")
    print(f"  Database: {db_path}")
    print(f"  Table: {table_name}")
    
    # Create database connection
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Build query
    if snapshot_date:
        query = f"SELECT * FROM {table_name} WHERE snapshot_date = '{snapshot_date}'"
        print(f"  Filtering by snapshot_date: {snapshot_date}")
    else:
        query = f"SELECT * FROM {table_name}"
        print(f"  Loading all data (no snapshot filter)")
    
    # Load data from database
    print(f"\n✓ Executing query...")
    df = pd.read_sql(query, engine)
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total records: {len(df):,}")
    print(f"  Total columns: {df.shape[1]}")
    
    # Check if snapshot_date column exists
    if 'snapshot_date' in df.columns:
        print(f"  Snapshot dates in data: {df['snapshot_date'].unique().tolist()}")
    
    # Separate by data_split
    train_df = df[df['data_split'] == 'Train'].copy()
    val_df = df[df['data_split'] == 'Validation'].copy()
    test_df = df[df['data_split'] == 'Test'].copy()
    
    print(f"\n✓ Data split completed:")
    print(f"  Training set:   {len(train_df):>6,} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df):>6,} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set:       {len(test_df):>6,} records ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df):
    """
    Preprocess data: separate features and target, encode categorical variables.
    
    Args:
        train_df, val_df, test_df: DataFrames for each split
        
    Returns:
        tuple: Preprocessed features and targets for all splits, plus encoders
    """
    print("\n" + "="*100)
    print("STEP 2: DATA PREPROCESSING")
    print("="*100)
    
    # Columns to drop (not features)
    drop_cols = ['customer_id', 'data_split', 'risk_profile']
    
    # Separate features and target
    X_train = train_df.drop(drop_cols, axis=1).copy()
    y_train = train_df['risk_profile'].copy()
    
    X_val = val_df.drop(drop_cols, axis=1).copy()
    y_val = val_df['risk_profile'].copy()
    
    X_test = test_df.drop(drop_cols, axis=1).copy()
    y_test = test_df['risk_profile'].copy()
    
    print(f"\n✓ Features and target separated:")
    print(f"  Training features shape:   {X_train.shape}")
    print(f"  Training target shape:     {y_train.shape}")
    print(f"  Number of features:        {X_train.shape[1]}")
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\n✓ Feature types identified:")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Numerical features:   {len(numerical_cols)}")
    print(f"\n  Categorical: {categorical_cols}")
    
    # Encode categorical features using LabelEncoder
    print(f"\n✓ Encoding categorical features...")
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        print(f"    • {col}: {len(le.classes_)} unique values")
    
    # Encode target variable
    print(f"\n✓ Encoding target variable...")
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_val_encoded = le_target.transform(y_val)
    y_test_encoded = le_target.transform(y_test)
    
    print(f"  Classes: {le_target.classes_}")
    print(f"  Encoded as: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # Class distribution
    print(f"\n✓ Class distribution in training set:")
    for i, cls in enumerate(le_target.classes_):
        count = (y_train_encoded == i).sum()
        pct = count / len(y_train_encoded) * 100
        print(f"  {cls:13} (class {i}): {count:>6,} ({pct:>5.2f}%)")
    
    return (X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded,
            label_encoders, le_target, categorical_cols, numerical_cols)


# ================================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================================

def perform_eda(X_train, y_train, categorical_cols, numerical_cols, le_target):
    """
    Perform exploratory data analysis on training data.
    
    Args:
        X_train: Training features
        y_train: Training target (encoded)
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        le_target: Label encoder for target
    """
    print("\n" + "="*100)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*100)
    
    print(f"\n✓ Dataset summary:")
    print(f"  Shape: {X_train.shape}")
    print(f"  Missing values: {X_train.isnull().sum().sum()}")
    print(f"  Duplicate rows: {X_train.duplicated().sum()}")
    
    # Statistical summary for numerical features
    print(f"\n✓ Numerical features - Statistical summary (first 10):")
    print(X_train[numerical_cols[:10]].describe().T[['mean', 'std', 'min', 'max']])
    
    # Check for correlations
    print(f"\n✓ Checking feature correlations...")
    corr_matrix = X_train[numerical_cols].corr()
    
    # Find highly correlated features (>0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (|corr| > 0.9):")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"    • {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print(f"  No highly correlated features found (|corr| > 0.9)")
    
    return high_corr_pairs


# ================================================================================
# 3. FEATURE ENGINEERING
# ================================================================================

def engineer_features(X_train, X_val, X_test):
    """
    Create additional engineered features.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        
    Returns:
        tuple: DataFrames with engineered features
    """
    print("\n" + "="*100)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*100)
    
    print(f"\n✓ Creating engineered features...")
    
    original_features = X_train.shape[1]
    
    # Note: Most important features are already engineered (ratios, etc.)
    # Adding a few interaction features
    
    for df in [X_train, X_val, X_test]:
        # Income per dependent (handling division by zero)
        df['income_per_dependent'] = df['annual_income'] / (df['dependents'] + 1)
        
        # Debt coverage ratio
        df['debt_coverage_ratio'] = df['monthly_income'] / (df['total_debt'] / 12 + 1)
        
        # Savings rate
        df['savings_rate'] = (df['monthly_income'] - df['monthly_expenses']) / (df['monthly_income'] + 1)
        
        # Investment efficiency
        df['investment_efficiency'] = df['investment_portfolio_value'] / (df['annual_income'] + 1)
    
    new_features = X_train.shape[1] - original_features
    print(f"  • Original features: {original_features}")
    print(f"  • New features created: {new_features}")
    print(f"  • Total features: {X_train.shape[1]}")
    print(f"\n  New features:")
    print(f"    • income_per_dependent")
    print(f"    • debt_coverage_ratio")
    print(f"    • savings_rate")
    print(f"    • investment_efficiency")
    
    return X_train, X_val, X_test


# ================================================================================
# 4. FEATURE SELECTION
# ================================================================================

def select_features_univariate(X_train, y_train, X_val, X_test, k=30):
    """
    Feature selection using univariate statistical tests (ANOVA F-test).
    
    Args:
        X_train, y_train: Training data
        X_val, X_test: Validation and test data
        k: Number of top features to select
        
    Returns:
        tuple: Selected features and selector object
    """
    print(f"\n{'='*100}")
    print(f"METHOD 1: UNIVARIATE FEATURE SELECTION (ANOVA F-test)")
    print(f"{'='*100}")
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    print(f"\n✓ Selected {k} features using ANOVA F-test:")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"\n  Top 10 selected features:")
    
    # Get feature scores
    scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    for i, (feat, score) in enumerate(scores.head(10).values, 1):
        print(f"    {i:2d}. {feat:30s} (score: {score:>10.2f})")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, selector


def select_features_mutual_info(X_train, y_train, X_val, X_test, k=30):
    """
    Feature selection using mutual information.
    
    Args:
        X_train, y_train: Training data
        X_val, X_test: Validation and test data
        k: Number of top features to select
        
    Returns:
        tuple: Selected features and selector object
    """
    print(f"\n{'='*100}")
    print(f"METHOD 2: MUTUAL INFORMATION FEATURE SELECTION")
    print(f"{'='*100}")
    
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    print(f"\n✓ Selected {k} features using Mutual Information:")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"\n  Top 10 selected features:")
    
    # Get feature scores
    scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    for i, (feat, score) in enumerate(scores.head(10).values, 1):
        print(f"    {i:2d}. {feat:30s} (score: {score:>10.4f})")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, selector


def select_features_rfe(X_train, y_train, X_val, X_test, n_features=30):
    """
    Recursive Feature Elimination using Random Forest.
    
    Args:
        X_train, y_train: Training data
        X_val, X_test: Validation and test data
        n_features: Number of features to select
        
    Returns:
        tuple: Selected features and selector object
    """
    print(f"\n{'='*100}")
    print(f"METHOD 3: RECURSIVE FEATURE ELIMINATION (RFE)")
    print(f"{'='*100}")
    
    # Use Random Forest as the estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    print(f"\n  Running RFE with Random Forest estimator...")
    print(f"  This may take a few minutes...")
    
    selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    print(f"\n✓ Selected {n_features} features using RFE:")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"\n  Selected features (top 10):")
    
    for i, feat in enumerate(selected_features[:10], 1):
        print(f"    {i:2d}. {feat}")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, selector


def select_features_tree_based(X_train, y_train, X_val, X_test, threshold='median'):
    """
    Feature selection using tree-based feature importance (XGBoost).
    
    Args:
        X_train, y_train: Training data
        X_val, X_test: Validation and test data
        threshold: Importance threshold ('median', 'mean', or float)
        
    Returns:
        tuple: Selected features and selector object
    """
    print(f"\n{'='*100}")
    print(f"METHOD 4: TREE-BASED FEATURE SELECTION (XGBoost)")
    print(f"{'='*100}")
    
    # Train XGBoost model for feature importance
    print(f"\n  Training XGBoost model for feature importance...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Use SelectFromModel to select features
    selector = SelectFromModel(xgb_model, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    print(f"\n✓ Selected features using tree-based importance (threshold='{threshold}'):")
    print(f"  Original features: {X_train.shape[1]}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"\n  Top 10 important features:")
    
    # Get feature importances
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (feat, imp) in enumerate(importance_df.head(10).values, 1):
        print(f"    {i:2d}. {feat:30s} (importance: {imp:>8.4f})")
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, selector


def compare_feature_selection_methods(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Compare multiple feature selection methods and choose the best one.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        
    Returns:
        dict: Results from all feature selection methods
    """
    print("\n" + "="*100)
    print("STEP 5: FEATURE SELECTION - COMPARING MULTIPLE METHODS")
    print("="*100)
    
    results = {}
    
    # Method 1: Univariate (ANOVA F-test)
    X_train_uni, X_val_uni, X_test_uni, features_uni, selector_uni = \
        select_features_univariate(X_train, y_train, X_val, X_test, k=30)
    results['univariate'] = {
        'X_train': X_train_uni, 'X_val': X_val_uni, 'X_test': X_test_uni,
        'features': features_uni, 'selector': selector_uni
    }
    
    # Method 2: Mutual Information
    X_train_mi, X_val_mi, X_test_mi, features_mi, selector_mi = \
        select_features_mutual_info(X_train, y_train, X_val, X_test, k=30)
    results['mutual_info'] = {
        'X_train': X_train_mi, 'X_val': X_val_mi, 'X_test': X_test_mi,
        'features': features_mi, 'selector': selector_mi
    }
    
    # Method 3: RFE
    X_train_rfe, X_val_rfe, X_test_rfe, features_rfe, selector_rfe = \
        select_features_rfe(X_train, y_train, X_val, X_test, n_features=30)
    results['rfe'] = {
        'X_train': X_train_rfe, 'X_val': X_val_rfe, 'X_test': X_test_rfe,
        'features': features_rfe, 'selector': selector_rfe
    }
    
    # Method 4: Tree-based
    X_train_tree, X_val_tree, X_test_tree, features_tree, selector_tree = \
        select_features_tree_based(X_train, y_train, X_val, X_test, threshold='median')
    results['tree_based'] = {
        'X_train': X_train_tree, 'X_val': X_val_tree, 'X_test': X_test_tree,
        'features': features_tree, 'selector': selector_tree
    }
    
    # Quick evaluation with simple XGBoost model
    print(f"\n{'='*100}")
    print(f"QUICK EVALUATION OF FEATURE SELECTION METHODS")
    print(f"{'='*100}")
    print(f"\nTraining simple XGBoost models to compare feature selection methods...")
    
    evaluation_results = {}
    
    for method_name, method_data in results.items():
        # Train simple model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(method_data['X_train'], y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(method_data['X_val'])
        val_acc = accuracy_score(y_val, y_val_pred)
        
        evaluation_results[method_name] = {
            'val_accuracy': val_acc,
            'n_features': len(method_data['features'])
        }
        
        print(f"\n  {method_name:15s}: Val Accuracy = {val_acc:.4f} ({val_acc*100:.2f}%) | Features = {len(method_data['features'])}")
    
    # Select best method based on validation accuracy
    best_method = max(evaluation_results.items(), key=lambda x: x[1]['val_accuracy'])
    
    print(f"\n{'='*100}")
    print(f"✓ BEST FEATURE SELECTION METHOD: {best_method[0].upper()}")
    print(f"  Validation Accuracy: {best_method[1]['val_accuracy']:.4f} ({best_method[1]['val_accuracy']*100:.2f}%)")
    print(f"  Number of Features: {best_method[1]['n_features']}")
    print(f"{'='*100}")
    
    return results, evaluation_results, best_method[0]


# ================================================================================
# 5. HYPERPARAMETER TUNING
# ================================================================================

def tune_hyperparameters_grid_search(X_train, y_train, X_val, y_val):
    """
    Hyperparameter tuning using Grid Search CV.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        tuple: Best parameters and best model
    """
    print("\n" + "="*100)
    print("STEP 6: HYPERPARAMETER TUNING - GRID SEARCH")
    print("="*100)
    
    print(f"\n✓ Setting up Grid Search...")
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    print(f"\n  Parameter grid:")
    for param, values in param_grid.items():
        print(f"    {param:20s}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n  Total combinations: {total_combinations:,}")
    print(f"  Note: This may take considerable time. Using CV=3 for speed.")
    
    # Base XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Grid Search with Cross-Validation
    print(f"\n  Running Grid Search (this may take 10-20 minutes)...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Grid Search completed!")
    print(f"\n  Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param:20s}: {value}")
    
    print(f"\n  Best cross-validation score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    # Evaluate on validation set
    y_val_pred = grid_search.best_estimator_.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return grid_search.best_params_, grid_search.best_estimator_


def tune_hyperparameters_random_search(X_train, y_train, X_val, y_val, n_iter=50):
    """
    Hyperparameter tuning using Randomized Search CV (faster than Grid Search).
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_iter: Number of parameter settings sampled
        
    Returns:
        tuple: Best parameters and best model
    """
    print("\n" + "="*100)
    print("STEP 6: HYPERPARAMETER TUNING - RANDOMIZED SEARCH")
    print("="*100)
    
    print(f"\n✓ Setting up Randomized Search...")
    
    # Define parameter distributions
    param_distributions = {
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        'n_estimators': [100, 150, 200, 250, 300, 350, 400],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        'colsample_bytree': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.25],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1]
    }
    
    print(f"\n  Parameter distributions:")
    for param, values in param_distributions.items():
        print(f"    {param:20s}: {values}")
    
    print(f"\n  Number of iterations: {n_iter}")
    print(f"  CV folds: 3")
    
    # Base XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Randomized Search with Cross-Validation
    print(f"\n  Running Randomized Search (this may take 5-10 minutes)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n✓ Randomized Search completed!")
    print(f"\n  Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"    {param:20s}: {value}")
    
    print(f"\n  Best cross-validation score: {random_search.best_score_:.4f} ({random_search.best_score_*100:.2f}%)")
    
    # Evaluate on validation set
    y_val_pred = random_search.best_estimator_.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return random_search.best_params_, random_search.best_estimator_


# ================================================================================
# 6. MODEL TRAINING
# ================================================================================

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Train final XGBoost model with best parameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        best_params: Best hyperparameters from tuning
        
    Returns:
        Trained XGBoost model
    """
    print("\n" + "="*100)
    print("STEP 7: TRAINING FINAL MODEL")
    print("="*100)
    
    print(f"\n✓ Training XGBoost model with optimized hyperparameters...")
    
    # Create model with best parameters
    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softmax',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Train with early stopping
    print(f"\n  Training with early stopping (patience=50)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"\n✓ Model training completed!")
    print(f"  Best iteration: {model.best_iteration}")
    
    return model


# ================================================================================
# 7. MODEL EVALUATION
# ================================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, le_target):
    """
    Comprehensive model evaluation on all datasets.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test/OOT data
        le_target: Label encoder for target
        
    Returns:
        dict: Evaluation metrics for all datasets
    """
    print("\n" + "="*100)
    print("STEP 8: MODEL EVALUATION")
    print("="*100)
    
    results = {}
    
    for dataset_name, (X, y) in [('Training', (X_train, y_train)),
                                   ('Validation', (X_val, y_val)),
                                   ('Test/OOT', (X_test, y_test))]:
        
        print(f"\n{'='*100}")
        print(f"{dataset_name.upper()} SET EVALUATION")
        print(f"{'='*100}")
        
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        print(f"\n🎯 Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Classification report
        print(f"\n📊 Classification Report:")
        print(classification_report(y, y_pred, 
                                   target_names=le_target.classes_,
                                   digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\n📉 Confusion Matrix:")
        cm_df = pd.DataFrame(cm, 
                            index=[f"Actual {c}" for c in le_target.classes_],
                            columns=[f"Pred {c}" for c in le_target.classes_])
        print(cm_df)
        
        # Store results
        results[dataset_name.lower()] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
    
    return results


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract and display feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        DataFrame: Feature importance
    """
    print(f"\n{'='*100}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*100}")
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n✓ Top {top_n} Most Important Features:")
    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Cumulative %'}")
    print("-" * 70)
    
    cumsum = 0
    for i, (feat, imp) in enumerate(importance_df.head(top_n).values, 1):
        cumsum += imp
        cumsum_pct = cumsum / importance_df['importance'].sum() * 100
        print(f"{i:<6} {feat:<35} {imp:<12.6f} {cumsum_pct:>6.2f}%")
    
    print(f"\n  Top {top_n} features account for {cumsum_pct:.2f}% of total importance")
    
    return importance_df


# ================================================================================
# 8. MLOPS - EXPERIMENT TRACKING WITH MLFLOW
# ================================================================================

def setup_mlflow(experiment_name="indian_banking_risk_profiling"):
    """
    Set up MLflow for experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
    """
    print("\n" + "="*100)
    print("STEP 9: MLOPS - EXPERIMENT TRACKING WITH MLFLOW")
    print("="*100)
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"\n✓ MLflow experiment set: {experiment_name}")
    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    
    return experiment_name


def log_experiment_to_mlflow(model, best_params, evaluation_results, 
                             feature_importance_df, feature_selection_method,
                             selected_features):
    """
    Log experiment details to MLflow.
    
    Args:
        model: Trained model
        best_params: Best hyperparameters
        evaluation_results: Evaluation metrics
        feature_importance_df: Feature importance DataFrame
        feature_selection_method: Name of feature selection method used
        selected_features: List of selected features
    """
    print(f"\n✓ Logging experiment to MLflow...")
    
    with mlflow.start_run(run_name=f"XGBoost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("feature_selection_method", feature_selection_method)
        mlflow.log_param("n_features", len(selected_features))
        
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        # Log metrics for all datasets
        for dataset_name, metrics in evaluation_results.items():
            mlflow.log_metric(f"{dataset_name}_accuracy", metrics['accuracy'])
            mlflow.log_metric(f"{dataset_name}_precision", metrics['precision'])
            mlflow.log_metric(f"{dataset_name}_recall", metrics['recall'])
            mlflow.log_metric(f"{dataset_name}_f1", metrics['f1'])
        
        # Log model
        signature = infer_signature(evaluation_results['training']['y_pred_proba'], 
                                    evaluation_results['training']['y_pred'])
        mlflow.xgboost.log_model(model, "model", signature=signature)
        
        # Log feature importance
        feature_importance_df.to_csv('/tmp/feature_importance.csv', index=False)
        mlflow.log_artifact('/tmp/feature_importance.csv')
        
        # Log selected features
        pd.DataFrame({'selected_features': selected_features}).to_csv(
            '/tmp/selected_features.csv', index=False)
        mlflow.log_artifact('/tmp/selected_features.csv')
        
        print(f"  ✓ Parameters logged")
        print(f"  ✓ Metrics logged")
        print(f"  ✓ Model logged")
        print(f"  ✓ Artifacts logged")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n  Run ID: {run_id}")
        
    return run_id


# ================================================================================
# 9. MODEL PERSISTENCE
# ================================================================================

def save_model_and_artifacts(model, feature_importance_df, selected_features, 
                            best_params, evaluation_results, label_encoders, 
                            le_target, output_dir='/mnt/user-data/outputs'):
    """
    Save model and all related artifacts.
    
    Args:
        model: Trained model
        feature_importance_df: Feature importance
        selected_features: List of selected features
        best_params: Best hyperparameters
        evaluation_results: Evaluation metrics
        label_encoders: Dictionary of label encoders
        le_target: Target label encoder
        output_dir: Output directory
    """
    print("\n" + "="*100)
    print("STEP 10: MODEL PERSISTENCE")
    print("="*100)
    
    print(f"\n✓ Saving model and artifacts to: {output_dir}")
    
    import json
    
    # Save XGBoost model in JSON format
    model_path = f"{output_dir}/risk_model.json"
    model.save_model(model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save feature importance
    importance_path = f"{output_dir}/feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved: {importance_path}")
    
    # Save selected features as JSON (for mlops compatibility)
    features_json_path = f"{output_dir}/selected_features.json"
    with open(features_json_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"  ✓ Selected features saved: {features_json_path}")
    
    # Save best parameters
    params_path = f"{output_dir}/best_params.json"
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"  ✓ Best parameters saved: {params_path}")
    
    # Save evaluation results (renamed for mlops compatibility)
    eval_path = f"{output_dir}/evaluation_metrics.json"
    eval_to_save = {}
    for dataset, metrics in evaluation_results.items():
        # Map keys for compatibility
        key = dataset.replace('/', '_')
        if key == 'test_oot':
            key = 'test'
        eval_to_save[key] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
    with open(eval_path, 'w') as f:
        json.dump(eval_to_save, f, indent=4)
    print(f"  ✓ Evaluation metrics saved: {eval_path}")
    
    # Save label encoder
    encoder_path = f"{output_dir}/label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(le_target, f)
    print(f"  ✓ Label encoder saved: {encoder_path}")
    
    # Save model metadata
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'n_features': len(selected_features),
        'n_estimators': best_params.get('n_estimators', 'unknown'),
        'test_accuracy': eval_to_save.get('test', {}).get('accuracy', 0),
        'framework': 'xgboost',
        'problem_type': 'multi-class classification'
    }
    metadata_path = f"{output_dir}/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"  ✓ Model metadata saved: {metadata_path}")
    
    print(f"\n✓ All artifacts saved successfully!")


# ================================================================================
# MAIN PIPELINE
# ================================================================================

def remove_pii_features(X_train, X_val, X_test, categorical_cols, numerical_cols):
    """
    Remove PII (Personally Identifiable Information) features.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        
    Returns:
        Updated DataFrames and column lists
    """
    print("\n" + "="*100)
    print("STEP: PII FEATURE REMOVAL")
    print("="*100)
    
    pii_to_remove = ['gender', 'marital_status']
    pii_found = [col for col in pii_to_remove if col in X_train.columns]
    
    if pii_found:
        X_train = X_train.drop(pii_found, axis=1)
        X_val = X_val.drop(pii_found, axis=1)
        X_test = X_test.drop(pii_found, axis=1)
        
        categorical_cols = [col for col in categorical_cols if col not in pii_found]
        numerical_cols = [col for col in numerical_cols if col not in pii_found]
        
        print(f"\n✓ PII columns dropped from features: {pii_found}")
        print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    else:
        print(f"\n✓ No PII columns found to remove")
    
    return X_train, X_val, X_test, categorical_cols, numerical_cols


def remove_highly_correlated_features(X_train, X_val, X_test, numerical_cols, threshold=0.9):
    """
    Remove highly correlated features based on correlation threshold.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        numerical_cols: List of numerical column names
        threshold: Correlation threshold (default 0.9)
        
    Returns:
        Updated DataFrames and numerical_cols list
    """
    print("\n" + "="*100)
    print("STEP: CORRELATION-BASED FEATURE REMOVAL")
    print("="*100)
    
    print(f"\n✓ Calculating correlation matrix for {len(numerical_cols)} numerical features...")
    corr_matrix = X_train[numerical_cols].corr()
    
    # Find highly correlated feature pairs
    features_to_drop = set()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > threshold:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_matrix.iloc[i, j]))
                features_to_drop.add(feat2)
    
    print(f"\n📊 Correlation Analysis Results:")
    print(f"  Correlation threshold: {threshold}")
    print(f"  Highly correlated pairs found: {len(high_corr_pairs)}")
    print(f"  Features to drop: {len(features_to_drop)}")
    
    if features_to_drop:
        orig_shape = X_train.shape[1]
        X_train = X_train.drop(columns=list(features_to_drop))
        X_val = X_val.drop(columns=list(features_to_drop))
        X_test = X_test.drop(columns=list(features_to_drop))
        numerical_cols = [col for col in numerical_cols if col not in features_to_drop]
        
        print(f"\n✓ Removed {len(features_to_drop)} highly correlated features")
        print(f"  Original features: {orig_shape}")
        print(f"  Remaining features: {X_train.shape[1]}")
    else:
        print(f"\n✅ No highly correlated features found (|correlation| > {threshold})")
    
    return X_train, X_val, X_test, numerical_cols, high_corr_pairs


def main():
    """
    Main function to run the complete end-to-end ML pipeline.
    """
    
    # Configuration - Use SQLite database (matching notebook)
    DB_PATH = 'data/risk_profiling.db'
    TABLE_NAME = 'risk_profiling_monthly_data'
    SNAPSHOT_DATE = get_last_month_end_date()
    OUTPUT_DIR = 'models'
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    # Step 1: Load Data from SQLite
    train_df, val_df, test_df = load_data_from_database(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        snapshot_date=SNAPSHOT_DATE
    )
    
    # Step 2: Preprocess Data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     label_encoders, le_target, categorical_cols, numerical_cols) = \
        preprocess_data(train_df, val_df, test_df)
    
    # Step 2.5: Remove PII Features (matching notebook)
    X_train, X_val, X_test, categorical_cols, numerical_cols = \
        remove_pii_features(X_train, X_val, X_test, categorical_cols, numerical_cols)
    
    # Step 3: EDA
    high_corr_pairs = perform_eda(X_train, y_train, categorical_cols, 
                                   numerical_cols, le_target)
    
    # Step 3.5: Remove Highly Correlated Features (matching notebook)
    X_train, X_val, X_test, numerical_cols, _ = \
        remove_highly_correlated_features(X_train, X_val, X_test, numerical_cols, threshold=0.9)
    
    # Step 4: Feature Engineering
    X_train, X_val, X_test = engineer_features(X_train, X_val, X_test)
    
    # Step 5: Feature Selection
    print(f"\n{'='*100}")
    print(f"NOTE: Feature selection may take 10-15 minutes. Please wait...")
    print(f"{'='*100}")
    
    fs_results, fs_evaluation, best_fs_method = \
        compare_feature_selection_methods(X_train, y_train, X_val, y_val, 
                                         X_test, y_test)
    
    # Use best feature selection method
    X_train_selected = fs_results[best_fs_method]['X_train']
    X_val_selected = fs_results[best_fs_method]['X_val']
    X_test_selected = fs_results[best_fs_method]['X_test']
    selected_features = fs_results[best_fs_method]['features']
    
    # Step 6: Hyperparameter Tuning
    print(f"\n{'='*100}")
    print(f"NOTE: Hyperparameter tuning may take 10-20 minutes. Please wait...")
    print(f"{'='*100}")
    
    # Use Randomized Search (faster than Grid Search)
    best_params, best_model_cv = tune_hyperparameters_random_search(
        X_train_selected, y_train, X_val_selected, y_val, n_iter=50
    )
    
    # Step 7: Train Final Model
    final_model = train_final_model(X_train_selected, y_train, 
                                    X_val_selected, y_val, best_params)
    
    # Step 8: Evaluate Model
    evaluation_results = evaluate_model(final_model, 
                                       X_train_selected, y_train,
                                       X_val_selected, y_val,
                                       X_test_selected, y_test,
                                       le_target)
    
    # Get Feature Importance
    feature_importance_df = get_feature_importance(final_model, selected_features, top_n=20)
    
    # Step 9: MLOps - MLflow Tracking
    experiment_name = setup_mlflow()
    run_id = log_experiment_to_mlflow(final_model, best_params, evaluation_results,
                                      feature_importance_df, best_fs_method,
                                      selected_features)
    
    # Step 10: Save Model and Artifacts
    save_model_and_artifacts(final_model, feature_importance_df, selected_features,
                            best_params, evaluation_results, label_encoders,
                            le_target, OUTPUT_DIR)
    
    # Final Summary
    print("\n" + "="*100)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*100)
    
    print(f"\n📊 Final Results:")
    print(f"  Feature Selection Method: {best_fs_method}")
    print(f"  Number of Features: {len(selected_features)}")
    print(f"  Training Accuracy:   {evaluation_results['training']['accuracy']:.4f} ({evaluation_results['training']['accuracy']*100:.2f}%)")
    print(f"  Validation Accuracy: {evaluation_results['validation']['accuracy']:.4f} ({evaluation_results['validation']['accuracy']*100:.2f}%)")
    print(f"  Test/OOT Accuracy:   {evaluation_results['test/oot']['accuracy']:.4f} ({evaluation_results['test/oot']['accuracy']*100:.2f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"  • Model: {OUTPUT_DIR}/risk_model.json")
    print(f"  • Feature Importance: {OUTPUT_DIR}/feature_importance.csv")
    print(f"  • Selected Features: {OUTPUT_DIR}/selected_features.json")
    print(f"  • Best Parameters: {OUTPUT_DIR}/best_params.json")
    print(f"  • Evaluation Metrics: {OUTPUT_DIR}/evaluation_metrics.json")
    print(f"  • Label Encoder: {OUTPUT_DIR}/label_encoder.pkl")
    print(f"  • Model Metadata: {OUTPUT_DIR}/model_metadata.json")
    
    print(f"\n🔬 MLflow:")
    print(f"  Experiment: {experiment_name}")
    print(f"  Run ID: {run_id}")
    print(f"  View experiments: mlflow ui --port 5000")
    
    print(f"\n⏰ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)


if __name__ == "__main__":
    main()
