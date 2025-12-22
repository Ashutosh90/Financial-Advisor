"""
Generate categorical encoders from existing database data
This creates the categorical_encoders.pkl file needed by risk_agent.py
"""
import pickle
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

def create_categorical_encoders():
    """Create and save categorical encoders from database data"""
    
    print("\n" + "="*80)
    print("GENERATING CATEGORICAL ENCODERS")
    print("="*80)
    
    # Load data from database
    db_path = './data/risk_profiling.db'
    print(f"\n1. Loading data from: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        # Load a sample to get all unique categorical values
        query = "SELECT customer_segment, education FROM risk_profiling_monthly_data"
        df = pd.read_sql(query, conn)
        conn.close()
        print(f"   ✓ Loaded {len(df):,} records")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Create encoders for categorical columns
    print(f"\n2. Creating LabelEncoders for categorical features...")
    categorical_cols = ['customer_segment', 'education']
    label_encoders = {}
    
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            label_encoders[col] = le
            
            print(f"   ✓ {col}:")
            print(f"      Classes: {list(le.classes_)}")
            print(f"      Mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
        except Exception as e:
            print(f"   ✗ Error encoding {col}: {e}")
            return False
    
    # Save encoders
    print(f"\n3. Saving categorical encoders...")
    output_path = './models/categorical_encoders.pkl'
    
    try:
        # Create models directory if it doesn't exist
        Path('./models').mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        print(f"   ✓ Saved to: {output_path}")
        
        # Verify the file
        file_size = Path(output_path).stat().st_size
        print(f"   ✓ File size: {file_size:,} bytes")
        
    except Exception as e:
        print(f"   ✗ Error saving encoders: {e}")
        return False
    
    # Verify by loading
    print(f"\n4. Verifying saved encoders...")
    try:
        with open(output_path, 'rb') as f:
            loaded_encoders = pickle.load(f)
        
        print(f"   ✓ Successfully loaded {len(loaded_encoders)} encoders")
        print(f"   ✓ Encoder keys: {list(loaded_encoders.keys())}")
        
    except Exception as e:
        print(f"   ✗ Error verifying encoders: {e}")
        return False
    
    print("\n" + "="*80)
    print("✅ CATEGORICAL ENCODERS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nThe risk_agent.py will now use these saved encoders instead of hardcoded mappings.")
    print(f"\nTo test:")
    print(f"  python -c \"from backend.agents.risk_agent import RiskAgent; agent = RiskAgent(); print('Encoders loaded:', len(agent.categorical_encoders))\"")
    print()
    
    return True


if __name__ == "__main__":
    success = create_categorical_encoders()
    sys.exit(0 if success else 1)
