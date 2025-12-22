"""
Drift Detector Module

Implements Population Stability Index (PSI) and Characteristic Stability Index (CSI)
for detecting data drift and triggering model retraining.

PSI Thresholds:
- PSI < 0.10: No significant drift
- 0.10 <= PSI < 0.25: Moderate drift (monitor)
- PSI >= 0.25: Significant drift (retrain required)

CSI Thresholds:
- CSI < 0.10: No significant drift per feature
- 0.10 <= CSI < 0.25: Moderate drift per feature
- CSI >= 0.25: Significant drift per feature
"""

import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Data class for drift detection results"""
    timestamp: str
    psi_score: float
    psi_threshold: float
    psi_status: str  # 'stable', 'moderate', 'drift'
    csi_scores: Dict[str, float]
    csi_drifted_features: List[str]
    csi_threshold: float
    overall_drift_detected: bool
    recommendation: str
    baseline_snapshot: str
    current_snapshot: str
    baseline_records: int
    current_records: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DriftDetector:
    """
    Detects model drift using PSI and CSI metrics.
    
    Population Stability Index (PSI):
    Measures the shift in the distribution of the target variable predictions
    or a key score between baseline (training) and current (production) data.
    
    Characteristic Stability Index (CSI):
    Measures the shift in the distribution of individual input features
    between baseline and current data.
    """
    
    # Drift thresholds
    PSI_THRESHOLD_STABLE = 0.10
    PSI_THRESHOLD_MODERATE = 0.25
    CSI_THRESHOLD = 0.25
    
    # Number of bins for discretization
    N_BINS = 10
    
    def __init__(
        self,
        db_path: str = "./data/risk_profiling.db",
        selected_features_path: str = "./models/selected_features.json",
        model_metadata_path: str = "./models/model_metadata.json"
    ):
        """
        Initialize the drift detector.
        
        Args:
            db_path: Path to the SQLite database with customer data
            selected_features_path: Path to selected features JSON
            model_metadata_path: Path to model metadata JSON
        """
        self.db_path = db_path
        self.selected_features_path = selected_features_path
        self.model_metadata_path = model_metadata_path
        
        # Load selected features
        self.selected_features = self._load_selected_features()
        
        # Exclude non-numeric and target features
        self.numeric_features = [
            f for f in self.selected_features 
            if f not in ['customer_segment', 'education', 'risk_profile', 'data_split', 'snapshot_date']
        ]
        
        logger.info(f"DriftDetector initialized with {len(self.numeric_features)} numeric features")
    
    def _load_selected_features(self) -> List[str]:
        """Load the list of selected features used by the model."""
        try:
            with open(self.selected_features_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Selected features file not found: {self.selected_features_path}")
            return []
    
    def _get_available_snapshots(self) -> List[str]:
        """Get list of available snapshot dates from database."""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT DISTINCT snapshot_date 
            FROM risk_profiling_monthly_data 
            ORDER BY snapshot_date
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df['snapshot_date'].tolist()
    
    def _load_data_by_snapshot(
        self, 
        snapshot_date: str,
        data_split: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific snapshot date.
        
        Args:
            snapshot_date: The snapshot date to load
            data_split: Optional filter for data split (Train/Validation/Test)
        """
        conn = sqlite3.connect(self.db_path)
        
        if data_split:
            query = f"""
                SELECT * FROM risk_profiling_monthly_data 
                WHERE snapshot_date = '{snapshot_date}' 
                AND data_split = '{data_split}'
            """
        else:
            query = f"""
                SELECT * FROM risk_profiling_monthly_data 
                WHERE snapshot_date = '{snapshot_date}'
            """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} records for snapshot {snapshot_date}")
        return df
    
    def _calculate_psi(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray,
        n_bins: int = None
    ) -> float:
        """
        Calculate Population Stability Index between baseline and current distributions.
        
        Formula: PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
        
        Args:
            baseline: Baseline (expected) distribution values
            current: Current (actual) distribution values
            n_bins: Number of bins for discretization
            
        Returns:
            PSI score (float)
        """
        n_bins = n_bins or self.N_BINS
        
        # Handle edge cases
        if len(baseline) == 0 or len(current) == 0:
            return 0.0
        
        # Create bins based on baseline distribution
        try:
            # Use percentile-based binning for robustness
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(baseline, percentiles)
            
            # Ensure unique bins
            bins = np.unique(bins)
            if len(bins) < 3:
                # Fall back to equal-width bins if percentile bins fail
                bins = np.linspace(baseline.min(), baseline.max(), n_bins + 1)
            
            # Calculate bin counts
            baseline_counts, _ = np.histogram(baseline, bins=bins)
            current_counts, _ = np.histogram(current, bins=bins)
            
            # Convert to percentages with epsilon to avoid division by zero
            epsilon = 1e-10
            baseline_pct = (baseline_counts + epsilon) / (len(baseline) + epsilon * len(baseline_counts))
            current_pct = (current_counts + epsilon) / (len(current) + epsilon * len(current_counts))
            
            # Calculate PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return float(psi)
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def _calculate_csi(
        self, 
        baseline_df: pd.DataFrame, 
        current_df: pd.DataFrame,
        features: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate Characteristic Stability Index for all features.
        
        CSI is essentially PSI applied to each individual feature.
        
        Args:
            baseline_df: Baseline data DataFrame
            current_df: Current data DataFrame
            features: List of features to analyze (uses numeric_features if None)
            
        Returns:
            Dictionary mapping feature names to CSI scores
        """
        features = features or self.numeric_features
        csi_scores = {}
        
        for feature in features:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature {feature} not found in data, skipping")
                continue
            
            try:
                baseline_values = baseline_df[feature].dropna().values
                current_values = current_df[feature].dropna().values
                
                if len(baseline_values) > 0 and len(current_values) > 0:
                    csi = self._calculate_psi(baseline_values, current_values)
                    csi_scores[feature] = round(csi, 4)
                else:
                    csi_scores[feature] = 0.0
                    
            except Exception as e:
                logger.warning(f"Error calculating CSI for {feature}: {e}")
                csi_scores[feature] = 0.0
        
        return csi_scores
    
    def _determine_psi_status(self, psi: float) -> str:
        """Determine drift status based on PSI score."""
        if psi < self.PSI_THRESHOLD_STABLE:
            return "stable"
        elif psi < self.PSI_THRESHOLD_MODERATE:
            return "moderate"
        else:
            return "drift"
    
    def _generate_recommendation(
        self, 
        psi_score: float, 
        psi_status: str,
        csi_drifted_features: List[str]
    ) -> str:
        """Generate actionable recommendation based on drift analysis."""
        
        if psi_status == "stable" and len(csi_drifted_features) == 0:
            return "No action required. Model is stable with no significant drift detected."
        
        elif psi_status == "moderate" or (0 < len(csi_drifted_features) <= 3):
            features_str = ", ".join(csi_drifted_features[:5]) if csi_drifted_features else "N/A"
            return (
                f"Monitor closely. Moderate drift detected (PSI: {psi_score:.4f}). "
                f"Drifted features: {features_str}. "
                "Consider scheduled retraining within next cycle."
            )
        
        else:
            features_str = ", ".join(csi_drifted_features[:5])
            if len(csi_drifted_features) > 5:
                features_str += f" (+{len(csi_drifted_features) - 5} more)"
            
            return (
                f"URGENT: Significant drift detected (PSI: {psi_score:.4f}). "
                f"Features with drift: {features_str}. "
                "Immediate model retraining recommended. "
                "Triggering automated retraining pipeline."
            )
    
    def detect_drift(
        self,
        baseline_snapshot: str = None,
        current_snapshot: str = None,
        use_training_as_baseline: bool = True
    ) -> DriftReport:
        """
        Perform comprehensive drift detection.
        
        Args:
            baseline_snapshot: Snapshot date to use as baseline (e.g., training data)
            current_snapshot: Snapshot date to compare against baseline
            use_training_as_baseline: If True, use 'Train' split as baseline
            
        Returns:
            DriftReport with PSI, CSI scores and recommendations
        """
        logger.info("Starting drift detection analysis...")
        
        # Get available snapshots
        snapshots = self._get_available_snapshots()
        
        if not snapshots:
            raise ValueError("No snapshots available in database")
        
        # Set default snapshots if not provided
        if baseline_snapshot is None:
            baseline_snapshot = snapshots[0]  # First/oldest snapshot
        if current_snapshot is None:
            current_snapshot = snapshots[-1]  # Latest snapshot
        
        logger.info(f"Baseline snapshot: {baseline_snapshot}")
        logger.info(f"Current snapshot: {current_snapshot}")
        
        # Load data
        if use_training_as_baseline:
            baseline_df = self._load_data_by_snapshot(baseline_snapshot, data_split='Train')
        else:
            baseline_df = self._load_data_by_snapshot(baseline_snapshot)
        
        current_df = self._load_data_by_snapshot(current_snapshot)
        
        if len(baseline_df) == 0 or len(current_df) == 0:
            raise ValueError("Insufficient data for drift detection")
        
        # Calculate PSI on risk_appetite_score (key predictor)
        # or we can use model prediction scores
        psi_feature = 'risk_appetite_score'
        if psi_feature in baseline_df.columns and psi_feature in current_df.columns:
            psi_score = self._calculate_psi(
                baseline_df[psi_feature].dropna().values,
                current_df[psi_feature].dropna().values
            )
        else:
            # Fallback: calculate average PSI across all features
            csi_all = self._calculate_csi(baseline_df, current_df)
            psi_score = np.mean(list(csi_all.values())) if csi_all else 0.0
        
        # Calculate CSI for all features
        csi_scores = self._calculate_csi(baseline_df, current_df)
        
        # Identify drifted features (CSI >= threshold)
        csi_drifted_features = [
            feature for feature, score in csi_scores.items() 
            if score >= self.CSI_THRESHOLD
        ]
        
        # Determine PSI status
        psi_status = self._determine_psi_status(psi_score)
        
        # Determine overall drift
        overall_drift = (
            psi_status == "drift" or 
            len(csi_drifted_features) >= 3  # Multiple features drifted
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            psi_score, psi_status, csi_drifted_features
        )
        
        # Create report
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            psi_score=round(psi_score, 4),
            psi_threshold=self.PSI_THRESHOLD_MODERATE,
            psi_status=psi_status,
            csi_scores=csi_scores,
            csi_drifted_features=csi_drifted_features,
            csi_threshold=self.CSI_THRESHOLD,
            overall_drift_detected=overall_drift,
            recommendation=recommendation,
            baseline_snapshot=baseline_snapshot,
            current_snapshot=current_snapshot,
            baseline_records=len(baseline_df),
            current_records=len(current_df)
        )
        
        logger.info(f"Drift detection complete. PSI: {psi_score:.4f}, Status: {psi_status}")
        logger.info(f"Drifted features: {len(csi_drifted_features)}")
        
        return report
    
    def save_report(self, report: DriftReport, output_dir: str = "./logs/drift_reports") -> str:
        """Save drift report to JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drift_report_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        with open(filepath, 'w') as f:
            f.write(report.to_json())
        
        logger.info(f"Drift report saved to: {filepath}")
        return str(filepath)
    
    def get_feature_drift_summary(self, report: DriftReport) -> pd.DataFrame:
        """Convert CSI scores to a sorted DataFrame for analysis."""
        df = pd.DataFrame([
            {"feature": k, "csi_score": v, "drifted": v >= self.CSI_THRESHOLD}
            for k, v in report.csi_scores.items()
        ])
        return df.sort_values("csi_score", ascending=False).reset_index(drop=True)


# Convenience functions for CLI usage
def check_drift(
    baseline_snapshot: str = None,
    current_snapshot: str = None,
    save_report: bool = True
) -> DriftReport:
    """
    Quick drift check function for CLI or script usage.
    
    Args:
        baseline_snapshot: Baseline snapshot date
        current_snapshot: Current snapshot date  
        save_report: Whether to save the report to file
        
    Returns:
        DriftReport object
    """
    detector = DriftDetector()
    report = detector.detect_drift(baseline_snapshot, current_snapshot)
    
    if save_report:
        detector.save_report(report)
    
    return report


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("DRIFT DETECTION ANALYSIS")
    print("=" * 60)
    
    detector = DriftDetector()
    
    try:
        report = detector.detect_drift()
        
        print(f"\nTimestamp: {report.timestamp}")
        print(f"Baseline Snapshot: {report.baseline_snapshot} ({report.baseline_records} records)")
        print(f"Current Snapshot: {report.current_snapshot} ({report.current_records} records)")
        print(f"\n--- PSI Analysis ---")
        print(f"PSI Score: {report.psi_score:.4f}")
        print(f"PSI Threshold: {report.psi_threshold}")
        print(f"PSI Status: {report.psi_status.upper()}")
        
        print(f"\n--- CSI Analysis ---")
        print(f"Features Analyzed: {len(report.csi_scores)}")
        print(f"Features with Drift (CSI >= {report.csi_threshold}): {len(report.csi_drifted_features)}")
        
        if report.csi_drifted_features:
            print(f"Drifted Features: {', '.join(report.csi_drifted_features[:10])}")
        
        print(f"\n--- Recommendation ---")
        print(report.recommendation)
        
        print(f"\n--- Overall Status ---")
        drift_status = "⚠️ DRIFT DETECTED - Retraining Required" if report.overall_drift_detected else "✅ Model Stable"
        print(drift_status)
        
        # Save report
        filepath = detector.save_report(report)
        print(f"\nReport saved to: {filepath}")
        
    except Exception as e:
        print(f"Error during drift detection: {e}")
        import traceback
        traceback.print_exc()
