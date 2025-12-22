"""
Monitoring Scheduler Module

Orchestrates the complete CI/CD pipeline:
1. Scheduled drift detection (PSI/CSI monitoring)
2. Automatic retraining trigger when drift is detected
3. Model promotion with validation
4. Notification and logging
"""

import os
import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from pathlib import Path
import logging
import threading

from .drift_detector import DriftDetector, DriftReport
from .model_retrainer import ModelRetrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """
    Automated monitoring scheduler that:
    - Runs drift detection on a schedule (daily, weekly, monthly)
    - Triggers retraining when drift is detected
    - Logs all monitoring activities
    - Supports webhooks for notifications
    """
    
    def __init__(
        self,
        db_path: str = "./data/risk_profiling.db",
        model_dir: str = "./models",
        log_dir: str = "./logs/monitoring",
        psi_threshold: float = 0.25,
        csi_threshold: float = 0.25,
        min_drifted_features: int = 3
    ):
        """
        Initialize the monitoring scheduler.
        
        Args:
            db_path: Path to SQLite database
            model_dir: Directory with model artifacts
            log_dir: Directory for monitoring logs
            psi_threshold: PSI threshold for triggering retraining
            csi_threshold: CSI threshold per feature
            min_drifted_features: Minimum drifted features to trigger retraining
        """
        self.db_path = db_path
        self.model_dir = model_dir
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.psi_threshold = psi_threshold
        self.csi_threshold = csi_threshold
        self.min_drifted_features = min_drifted_features
        
        # Initialize components
        self.drift_detector = DriftDetector(
            db_path=db_path,
            selected_features_path=f"{model_dir}/selected_features.json"
        )
        self.drift_detector.PSI_THRESHOLD_MODERATE = psi_threshold
        self.drift_detector.CSI_THRESHOLD = csi_threshold
        
        self.model_retrainer = ModelRetrainer(
            db_path=db_path,
            model_dir=model_dir
        )
        
        # Callback for notifications
        self.notification_callback: Optional[Callable] = None
        
        # Tracking
        self.last_check: Optional[datetime] = None
        self.last_retrain: Optional[datetime] = None
        self.is_running = False
        
        logger.info("MonitoringScheduler initialized")
        logger.info(f"PSI Threshold: {psi_threshold}, CSI Threshold: {csi_threshold}")
    
    def set_notification_callback(self, callback: Callable):
        """Set callback function for notifications."""
        self.notification_callback = callback
    
    def _send_notification(self, title: str, message: str, level: str = "info"):
        """Send notification via callback if configured."""
        notification = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'level': level
        }
        
        # Log notification
        log_path = self.log_dir / "notifications.jsonl"
        with open(log_path, 'a') as f:
            f.write(json.dumps(notification) + "\n")
        
        # Call callback if set
        if self.notification_callback:
            try:
                self.notification_callback(notification)
            except Exception as e:
                logger.error(f"Notification callback error: {e}")
        
        logger.info(f"[{level.upper()}] {title}: {message}")
    
    def run_drift_check(
        self,
        baseline_snapshot: str = None,
        current_snapshot: str = None
    ) -> Dict:
        """
        Run a single drift detection check.
        
        Args:
            baseline_snapshot: Baseline snapshot date
            current_snapshot: Current snapshot date
            
        Returns:
            Dictionary with check results and actions taken
        """
        check_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting drift check - ID: {check_id}")
        
        result = {
            'check_id': check_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'started'
        }
        
        try:
            # Run drift detection
            report = self.drift_detector.detect_drift(
                baseline_snapshot=baseline_snapshot,
                current_snapshot=current_snapshot
            )
            
            result['drift_report'] = {
                'psi_score': report.psi_score,
                'psi_status': report.psi_status,
                'csi_drifted_features_count': len(report.csi_drifted_features),
                'csi_drifted_features': report.csi_drifted_features[:10],
                'overall_drift_detected': report.overall_drift_detected,
                'recommendation': report.recommendation
            }
            
            # Save drift report
            report_path = self.drift_detector.save_report(report, str(self.log_dir / "drift_reports"))
            result['report_path'] = report_path
            
            # Determine if retraining is needed
            should_retrain = (
                report.psi_score >= self.psi_threshold or
                len(report.csi_drifted_features) >= self.min_drifted_features
            )
            
            result['retraining_triggered'] = should_retrain
            
            if should_retrain:
                self._send_notification(
                    "🚨 Model Drift Detected",
                    f"PSI: {report.psi_score:.4f}, Drifted Features: {len(report.csi_drifted_features)}. "
                    f"Triggering automatic retraining.",
                    level="warning"
                )
                
                # Trigger retraining
                logger.info("Drift threshold exceeded - triggering retraining")
                retrain_result = self.model_retrainer.retrain()
                result['retraining_result'] = {
                    'status': retrain_result.get('status'),
                    'model_promoted': retrain_result.get('model_promoted'),
                    'report_path': retrain_result.get('report_path')
                }
                
                self.last_retrain = datetime.now()
                
                if retrain_result.get('model_promoted'):
                    self._send_notification(
                        "✅ Model Retrained Successfully",
                        f"New model promoted. Test Accuracy: {retrain_result.get('metrics', {}).get('test', {}).get('accuracy', 'N/A'):.4f}",
                        level="success"
                    )
                else:
                    self._send_notification(
                        "⚠️ Retraining Completed - Model Not Promoted",
                        f"New model performance was not sufficient. Check logs for details.",
                        level="warning"
                    )
            else:
                self._send_notification(
                    "✅ Model Stability Check Passed",
                    f"PSI: {report.psi_score:.4f}, Status: {report.psi_status}. No retraining needed.",
                    level="info"
                )
            
            result['status'] = 'completed'
            self.last_check = datetime.now()
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            
            self._send_notification(
                "❌ Drift Check Failed",
                f"Error during drift detection: {str(e)}",
                level="error"
            )
            
            logger.error(f"Drift check failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Save check result
        result_path = self.log_dir / f"check_{check_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    def schedule_monitoring(
        self,
        interval: str = "monthly",
        time_of_day: str = "02:00"
    ):
        """
        Schedule regular drift monitoring.
        
        Args:
            interval: 'daily', 'weekly', 'monthly' (default: monthly)
            time_of_day: Time to run (HH:MM format)
        """
        logger.info(f"Scheduling {interval} monitoring at {time_of_day}")
        
        job = schedule.every()
        
        if interval == "daily":
            job = schedule.every().day.at(time_of_day)
        elif interval == "weekly":
            job = schedule.every().monday.at(time_of_day)
        elif interval == "monthly":
            # Run on the 1st of each month
            job = schedule.every().day.at(time_of_day)
            # Will need to check day in the job
        else:
            raise ValueError(f"Invalid interval: {interval}")
        
        def monitoring_job():
            if interval == "monthly":
                if datetime.now().day != 1:
                    return
            logger.info("Running scheduled drift check...")
            self.run_drift_check()
        
        job.do(monitoring_job)
        
        logger.info(f"Monitoring scheduled: {interval} at {time_of_day}")
        
        return job
    
    def start(self, blocking: bool = True):
        """
        Start the monitoring scheduler.
        
        Args:
            blocking: If True, blocks the main thread. If False, runs in background.
        """
        self.is_running = True
        logger.info("Starting monitoring scheduler...")
        
        if blocking:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        else:
            def run_schedule():
                while self.is_running:
                    schedule.run_pending()
                    time.sleep(60)
            
            thread = threading.Thread(target=run_schedule, daemon=True)
            thread.start()
            logger.info("Monitoring running in background thread")
    
    def stop(self):
        """Stop the monitoring scheduler."""
        self.is_running = False
        schedule.clear()
        logger.info("Monitoring scheduler stopped")
    
    def get_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'is_running': self.is_running,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'psi_threshold': self.psi_threshold,
            'csi_threshold': self.csi_threshold,
            'pending_jobs': len(schedule.jobs)
        }


class CICDPipeline:
    """
    Complete CI/CD Pipeline for model lifecycle management.
    
    Stages:
    1. Monitor: Regular drift detection
    2. Detect: Identify when retraining is needed
    3. Retrain: Automated model retraining
    4. Validate: Compare new model performance
    5. Deploy: Promote new model to production
    6. Notify: Alert stakeholders
    """
    
    def __init__(
        self,
        db_path: str = "./data/risk_profiling.db",
        model_dir: str = "./models",
        psi_threshold: float = 0.25,
        csi_threshold: float = 0.25
    ):
        self.scheduler = MonitoringScheduler(
            db_path=db_path,
            model_dir=model_dir,
            psi_threshold=psi_threshold,
            csi_threshold=csi_threshold
        )
        
        self.pipeline_log = Path("./logs/cicd_pipeline.jsonl")
        self.pipeline_log.parent.mkdir(parents=True, exist_ok=True)
    
    def _log_event(self, stage: str, status: str, details: Dict = None):
        """Log pipeline event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'status': status,
            'details': details or {}
        }
        with open(self.pipeline_log, 'a') as f:
            f.write(json.dumps(event) + "\n")
    
    def run_pipeline(
        self,
        baseline_snapshot: str = None,
        current_snapshot: str = None,
        force_retrain: bool = False
    ) -> Dict:
        """
        Execute the complete CI/CD pipeline.
        
        Args:
            baseline_snapshot: Baseline data snapshot
            current_snapshot: Current data snapshot
            force_retrain: Force retraining regardless of drift
            
        Returns:
            Pipeline execution results
        """
        pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"=" * 60)
        logger.info(f"CI/CD PIPELINE EXECUTION - {pipeline_id}")
        logger.info(f"=" * 60)
        
        result = {
            'pipeline_id': pipeline_id,
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        # Stage 1: Monitor & Detect
        self._log_event("monitor", "started")
        logger.info("\n📊 Stage 1: DRIFT DETECTION")
        
        try:
            drift_report = self.scheduler.drift_detector.detect_drift(
                baseline_snapshot=baseline_snapshot,
                current_snapshot=current_snapshot
            )
            
            result['stages']['monitor'] = {
                'status': 'completed',
                'psi_score': drift_report.psi_score,
                'psi_status': drift_report.psi_status,
                'drifted_features': len(drift_report.csi_drifted_features),
                'drift_detected': drift_report.overall_drift_detected
            }
            
            self._log_event("monitor", "completed", result['stages']['monitor'])
            
        except Exception as e:
            result['stages']['monitor'] = {'status': 'failed', 'error': str(e)}
            self._log_event("monitor", "failed", {'error': str(e)})
            logger.error(f"Monitor stage failed: {e}")
            return result
        
        # Determine if we should proceed
        should_proceed = force_retrain or drift_report.overall_drift_detected
        
        if not should_proceed:
            logger.info("\n✅ No drift detected. Pipeline complete.")
            result['stages']['decision'] = {
                'action': 'skip_retraining',
                'reason': 'No significant drift detected'
            }
            return result
        
        # Stage 2: Retrain
        self._log_event("retrain", "started")
        logger.info("\n🔧 Stage 2: MODEL RETRAINING")
        
        try:
            retrain_result = self.scheduler.model_retrainer.retrain(force=force_retrain)
            
            result['stages']['retrain'] = {
                'status': retrain_result.get('status'),
                'metrics': retrain_result.get('metrics', {}).get('test', {}),
                'comparison': retrain_result.get('comparison', {})
            }
            
            self._log_event("retrain", retrain_result.get('status'), result['stages']['retrain'])
            
        except Exception as e:
            result['stages']['retrain'] = {'status': 'failed', 'error': str(e)}
            self._log_event("retrain", "failed", {'error': str(e)})
            logger.error(f"Retrain stage failed: {e}")
            return result
        
        # Stage 3: Validate & Deploy
        self._log_event("deploy", "started")
        logger.info("\n🚀 Stage 3: VALIDATION & DEPLOYMENT")
        
        model_promoted = retrain_result.get('model_promoted', False)
        
        result['stages']['deploy'] = {
            'status': 'completed',
            'model_promoted': model_promoted,
            'reason': retrain_result.get('comparison', {}).get('reason', 'Unknown')
        }
        
        if model_promoted:
            logger.info("✅ New model deployed to production")
        else:
            logger.info("⚠️ New model not promoted - performance validation failed")
        
        self._log_event("deploy", "completed", result['stages']['deploy'])
        
        # Stage 4: Notify
        logger.info("\n📬 Stage 4: NOTIFICATION")
        
        notification_msg = (
            f"Pipeline {pipeline_id} completed. "
            f"Drift: PSI={drift_report.psi_score:.4f}. "
            f"Model {'promoted' if model_promoted else 'not promoted'}."
        )
        
        result['stages']['notify'] = {
            'status': 'completed',
            'message': notification_msg
        }
        
        self._log_event("notify", "completed", result['stages']['notify'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE - Model Promoted: {model_promoted}")
        logger.info(f"{'='*60}")
        
        return result
    
    def schedule_pipeline(
        self,
        interval: str = "monthly",
        time_of_day: str = "02:00"
    ):
        """Schedule regular pipeline execution (default: monthly)."""
        self.scheduler.schedule_monitoring(interval, time_of_day)
        return self.scheduler


def run_cicd_pipeline(
    baseline_snapshot: str = None,
    current_snapshot: str = None,
    force_retrain: bool = False
) -> Dict:
    """
    Convenience function to run the CI/CD pipeline.
    
    Args:
        baseline_snapshot: Baseline data snapshot
        current_snapshot: Current data snapshot  
        force_retrain: Force retraining regardless of drift
        
    Returns:
        Pipeline execution results
    """
    pipeline = CICDPipeline()
    return pipeline.run_pipeline(
        baseline_snapshot=baseline_snapshot,
        current_snapshot=current_snapshot,
        force_retrain=force_retrain
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CI/CD PIPELINE - MODEL MONITORING & RETRAINING")
    print("=" * 60)
    
    # Run single check
    result = run_cicd_pipeline()
    
    print("\n--- Pipeline Results ---")
    print(json.dumps(result, indent=2, default=str))
