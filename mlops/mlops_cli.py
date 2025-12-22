#!/usr/bin/env python3
"""
MLOps CLI Tool for Financial Advisor

Commands:
  drift-check    - Run drift detection analysis
  retrain        - Trigger model retraining
  pipeline       - Run full CI/CD pipeline
  schedule       - Start scheduled monitoring
  status         - Show monitoring status
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.drift_detector import DriftDetector, check_drift
from mlops.model_retrainer import ModelRetrainer, trigger_retraining
from mlops.monitoring_scheduler import CICDPipeline, run_cicd_pipeline


def cmd_drift_check(args):
    """Run drift detection analysis."""
    print("=" * 60)
    print("DRIFT DETECTION ANALYSIS")
    print("=" * 60)
    
    detector = DriftDetector()
    
    try:
        report = detector.detect_drift(
            baseline_snapshot=args.baseline,
            current_snapshot=args.current
        )
        
        print(f"\nTimestamp: {report.timestamp}")
        print(f"Baseline: {report.baseline_snapshot} ({report.baseline_records} records)")
        print(f"Current: {report.current_snapshot} ({report.current_records} records)")
        
        print(f"\n--- PSI Analysis ---")
        print(f"PSI Score: {report.psi_score:.4f}")
        print(f"PSI Threshold: {report.psi_threshold}")
        print(f"PSI Status: {report.psi_status.upper()}")
        
        print(f"\n--- CSI Analysis ---")
        print(f"Features Analyzed: {len(report.csi_scores)}")
        print(f"Features with Drift: {len(report.csi_drifted_features)}")
        
        if report.csi_drifted_features:
            print(f"\nDrifted Features:")
            for feat in report.csi_drifted_features[:10]:
                print(f"  - {feat}: {report.csi_scores.get(feat, 0):.4f}")
            if len(report.csi_drifted_features) > 10:
                print(f"  ... and {len(report.csi_drifted_features) - 10} more")
        
        print(f"\n--- Recommendation ---")
        print(report.recommendation)
        
        print(f"\n--- Status ---")
        if report.overall_drift_detected:
            print("⚠️  DRIFT DETECTED - Retraining Recommended")
            return 1
        else:
            print("✅ Model Stable - No Action Required")
            return 0
        
        if args.save:
            filepath = detector.save_report(report)
            print(f"\nReport saved: {filepath}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_retrain(args):
    """Trigger model retraining."""
    print("=" * 60)
    print("MODEL RETRAINING PIPELINE")
    print("=" * 60)
    
    try:
        result = trigger_retraining(
            snapshot_date=args.snapshot,
            force=args.force
        )
        
        print(f"\nRun ID: {result.get('run_id')}")
        print(f"Status: {result.get('status')}")
        
        if 'metrics' in result:
            print(f"\n--- Test Metrics ---")
            for metric, value in result['metrics'].get('test', {}).items():
                print(f"  {metric}: {value:.4f}")
        
        if 'comparison' in result:
            print(f"\n--- Model Comparison ---")
            print(f"  Accuracy Change: {result['comparison'].get('accuracy_change', 0):+.4f}")
            print(f"  ROC-AUC Change: {result['comparison'].get('roc_auc_change', 0):+.4f}")
        
        print(f"\n--- Result ---")
        if result.get('model_promoted'):
            print("✅ New model promoted to production")
            return 0
        else:
            print("⚠️  Model not promoted")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_pipeline(args):
    """Run full CI/CD pipeline."""
    print("=" * 60)
    print("CI/CD PIPELINE EXECUTION")
    print("=" * 60)
    
    try:
        result = run_cicd_pipeline(
            baseline_snapshot=args.baseline,
            current_snapshot=args.current,
            force_retrain=args.force
        )
        
        print(f"\nPipeline ID: {result.get('pipeline_id')}")
        
        for stage, details in result.get('stages', {}).items():
            status = details.get('status', 'unknown')
            icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "⏳"
            print(f"\n{icon} Stage: {stage.upper()}")
            for key, value in details.items():
                if key != 'status':
                    print(f"   {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_schedule(args):
    """Start scheduled monitoring."""
    print("=" * 60)
    print("SCHEDULED MONITORING")
    print("=" * 60)
    
    from mlops.monitoring_scheduler import MonitoringScheduler
    
    scheduler = MonitoringScheduler(
        psi_threshold=args.psi_threshold,
        csi_threshold=args.csi_threshold
    )
    
    print(f"\nConfiguration:")
    print(f"  Interval: {args.interval}")
    print(f"  Time: {args.time}")
    print(f"  PSI Threshold: {args.psi_threshold}")
    print(f"  CSI Threshold: {args.csi_threshold}")
    
    scheduler.schedule_monitoring(
        interval=args.interval,
        time_of_day=args.time
    )
    
    print(f"\n✅ Monitoring scheduled. Press Ctrl+C to stop.\n")
    
    try:
        scheduler.start(blocking=True)
    except KeyboardInterrupt:
        print("\n\nStopping scheduler...")
        scheduler.stop()
        print("✅ Scheduler stopped")
    
    return 0


def cmd_status(args):
    """Show monitoring status."""
    print("=" * 60)
    print("MONITORING STATUS")
    print("=" * 60)
    
    # Check for recent reports
    drift_reports = Path("./logs/drift_reports")
    retrain_reports = Path("./logs/retraining_reports")
    
    print("\n--- Recent Drift Checks ---")
    if drift_reports.exists():
        reports = sorted(drift_reports.glob("*.json"), reverse=True)[:5]
        for report_path in reports:
            with open(report_path) as f:
                report = json.load(f)
            print(f"  {report_path.name}")
            print(f"    PSI: {report.get('psi_score', 'N/A'):.4f}, "
                  f"Status: {report.get('psi_status', 'N/A')}, "
                  f"Drift: {report.get('overall_drift_detected', 'N/A')}")
    else:
        print("  No drift reports found")
    
    print("\n--- Recent Retraining ---")
    if retrain_reports.exists():
        reports = sorted(retrain_reports.glob("*.json"), reverse=True)[:5]
        for report_path in reports:
            with open(report_path) as f:
                report = json.load(f)
            print(f"  {report_path.name}")
            print(f"    Status: {report.get('status', 'N/A')}, "
                  f"Promoted: {report.get('model_promoted', 'N/A')}")
    else:
        print("  No retraining reports found")
    
    print("\n--- Current Model ---")
    model_metadata = Path("./models/model_metadata.json")
    if model_metadata.exists():
        with open(model_metadata) as f:
            metadata = json.load(f)
        print(f"  Training Date: {metadata.get('training_date', 'N/A')}")
        print(f"  Features: {metadata.get('n_features', 'N/A')}")
        print(f"  Test Accuracy: {metadata.get('test_accuracy', 'N/A')}")
        print(f"  Test ROC-AUC: {metadata.get('test_roc_auc', 'N/A')}")
    else:
        print("  Model metadata not found")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="MLOps CLI for Financial Advisor Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlops_cli.py drift-check
  python mlops_cli.py drift-check --baseline 2025-11-30 --current 2025-12-31
  python mlops_cli.py retrain --force
  python mlops_cli.py pipeline
  python mlops_cli.py schedule --interval monthly --time 02:00
  python mlops_cli.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # drift-check command
    drift_parser = subparsers.add_parser('drift-check', help='Run drift detection')
    drift_parser.add_argument('--baseline', type=str, help='Baseline snapshot date')
    drift_parser.add_argument('--current', type=str, help='Current snapshot date')
    drift_parser.add_argument('--save', action='store_true', help='Save report to file')
    
    # retrain command
    retrain_parser = subparsers.add_parser('retrain', help='Trigger model retraining')
    retrain_parser.add_argument('--snapshot', type=str, help='Snapshot date to train on')
    retrain_parser.add_argument('--force', action='store_true', help='Force promotion')
    
    # pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full CI/CD pipeline')
    pipeline_parser.add_argument('--baseline', type=str, help='Baseline snapshot date')
    pipeline_parser.add_argument('--current', type=str, help='Current snapshot date')
    pipeline_parser.add_argument('--force', action='store_true', help='Force retraining')
    
    # schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Start scheduled monitoring')
    schedule_parser.add_argument('--interval', type=str, default='monthly',
                                 choices=['daily', 'weekly', 'monthly'],
                                 help='Monitoring interval (default: monthly)')
    schedule_parser.add_argument('--time', type=str, default='02:00',
                                help='Time of day (HH:MM)')
    schedule_parser.add_argument('--psi-threshold', type=float, default=0.25,
                                help='PSI threshold for retraining')
    schedule_parser.add_argument('--csi-threshold', type=float, default=0.25,
                                help='CSI threshold per feature')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Show monitoring status')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute command
    commands = {
        'drift-check': cmd_drift_check,
        'retrain': cmd_retrain,
        'pipeline': cmd_pipeline,
        'schedule': cmd_schedule,
        'status': cmd_status
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
