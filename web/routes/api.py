"""
REST API routes
"""

from flask import Blueprint, jsonify, current_app
from pathlib import Path
import json

api_bp = Blueprint('api', __name__)


@api_bp.route('/model-info')
def model_info():
    """Get model information"""
    project_root = current_app.config['PROJECT_ROOT']

    info = {
        'name': 'XGB_regularized',
        'type': 'XGBClassifier',
        'features': 79,
        'threshold': 0.5
    }

    # Try to load experiment results
    try:
        results_path = project_root / 'training_artifacts' / 'experiment_results.csv'
        if results_path.exists():
            import pandas as pd
            df = pd.read_csv(results_path)
            best = df[df['model_name'] == 'XGB_regularized']
            if len(best) > 0:
                info['test_f1'] = round(best['test_f1'].values[0], 4)
                info['test_auc'] = round(best['test_roc_auc'].values[0], 4)
    except:
        pass

    # Load feature importance
    try:
        importance_path = project_root / 'training_artifacts' / 'feature_importance.csv'
        if importance_path.exists():
            import pandas as pd
            df = pd.read_csv(importance_path)
            info['top_features'] = df.head(10).to_dict('records')
    except:
        pass

    return jsonify(info)


@api_bp.route('/sessions')
def list_sessions():
    """List previous analysis sessions"""
    project_root = current_app.config['PROJECT_ROOT']
    sessions_dir = project_root / 'realtime_testing_artifacts'

    sessions = []
    if sessions_dir.exists():
        for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
            if session_dir.is_dir():
                stats_file = session_dir / 'stats.json'
                if stats_file.exists():
                    try:
                        with open(stats_file) as f:
                            stats = json.load(f)
                        sessions.append({
                            'id': session_dir.name,
                            'start_time': stats.get('start_time'),
                            'runtime': round(stats.get('runtime_seconds', 0)),
                            'flows': stats.get('total_flows', 0),
                            'attacks': stats.get('total_attacks', 0)
                        })
                    except:
                        pass

    return jsonify(sessions[:20])  # Last 20 sessions


@api_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})
