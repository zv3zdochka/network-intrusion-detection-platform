"""
Flask application factory
"""

import os
import sys
from pathlib import Path
from flask import Flask

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_app(config=None):
    """Create and configure Flask application"""

    app = Flask(__name__,
                template_folder=str(Path(__file__).parent / 'templates'),
                static_folder=str(Path(__file__).parent / 'static'))

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
    app.config['PROJECT_ROOT'] = PROJECT_ROOT
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

    if config:
        app.config.update(config)

    # Register blueprints
    from .routes.main import main_bp
    from .routes.realtime import realtime_bp
    from .routes.offline import offline_bp
    from .routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(realtime_bp, url_prefix='/realtime')
    app.register_blueprint(offline_bp, url_prefix='/offline')
    app.register_blueprint(api_bp, url_prefix='/api')

    return app


def run_app(host='127.0.0.1', port=5000, debug=True):
    """Run the Flask application"""
    app = create_app()
    print(f"\n{'=' * 60}")
    print(f"  Network IDS Web Interface")
    print(f"  Running at: http://{host}:{port}")
    print(f"{'=' * 60}\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_app()
