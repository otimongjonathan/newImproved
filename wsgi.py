from app import create_app
from app.services.data_service import data_service
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = create_app('production')

# Load data on startup
with app.app_context():
    try:
        data_service.load_data(app.config['DATA_FILES'])
        logger.info("✓ Data files loaded successfully")
    except Exception as e:
        logger.error(f"⚠ Error loading data files: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    workers = int(os.environ.get('GUNICORN_WORKERS', 2))
    
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(host=host, port=port, debug=True)
    else:
        import gunicorn.app.base
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
                
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)
                    
            def load(self):
                return self.application
        
        options = {
            'bind': f'{host}:{port}',
            'workers': workers,
            'worker_class': 'sync',
            'threads': 2,
            'timeout': 120,
            'preload_app': True
        }
        
        StandaloneApplication(app, options).run()

"""
WSGI entry point for production deployment (Heroku/Railway/etc.)
"""
from app import app

if __name__ == "__main__":
    app.run()