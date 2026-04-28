from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'

    def ready(self):
        # Prevent running in management commands like migrate/makemigrations
        import sys
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv:
            from .ml_engine import MLEngine
            MLEngine() # Initialize singleton
