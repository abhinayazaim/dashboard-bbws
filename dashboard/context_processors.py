from .ml_engine import MLEngine

def threshold_context(request):
    """
    Makes the current model threshold available to all templates.
    """
    engine = MLEngine()
    return {
        'global_threshold': engine.get_threshold()
    }
