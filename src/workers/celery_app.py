"""
Celery Application Configuration
=================================

Multi-tier worker architecture with auto-scaling support.

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

import os
from celery import Celery
from kombu import Queue, Exchange

# Initialize Celery app
celery_app = Celery('e2i_causal_analytics')

# Redis connection from environment
REDIS_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
REDIS_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

# =============================================================================
# CELERY CONFIGURATION
# =============================================================================

celery_app.conf.update(
    # Broker settings
    broker_url=REDIS_URL,
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,

    # Result backend
    result_backend=REDIS_BACKEND,
    result_expires=3600,  # 1 hour
    result_extended=True,

    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Performance settings
    worker_prefetch_multiplier=1,  # Prefetch 1 task to avoid blocking
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,  # Requeue if worker crashes

    # Time limits
    task_time_limit=7200,  # 2 hours hard limit
    task_soft_time_limit=6600,  # 1h 50m soft limit

    # Retry settings
    task_autoretry_for=(Exception,),
    task_retry_kwargs={'max_retries': 3},
    task_retry_backoff=True,
    task_retry_backoff_max=600,  # 10 minutes max backoff

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# =============================================================================
# QUEUE DEFINITIONS
# =============================================================================

# Default exchange
default_exchange = Exchange('default', type='direct')

celery_app.conf.task_queues = (
    # Light worker queues
    Queue('default', exchange=default_exchange, routing_key='default'),
    Queue('quick', exchange=default_exchange, routing_key='quick'),
    Queue('api', exchange=default_exchange, routing_key='api'),

    # Medium worker queues
    Queue('analytics', exchange=default_exchange, routing_key='analytics'),
    Queue('reports', exchange=default_exchange, routing_key='reports'),
    Queue('aggregations', exchange=default_exchange, routing_key='aggregations'),

    # Heavy worker queues
    Queue('shap', exchange=default_exchange, routing_key='shap'),
    Queue('causal', exchange=default_exchange, routing_key='causal'),
    Queue('ml', exchange=default_exchange, routing_key='ml'),
    Queue('twins', exchange=default_exchange, routing_key='twins'),
)

# Default queue
celery_app.conf.task_default_queue = 'default'
celery_app.conf.task_default_exchange = 'default'
celery_app.conf.task_default_routing_key = 'default'

# =============================================================================
# TASK ROUTING
# =============================================================================

celery_app.conf.task_routes = {
    # -------------------------------------------------------------------------
    # Light Worker Tasks (2 CPUs, 2GB RAM)
    # -------------------------------------------------------------------------
    # API-related tasks
    'src.tasks.api.*': {'queue': 'api'},
    'src.tasks.fetch_*': {'queue': 'api'},
    'src.tasks.get_*': {'queue': 'api'},

    # Cache operations
    'src.tasks.cache.*': {'queue': 'quick'},
    'src.tasks.invalidate_cache': {'queue': 'quick'},
    'src.tasks.warm_cache': {'queue': 'quick'},

    # Notifications
    'src.tasks.notify.*': {'queue': 'quick'},
    'src.tasks.send_email': {'queue': 'quick'},
    'src.tasks.send_alert': {'queue': 'quick'},

    # Quick data operations
    'src.tasks.save_*': {'queue': 'quick'},
    'src.tasks.update_*': {'queue': 'quick'},
    'src.tasks.delete_*': {'queue': 'quick'},

    # -------------------------------------------------------------------------
    # Medium Worker Tasks (4 CPUs, 8GB RAM)
    # -------------------------------------------------------------------------
    # Analytics and aggregations
    'src.tasks.calculate_metrics': {'queue': 'analytics'},
    'src.tasks.aggregate_*': {'queue': 'aggregations'},
    'src.tasks.compute_statistics': {'queue': 'analytics'},

    # Report generation
    'src.tasks.generate_report': {'queue': 'reports'},
    'src.tasks.export_report': {'queue': 'reports'},
    'src.tasks.create_dashboard': {'queue': 'reports'},

    # Data processing
    'src.tasks.process_batch': {'queue': 'analytics'},
    'src.tasks.transform_data': {'queue': 'analytics'},

    # -------------------------------------------------------------------------
    # Heavy Worker Tasks (16 CPUs, 32GB RAM)
    # -------------------------------------------------------------------------
    # SHAP explanations
    'src.tasks.shap_explain': {'queue': 'shap'},
    'src.tasks.shap_explainer.*': {'queue': 'shap'},
    'src.tasks.compute_shap_values': {'queue': 'shap'},
    'src.tasks.shap_summary': {'queue': 'shap'},

    # Causal inference
    'src.tasks.causal_refutation': {'queue': 'causal'},
    'src.tasks.causal_sensitivity': {'queue': 'causal'},
    'src.tasks.estimate_effect': {'queue': 'causal'},
    'src.tasks.refutation.*': {'queue': 'causal'},
    'src.tasks.sensitivity_analysis': {'queue': 'causal'},
    'src.tasks.bootstrap_*': {'queue': 'causal'},

    # ML training and cross-validation
    'src.tasks.train_model': {'queue': 'ml'},
    'src.tasks.cross_validate_model': {'queue': 'ml'},
    'src.tasks.hyperparameter_tune': {'queue': 'ml'},
    'src.tasks.train_*': {'queue': 'ml'},
    'src.tasks.fit_*': {'queue': 'ml'},

    # Digital twin generation
    'src.tasks.generate_twins': {'queue': 'twins'},
    'src.tasks.twin.*': {'queue': 'twins'},
    'src.tasks.train_twin_model': {'queue': 'ml'},
    'src.tasks.simulate_population': {'queue': 'twins'},
}

# =============================================================================
# BEAT SCHEDULE (for scheduler service)
# =============================================================================

celery_app.conf.beat_schedule = {
    # Drift monitoring every 6 hours
    'monitor-drift': {
        'task': 'src.tasks.monitor_model_drift',
        'schedule': 21600.0,  # 6 hours
        'options': {'queue': 'analytics'}
    },

    # Health checks every hour
    'health-check': {
        'task': 'src.tasks.health_check',
        'schedule': 3600.0,  # 1 hour
        'options': {'queue': 'quick'}
    },

    # Cache cleanup every day
    'cache-cleanup': {
        'task': 'src.tasks.cleanup_old_cache',
        'schedule': 86400.0,  # 24 hours
        'options': {'queue': 'quick'}
    },

    # Queue metrics every 5 minutes (for autoscaler)
    'queue-metrics': {
        'task': 'src.tasks.collect_queue_metrics',
        'schedule': 300.0,  # 5 minutes
        'options': {'queue': 'quick'}
    },
}

# =============================================================================
# AUTO-DISCOVERY
# =============================================================================

# Auto-discover tasks in these modules
celery_app.autodiscover_tasks([
    'src.tasks',
    'src.mlops',
    'src.causal',
    'src.digital_twin',
    'src.agents',
])

# =============================================================================
# CUSTOM CONFIGURATION
# =============================================================================

@celery_app.task(bind=True, name='src.tasks.debug_task')
def debug_task(self):
    """Debug task for testing worker connectivity."""
    return f'Request: {self.request!r}'


# Worker tier information
def get_worker_info():
    """Get current worker tier information."""
    worker_type = os.getenv('WORKER_TYPE', 'unknown')
    return {
        'type': worker_type,
        'queues': {
            'light': ['default', 'quick', 'api'],
            'medium': ['analytics', 'reports', 'aggregations'],
            'heavy': ['shap', 'causal', 'ml', 'twins'],
        }.get(worker_type, []),
    }
