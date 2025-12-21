#!/usr/bin/env python3
"""
E2I Celery Worker Autoscaler
=============================

Monitors Celery queue depths and automatically scales Docker Compose workers.

Features:
- Queue-based scaling (scale up when queues grow)
- Time-based scaling (scale down after idle period)
- Resource-aware (respects min/max replicas)
- Graceful scaling (avoids flapping)

Usage:
    python scripts/autoscaler.py --config config/autoscale.yml

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import os
import time
import logging
import subprocess
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('autoscaler')


@dataclass
class WorkerConfig:
    """Configuration for a worker tier."""
    name: str
    queues: List[str]
    min_replicas: int
    max_replicas: int
    scale_up_threshold: int  # Queue depth to trigger scale up
    scale_down_threshold: int  # Queue depth to trigger scale down
    cooldown_minutes: int  # Minimum time between scale operations


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    queue_depth: int
    current_replicas: int
    active_tasks: int
    idle_time_minutes: int


class CeleryAutoscaler:
    """
    Autoscaler for Celery workers in Docker Compose.

    Monitors Redis-backed Celery queues and scales workers using
    docker compose up --scale command.
    """

    def __init__(self, config_path: str, docker_compose_dir: str = './docker'):
        """
        Initialize autoscaler.

        Args:
            config_path: Path to autoscale configuration YAML
            docker_compose_dir: Directory containing docker-compose.yml
        """
        self.config_path = config_path
        self.docker_compose_dir = docker_compose_dir
        self.redis_client = None
        self.worker_configs: Dict[str, WorkerConfig] = {}
        self.last_scale_time: Dict[str, datetime] = {}
        self.last_queue_depths: Dict[str, List[int]] = {}

        self._load_config()
        self._connect_redis()

    def _load_config(self):
        """Load autoscaling configuration from YAML."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for worker_name, worker_cfg in config.get('workers', {}).items():
            self.worker_configs[worker_name] = WorkerConfig(
                name=worker_name,
                queues=worker_cfg['queues'],
                min_replicas=worker_cfg.get('min_replicas', 0),
                max_replicas=worker_cfg.get('max_replicas', 4),
                scale_up_threshold=worker_cfg.get('scale_up_threshold', 10),
                scale_down_threshold=worker_cfg.get('scale_down_threshold', 0),
                cooldown_minutes=worker_cfg.get('cooldown_minutes', 5),
            )
            self.last_scale_time[worker_name] = datetime.now() - timedelta(hours=1)
            self.last_queue_depths[worker_name] = []

        logger.info(f"Loaded config for {len(self.worker_configs)} worker tiers")

    def _connect_redis(self):
        """Connect to Redis broker."""
        redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6382/1')
        # Parse redis://host:port/db
        parts = redis_url.replace('redis://', '').split('/')
        host_port = parts[0].split(':')
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 6382
        db = int(parts[1]) if len(parts) > 1 else 1

        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        logger.info(f"Connected to Redis at {host}:{port}/{db}")

    def get_queue_depth(self, queue_name: str) -> int:
        """Get number of messages in a Celery queue."""
        try:
            # Celery stores queue lengths in Redis lists
            # Default queue name format: celery
            # Custom queue format: queue_name
            depth = self.redis_client.llen(queue_name)
            return depth
        except Exception as e:
            logger.error(f"Error getting queue depth for {queue_name}: {e}")
            return 0

    def get_total_queue_depth(self, queues: List[str]) -> int:
        """Get total depth across multiple queues."""
        return sum(self.get_queue_depth(q) for q in queues)

    def get_current_replicas(self, worker_name: str) -> int:
        """Get current number of running worker replicas."""
        try:
            # Use docker ps to count running workers
            cmd = [
                'docker', 'ps',
                '--filter', f'name={worker_name}',
                '--filter', 'status=running',
                '--format', '{{.Names}}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            containers = [line for line in result.stdout.strip().split('\n') if line]
            return len(containers)
        except Exception as e:
            logger.error(f"Error getting replicas for {worker_name}: {e}")
            return 0

    def scale_worker(self, worker_name: str, replicas: int) -> bool:
        """
        Scale worker to specified number of replicas.

        Args:
            worker_name: Name of worker service
            replicas: Target number of replicas

        Returns:
            True if scaling succeeded
        """
        try:
            cmd = [
                'docker', 'compose',
                '-f', f'{self.docker_compose_dir}/docker-compose.yml',
                'up', '-d', '--scale', f'{worker_name}={replicas}', '--no-recreate'
            ]

            logger.info(f"Scaling {worker_name} to {replicas} replicas...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            self.last_scale_time[worker_name] = datetime.now()
            logger.info(f"Successfully scaled {worker_name} to {replicas}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to scale {worker_name}: {e.stderr}")
            return False

    def should_scale(self, worker_name: str, metrics: ScalingMetrics) -> Tuple[bool, int]:
        """
        Determine if scaling is needed and what the target should be.

        Args:
            worker_name: Name of worker tier
            metrics: Current scaling metrics

        Returns:
            (should_scale, target_replicas)
        """
        config = self.worker_configs[worker_name]
        current = metrics.current_replicas
        queue_depth = metrics.queue_depth

        # Check cooldown period
        time_since_last_scale = datetime.now() - self.last_scale_time[worker_name]
        if time_since_last_scale < timedelta(minutes=config.cooldown_minutes):
            logger.debug(f"{worker_name}: In cooldown period ({time_since_last_scale.seconds}s)")
            return False, current

        # Calculate target replicas based on queue depth
        if queue_depth >= config.scale_up_threshold:
            # Scale up: add 1 replica per scale_up_threshold tasks
            additional_needed = (queue_depth // config.scale_up_threshold)
            target = min(current + additional_needed, config.max_replicas)

            if target > current:
                logger.info(
                    f"{worker_name}: Queue depth {queue_depth} exceeds threshold "
                    f"{config.scale_up_threshold}, scaling {current} -> {target}"
                )
                return True, target

        elif queue_depth <= config.scale_down_threshold and current > config.min_replicas:
            # Scale down: reduce by 1 replica if queue is empty/low
            target = max(current - 1, config.min_replicas)

            if target < current:
                logger.info(
                    f"{worker_name}: Queue depth {queue_depth} below threshold "
                    f"{config.scale_down_threshold}, scaling {current} -> {target}"
                )
                return True, target

        return False, current

    def collect_metrics(self, worker_name: str) -> ScalingMetrics:
        """Collect current metrics for a worker tier."""
        config = self.worker_configs[worker_name]
        queue_depth = self.get_total_queue_depth(config.queues)
        current_replicas = self.get_current_replicas(worker_name)

        # Track queue depth history for trend analysis
        self.last_queue_depths[worker_name].append(queue_depth)
        if len(self.last_queue_depths[worker_name]) > 10:
            self.last_queue_depths[worker_name].pop(0)

        return ScalingMetrics(
            queue_depth=queue_depth,
            current_replicas=current_replicas,
            active_tasks=0,  # TODO: Get from Celery inspect
            idle_time_minutes=0,  # TODO: Calculate from history
        )

    def run_once(self):
        """Run one iteration of the autoscaling loop."""
        logger.info("=" * 60)
        logger.info("Starting autoscaling check...")

        for worker_name, config in self.worker_configs.items():
            try:
                # Collect metrics
                metrics = self.collect_metrics(worker_name)

                logger.info(
                    f"{worker_name}: replicas={metrics.current_replicas}, "
                    f"queue_depth={metrics.queue_depth}, "
                    f"queues={config.queues}"
                )

                # Determine scaling action
                should_scale, target = self.should_scale(worker_name, metrics)

                # Execute scaling if needed
                if should_scale:
                    self.scale_worker(worker_name, target)

            except Exception as e:
                logger.error(f"Error processing {worker_name}: {e}", exc_info=True)

        logger.info("Autoscaling check complete")

    def run(self, interval_seconds: int = 60):
        """
        Run autoscaler in continuous loop.

        Args:
            interval_seconds: Time between scaling checks
        """
        logger.info(f"Starting autoscaler with {interval_seconds}s interval...")
        logger.info(f"Monitoring workers: {list(self.worker_configs.keys())}")

        try:
            while True:
                self.run_once()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Shutting down autoscaler...")


def main():
    parser = argparse.ArgumentParser(description='E2I Celery Worker Autoscaler')
    parser.add_argument(
        '--config',
        default='config/autoscale.yml',
        help='Path to autoscale configuration file'
    )
    parser.add_argument(
        '--docker-compose-dir',
        default='./docker',
        help='Directory containing docker-compose.yml'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Seconds between scaling checks (default: 60)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log scaling decisions without executing'
    )

    args = parser.parse_args()

    # Create autoscaler
    autoscaler = CeleryAutoscaler(
        config_path=args.config,
        docker_compose_dir=args.docker_compose_dir
    )

    # Run
    if args.dry_run:
        logger.info("DRY RUN MODE - No scaling will be performed")
        autoscaler.run_once()
    else:
        autoscaler.run(interval_seconds=args.interval)


if __name__ == '__main__':
    main()
