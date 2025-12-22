#!/usr/bin/env python3
"""Model Deployment Script for E2I Causal Analytics.

This script provides a CLI interface for deploying ML models using BentoML.

Usage:
    python scripts/deploy_model.py --help
    python scripts/deploy_model.py register --model-path model.pkl --name churn_model
    python scripts/deploy_model.py build --model-tag churn_model:latest --service-type classification
    python scripts/deploy_model.py containerize --bento-tag churn_service:latest
    python scripts/deploy_model.py deploy --image e2i-churn-service:latest

Version: 1.0.0
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def register_model(
    model_path: str,
    name: str,
    framework: str = "sklearn",
    metadata_path: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> str:
    """Register a model with BentoML.

    Args:
        model_path: Path to the model file
        name: Name for the model
        framework: ML framework (sklearn, xgboost, lightgbm, econml)
        metadata_path: Optional path to metadata JSON
        labels: Optional labels dict

    Returns:
        Model tag string
    """
    from src.mlops.bentoml_packaging import register_model_auto

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load metadata if provided
    metadata = {}
    if metadata_path:
        with open(metadata_path) as f:
            metadata = json.load(f)

    metadata["source_path"] = model_path
    metadata["deployed_at"] = datetime.utcnow().isoformat()

    # Register model
    tag = register_model_auto(
        model=model,
        model_name=name,
        framework=framework,
        metadata=metadata,
        labels=labels,
    )

    logger.info(f"Model registered: {tag}")
    return tag


def build_bento(
    model_tag: str,
    service_type: str,
    service_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    **service_kwargs,
) -> str:
    """Build a Bento from a registered model.

    Args:
        model_tag: BentoML model tag
        service_type: Type of service (classification, regression, causal)
        service_name: Optional service name override
        output_dir: Optional output directory
        **service_kwargs: Additional service configuration

    Returns:
        Bento tag string
    """
    import tempfile

    from src.mlops.bentoml_packaging import (
        BentoConfig,
        build_bento,
        create_bentofile,
        generate_service_file,
    )

    # Use temp directory if not specified
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="bento_build_")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate service name from model if not provided
    if service_name is None:
        model_name = model_tag.split(":")[0]
        service_name = f"{model_name}_service"

    # Generate service file
    service_path = output_dir / "service.py"
    generate_service_file(
        model_tag=model_tag,
        service_type=service_type,
        output_path=service_path,
        service_name=service_name,
        **service_kwargs,
    )

    # Determine Python packages based on service type
    base_packages = [
        "bentoml>=1.4.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
    ]

    if service_type == "classification" or service_type == "regression":
        packages = base_packages + ["scikit-learn>=1.3.0"]
    elif service_type == "causal":
        packages = base_packages + ["econml>=0.14.0", "dowhy>=0.11.0"]
    else:
        packages = base_packages

    # Create bentofile
    config = BentoConfig(
        service_name=service_name,
        service_file="service.py",
        description=f"E2I {service_type.title()} Service for {model_tag}",
        labels={
            "framework": service_type,
            "project": "e2i-causal-analytics",
        },
        python_packages=packages,
    )

    create_bentofile(output_dir, config)

    # Build bento
    bento_tag = build_bento(output_dir)

    logger.info(f"Bento built: {bento_tag}")
    return bento_tag


def containerize(
    bento_tag: str,
    image_name: Optional[str] = None,
    image_tag: str = "latest",
    registry: Optional[str] = None,
    push: bool = False,
) -> str:
    """Containerize a Bento into a Docker image.

    Args:
        bento_tag: Tag of the Bento to containerize
        image_name: Optional image name override
        image_tag: Image tag (default: latest)
        registry: Optional registry to push to
        push: Whether to push to registry

    Returns:
        Full image name
    """
    from src.mlops.bentoml_packaging import ContainerConfig, containerize_bento

    if image_name is None:
        bento_name = bento_tag.split(":")[0]
        image_name = f"e2i-{bento_name}"

    config = ContainerConfig(
        image_name=image_name,
        image_tag=image_tag,
        registry=registry,
    )

    full_image = containerize_bento(bento_tag, config, push=push)

    logger.info(f"Container created: {full_image}")
    return full_image


def deploy_local(
    image_name: str,
    port: int = 3000,
    detach: bool = True,
    container_name: Optional[str] = None,
) -> str:
    """Deploy a containerized model locally.

    Args:
        image_name: Docker image name
        port: Port to expose
        detach: Run in background
        container_name: Optional container name

    Returns:
        Container ID
    """
    if container_name is None:
        container_name = f"e2i-model-{port}"

    cmd = [
        "docker", "run",
        "-p", f"{port}:3000",
        "--name", container_name,
    ]

    if detach:
        cmd.append("-d")

    cmd.append(image_name)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Deployment failed: {result.stderr}")
        raise RuntimeError(f"Deployment failed: {result.stderr}")

    container_id = result.stdout.strip()
    logger.info(f"Deployed container: {container_id}")
    logger.info(f"Service available at: http://localhost:{port}")

    return container_id


def check_health(url: str) -> Dict[str, Any]:
    """Check health of a deployed service.

    Args:
        url: Base URL of the service

    Returns:
        Health check response
    """
    import requests

    health_url = f"{url.rstrip('/')}/health"

    try:
        response = requests.get(health_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def list_models() -> None:
    """List all registered models."""
    try:
        import bentoml
        models = bentoml.models.list()
        if not models:
            print("No models registered.")
            return

        print("\nRegistered Models:")
        print("-" * 60)
        for model in models:
            print(f"  {model.tag}")
            print(f"    Created: {model.creation_time}")
            if model.info.metadata:
                print(f"    Framework: {model.info.metadata.get('framework', 'unknown')}")
        print("-" * 60)
    except ImportError:
        print("BentoML not installed")
    except Exception as e:
        print(f"Error listing models: {e}")


def list_bentos() -> None:
    """List all built Bentos."""
    try:
        import bentoml
        bentos = bentoml.list()
        if not bentos:
            print("No Bentos built.")
            return

        print("\nBuilt Bentos:")
        print("-" * 60)
        for bento in bentos:
            print(f"  {bento.tag}")
            print(f"    Created: {bento.creation_time}")
        print("-" * 60)
    except ImportError:
        print("BentoML not installed")
    except Exception as e:
        print(f"Error listing Bentos: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="E2I Model Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    register_parser = subparsers.add_parser(
        "register",
        help="Register a model with BentoML",
    )
    register_parser.add_argument(
        "--model-path",
        required=True,
        help="Path to model file (pickle)",
    )
    register_parser.add_argument(
        "--name",
        required=True,
        help="Name for the model",
    )
    register_parser.add_argument(
        "--framework",
        default="sklearn",
        choices=["sklearn", "xgboost", "lightgbm", "econml"],
        help="ML framework",
    )
    register_parser.add_argument(
        "--metadata",
        help="Path to metadata JSON file",
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build a Bento from a registered model",
    )
    build_parser.add_argument(
        "--model-tag",
        required=True,
        help="BentoML model tag",
    )
    build_parser.add_argument(
        "--service-type",
        required=True,
        choices=["classification", "regression", "causal"],
        help="Type of service to create",
    )
    build_parser.add_argument(
        "--service-name",
        help="Optional service name override",
    )
    build_parser.add_argument(
        "--output-dir",
        help="Output directory for build files",
    )

    # Containerize command
    container_parser = subparsers.add_parser(
        "containerize",
        help="Containerize a Bento",
    )
    container_parser.add_argument(
        "--bento-tag",
        required=True,
        help="Tag of the Bento to containerize",
    )
    container_parser.add_argument(
        "--image-name",
        help="Docker image name",
    )
    container_parser.add_argument(
        "--image-tag",
        default="latest",
        help="Docker image tag",
    )
    container_parser.add_argument(
        "--registry",
        help="Docker registry to push to",
    )
    container_parser.add_argument(
        "--push",
        action="store_true",
        help="Push to registry after building",
    )

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy a containerized model locally",
    )
    deploy_parser.add_argument(
        "--image",
        required=True,
        help="Docker image to deploy",
    )
    deploy_parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to expose",
    )
    deploy_parser.add_argument(
        "--name",
        help="Container name",
    )

    # Health check command
    health_parser = subparsers.add_parser(
        "health",
        help="Check health of a deployed service",
    )
    health_parser.add_argument(
        "--url",
        required=True,
        help="Base URL of the service",
    )

    # List commands
    subparsers.add_parser("list-models", help="List registered models")
    subparsers.add_parser("list-bentos", help="List built Bentos")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "register":
            tag = register_model(
                model_path=args.model_path,
                name=args.name,
                framework=args.framework,
                metadata_path=args.metadata,
            )
            print(f"Registered: {tag}")

        elif args.command == "build":
            tag = build_bento(
                model_tag=args.model_tag,
                service_type=args.service_type,
                service_name=args.service_name,
                output_dir=args.output_dir,
            )
            print(f"Built: {tag}")

        elif args.command == "containerize":
            image = containerize(
                bento_tag=args.bento_tag,
                image_name=args.image_name,
                image_tag=args.image_tag,
                registry=args.registry,
                push=args.push,
            )
            print(f"Created: {image}")

        elif args.command == "deploy":
            container_id = deploy_local(
                image_name=args.image,
                port=args.port,
                container_name=args.name,
            )
            print(f"Deployed: {container_id}")

        elif args.command == "health":
            result = check_health(args.url)
            print(json.dumps(result, indent=2))

        elif args.command == "list-models":
            list_models()

        elif args.command == "list-bentos":
            list_bentos()

    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
