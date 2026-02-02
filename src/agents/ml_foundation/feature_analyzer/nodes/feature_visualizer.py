"""Feature Importance Visualization Node - NO LLM.

Generates visualizations for feature analysis results:
- Feature importance bar charts
- Selection process summary
- Correlation heatmaps
- SHAP summary plots

This is a deterministic computation node with no LLM calls.
"""

import io
import logging
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualization features disabled.")


async def generate_visualizations(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate visualizations for feature analysis results.

    Creates:
    1. Feature importance bar chart
    2. Selection funnel chart
    3. Correlation heatmap (if correlation data available)
    4. SHAP summary plot (if SHAP values available)

    Args:
        state: Feature analyzer state with analysis results

    Returns:
        State updates with visualization paths/data
    """
    if not MATPLOTLIB_AVAILABLE:
        return {
            "visualizations_generated": False,
            "visualization_error": "matplotlib not available",
        }

    visualizations: Dict[str, Any] = {}
    output_dir = state.get("visualization_output_dir")

    try:
        # 1. Feature Importance Chart
        feature_importance = state.get("feature_importance", {})
        if feature_importance:
            viz_data = _create_importance_chart(
                feature_importance,
                title="Feature Importance",
                output_path=_get_output_path(output_dir, "feature_importance.png"),
            )
            visualizations["importance_chart"] = viz_data

        # 2. Selection Funnel Chart
        selection_history = state.get("selection_history", [])
        if selection_history:
            viz_data = _create_selection_funnel(
                selection_history,
                original_count=state.get("original_feature_count", 0),
                output_path=_get_output_path(output_dir, "selection_funnel.png"),
            )
            visualizations["selection_funnel"] = viz_data

        # 3. Feature Statistics Summary
        feature_statistics = state.get("feature_statistics", {})
        if feature_statistics:
            viz_data = _create_statistics_table(
                feature_statistics,
                output_path=_get_output_path(output_dir, "feature_statistics.png"),
            )
            visualizations["statistics_table"] = viz_data

        # 4. SHAP Summary (if available)
        shap_values = state.get("shap_values")
        feature_names = state.get("feature_names", [])
        if shap_values is not None and feature_names:
            viz_data = _create_shap_summary(
                shap_values,
                feature_names,
                output_path=_get_output_path(output_dir, "shap_summary.png"),
            )
            visualizations["shap_summary"] = viz_data

        logger.info(f"Generated {len(visualizations)} visualizations")

        return {
            "visualizations": visualizations,
            "visualizations_generated": True,
            "visualization_count": len(visualizations),
        }

    except Exception as e:
        logger.exception("Visualization generation failed")
        return {
            "visualizations_generated": False,
            "visualization_error": str(e),
        }


def _get_output_path(output_dir: Optional[str], filename: str) -> Optional[Path]:
    """Get output path for visualization file."""
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path / filename
    return None


def _fig_to_base64(fig: "plt.Figure") -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return b64encode(buf.getvalue()).decode("utf-8")


def _create_importance_chart(
    feature_importance: Dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create horizontal bar chart for feature importance.

    Args:
        feature_importance: Dict mapping feature names to importance scores
        title: Chart title
        top_n: Number of top features to show
        output_path: Optional path to save the figure

    Returns:
        Dict with chart data and optional base64 image
    """
    # Sort by importance and take top N
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    if not sorted_features:
        return {"error": "No features to visualize"}

    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color="#1f77b4", edgecolor="white")

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Importance Score")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add value labels
    for bar, importance in zip(bars, importances, strict=False):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{importance:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    result: Dict[str, Any] = {
        "type": "importance_bar_chart",
        "features": features,
        "importances": importances,
    }

    # Save or encode
    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        result["path"] = str(output_path)
    else:
        result["image_base64"] = _fig_to_base64(fig)

    plt.close(fig)
    return result


def _create_selection_funnel(
    selection_history: List[Dict[str, Any]],
    original_count: int,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create funnel chart showing feature selection process.

    Args:
        selection_history: List of selection steps with before/after counts
        original_count: Original feature count
        output_path: Optional path to save the figure

    Returns:
        Dict with chart data
    """
    if not selection_history:
        return {"error": "No selection history"}

    # Build funnel data
    steps = ["Original"]
    counts = [original_count]

    for step in selection_history:
        step_name = step.get("step", "Unknown")
        after_count = step.get("features_after", counts[-1])
        steps.append(step_name.replace("_", " ").title())
        counts.append(after_count)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    x_pos = np.arange(len(steps))
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(steps)))

    bars = ax.bar(x_pos, counts, color=colors, edgecolor="white", linewidth=1.5)

    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(steps, rotation=45, ha="right")
    ax.set_ylabel("Number of Features")
    ax.set_title("Feature Selection Funnel", fontsize=14, fontweight="bold")

    # Add value labels and reduction percentages
    for i, (bar, count) in enumerate(zip(bars, counts, strict=False)):
        # Count label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        # Reduction percentage (except for first bar)
        if i > 0 and counts[i - 1] > 0:
            reduction = (1 - count / counts[i - 1]) * 100
            if reduction > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"-{reduction:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )

    plt.tight_layout()

    result: Dict[str, Any] = {
        "type": "selection_funnel",
        "steps": steps,
        "counts": counts,
        "final_reduction": (
            f"{(1 - counts[-1] / counts[0]) * 100:.1f}%" if counts[0] > 0 else "N/A"
        ),
    }

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        result["path"] = str(output_path)
    else:
        result["image_base64"] = _fig_to_base64(fig)

    plt.close(fig)
    return result


def _create_statistics_table(
    feature_statistics: Dict[str, Dict[str, Any]],
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a table visualization of feature statistics.

    Args:
        feature_statistics: Dict mapping feature names to statistics
        top_n: Number of features to show
        output_path: Optional path to save

    Returns:
        Dict with table data
    """
    if not feature_statistics:
        return {"error": "No statistics available"}

    # Build DataFrame
    data = []
    for feature, stats in list(feature_statistics.items())[:top_n]:
        row = {
            "Feature": feature[:30] + "..." if len(feature) > 30 else feature,
            "Mean": f"{stats.get('mean', 'N/A'):.3f}" if stats.get("mean") else "N/A",
            "Std": f"{stats.get('std', 'N/A'):.3f}" if stats.get("std") else "N/A",
            "Min": f"{stats.get('min', 'N/A'):.3f}" if stats.get("min") else "N/A",
            "Max": f"{stats.get('max', 'N/A'):.3f}" if stats.get("max") else "N/A",
            "Null%": f"{stats.get('null_pct', 0) * 100:.1f}%",
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, max(4, len(data) * 0.35)))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#1f77b4"] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Feature Statistics Summary",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    result: Dict[str, Any] = {
        "type": "statistics_table",
        "features_shown": len(data),
        "total_features": len(feature_statistics),
    }

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        result["path"] = str(output_path)
    else:
        result["image_base64"] = _fig_to_base64(fig)

    plt.close(fig)
    return result


def _create_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 15,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create SHAP summary bar chart.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: Feature names
        top_n: Number of top features to show
        output_path: Optional path to save

    Returns:
        Dict with chart data
    """
    if shap_values is None or len(feature_names) == 0:
        return {"error": "No SHAP values available"}

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort and take top N
    indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = mean_abs_shap[indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.4)))

    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_values, color="#ff7f0e", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")

    # Add value labels
    for bar, value in zip(bars, top_values, strict=False):
        ax.text(
            bar.get_width() + max(top_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    result: Dict[str, Any] = {
        "type": "shap_summary",
        "features": top_features,
        "mean_shap_values": top_values.tolist(),
    }

    if output_path:
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        result["path"] = str(output_path)
    else:
        result["image_base64"] = _fig_to_base64(fig)

    plt.close(fig)
    return result
