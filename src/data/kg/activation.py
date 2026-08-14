"""KG Layer-2 activation: bind a cohort's manifest to its committed KG cache.

Phase 2.9 Stage 2 shipped the KG *code* — the cache builder, the loader, the
voter's ``classify_kg_signal`` — but never the *data*. ``kg_cache_path`` and
``kg_mode`` both defaulted to ``None``, ``_resolve_kg_mode(None)`` returned
``"off"``, and no committed config set either field, so ``kg_edges`` was always
``()`` and every feature classified as ``no_signal`` (#1607).

This module is the missing binding. It maps a feature-manifest source to the
cache artifact built for it, and stamps ``kg_cache_path`` / ``kg_mode`` /
``target_entity_codes`` onto the cohort's ``scope_spec``.

Two design points worth keeping:

* **The cache is committed, not provisioned.** ``data/kg_cache/*.json`` is
  gitignored by default; the one artifact the pipeline depends on carries an
  explicit un-ignore. An artifact that exists only on the machine that built it
  repeats #600, where a gitignored tier0 cache silently skipped agent execution
  in CI — the failure looked like "no signal", not like "missing file".

* **Activation is fail-loud, never silent.** A configured-but-missing cache
  logs an error and leaves ``kg_mode`` off rather than proceeding as if the KG
  layer were simply quiet. "No signal" and "no cache" must not look alike; that
  ambiguity is the whole reason this gap survived so long.

Promotion to ``kg_mode="promoted"`` stays operator-driven per
``compute_promotion_eligibility`` and is deliberately out of scope here — the
point of ``shadow`` is to observe signal quality on a real cohort first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Repo-root-relative default. Callers may override for tests.
DEFAULT_KG_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "kg_cache"


@dataclass(frozen=True)
class KGActivation:
    """The KG cache binding for one feature-manifest source."""

    cache_filename: str
    #: ``(system, code)`` pairs identifying the cohort's prediction target. These
    #: MUST match what the cache was built with — the cache filename embeds a
    #: fingerprint of them, so a mismatch means the loader reads a file whose
    #: edges were computed against a different target.
    target_entity_codes: List[Tuple[str, str]] = field(default_factory=list)
    mode: str = "shadow"
    #: Free-text note surfaced in logs so an operator can tell WHY this cohort
    #: has KG on without reading the source.
    note: str = ""


# Optum / CSU. Built 2026-08-14 by::
#
#     python scripts/build_kg_cache.py --live \
#         --manifest-module src.data.manifests.optum_feature_manifest \
#         --features-attr OPTUM_FEATURES \
#         --target-entity-codes RXNORM:302379 \
#         --out data/kg_cache
#
# RXNORM:302379 is omalizumab, the standard-of-care biologic for chronic
# spontaneous urticaria. The build resolves it through RxNav (code -> name) and
# Open Targets (name -> ChEMBL id), then asks Open Targets which of the
# manifest's disease concepts that drug is APPROVED to treat.
#
# Measured on the committed artifact: 7 of 74 features classify as
# ``leak_drug_treats_disease`` — the urticaria diagnosis-code features
# (dx_l50_1/8/9_count, dx_total_csu, primary_diagnosis_code) and the asthma
# features (omalizumab is approved for asthma as well). The remaining 67 are
# ``no_signal``, which is the honest answer for labs and utilisation counts.
KG_ACTIVATIONS: Dict[str, KGActivation] = {
    "optum": KGActivation(
        cache_filename="1cdaa038__96bfd2e0.json",
        target_entity_codes=[("RXNORM", "302379")],
        mode="shadow",
        note="Optum/CSU vs omalizumab (RXNORM:302379); shadow observation window",
    ),
}


def apply_kg_activation(
    scope_spec: Dict[str, Any],
    manifest_source: Optional[str],
    *,
    cache_dir: Optional[Path] = None,
) -> bool:
    """Stamp ``kg_cache_path`` / ``kg_mode`` / ``target_entity_codes`` in place.

    Returns True when KG was activated. Idempotent. An explicit ``kg_mode`` an
    operator set is never overwritten — but only ``"off"`` short-circuits the
    binding.

    That distinction is load-bearing. An explicit ``"shadow"`` or ``"promoted"``
    is an instruction to turn the KG layer **on**, and it is the most natural
    way for an operator to ask for exactly what this module provides. Treating
    it as "hands off" left ``kg_cache_path`` unset, so ``_load_kg_cache``
    returned nothing and every feature came back ``no_signal`` — with no ERROR
    logged, because the missing-cache check was never reached. That is the
    "no cache is indistinguishable from no signal" failure #1607 exists to kill,
    reintroduced by the guard meant to respect operators. Under ``"promoted"``
    it is worse still: the operator believes the KG can now drop features.

    A missing cache file is an ERROR log and a no-op, not a silent pass.
    """
    if not manifest_source:
        return False
    activation = KG_ACTIVATIONS.get(manifest_source)
    if activation is None:
        return False

    explicit_mode = scope_spec.get("kg_mode")
    if explicit_mode is not None and str(explicit_mode).strip().lower() == "off":
        logger.info(
            "KG activation: scope_spec sets kg_mode=%r for %s; staying off",
            explicit_mode,
            manifest_source,
        )
        return False

    directory = cache_dir if cache_dir is not None else DEFAULT_KG_CACHE_DIR
    cache_path = Path(directory) / activation.cache_filename
    if not cache_path.is_file():
        logger.error(
            "KG activation: cache %s is MISSING for manifest source %r — KG Layer 2 "
            "stays off. Rebuild it with scripts/build_kg_cache.py --live "
            "(see docs/runbooks/kg_cache.md). Do NOT treat the resulting "
            "kg_signal=no_signal as evidence the KG has nothing to say.",
            cache_path,
            manifest_source,
        )
        return False

    scope_spec["kg_cache_path"] = str(cache_path)
    # Bind the data, but let an operator's explicit on-mode stand. Promotion is
    # an operator decision (``compute_promotion_eligibility``); activation
    # supplies the cache it needs, it does not overrule it.
    if explicit_mode is None:
        scope_spec["kg_mode"] = activation.mode
    # Only set target codes when the cohort has not already declared them; the
    # cache fingerprint is derived from these, so a caller-supplied value that
    # disagrees would silently read a cache built for a different target.
    existing_targets = scope_spec.get("target_entity_codes")
    if not existing_targets:
        scope_spec["target_entity_codes"] = [
            (system, code) for system, code in activation.target_entity_codes
        ]
    elif [tuple(t) for t in existing_targets] != list(activation.target_entity_codes):
        logger.warning(
            "KG activation: scope_spec target_entity_codes %r differ from the codes "
            "the cache %s was built with (%r). The cache filename embeds a "
            "fingerprint of the target codes, so the loaded edges may not relate to "
            "this cohort's target.",
            existing_targets,
            activation.cache_filename,
            activation.target_entity_codes,
        )

    logger.info(
        "KG activation: %s -> mode=%s cache=%s (%s)",
        manifest_source,
        scope_spec["kg_mode"],
        cache_path.name,
        activation.note,
    )
    return True
