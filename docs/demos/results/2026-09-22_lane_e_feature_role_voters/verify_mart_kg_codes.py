"""Cheapest disproof of the authored mart comorbidity -> concept mapping.

Reads every candidate (system, code) through the SAME resolution path the
cache builder uses (``UMLSClient.cui_lookup`` for UMLS CUIs,
``EntityLinker.resolve`` for ICD10CM codes) and prints the preferred name the
vocabulary actually returns, so a wrong-but-valid code is visible BEFORE it
enters the manifest. Public read calls only (UMLS UTS key from .env).
"""

from __future__ import annotations

import datetime
import sys

import src

assert ".worktrees/lane-e-feature-role-voters" in src.__file__, src.__file__

from src.data.kg.entity_linker import EntityLinker  # noqa: E402

CANDIDATES: dict[str, tuple[tuple[str, str], ...]] = {
    "cci_mi": (("ICD10CM", "I21.9"), ("UMLS", "C0027051")),
    "cci_chf": (("ICD10CM", "I50.9"), ("UMLS", "C0018802")),
    "cci_pvd": (("ICD10CM", "I73.9"), ("UMLS", "C0085096")),
    "cci_cerebrovascular": (("ICD10CM", "I67.9"), ("UMLS", "C0007820")),
    "cci_dementia": (("ICD10CM", "F03.90"), ("UMLS", "C0497327")),
    "cci_chronic_pulmonary": (
        ("ICD10CM", "J44.9"),
        ("UMLS", "C0024117"),
        ("ICD10CM", "J45.909"),
        ("UMLS", "C0004096"),
    ),
    "cci_rheumatic": (("ICD10CM", "M06.9"), ("UMLS", "C0003873")),
    "cci_peptic_ulcer": (("ICD10CM", "K27.9"), ("UMLS", "C0030920")),
    "cci_mild_liver": (("ICD10CM", "K76.9"), ("UMLS", "C0023895")),
    "cci_diabetes_no_complication": (("ICD10CM", "E11.9"), ("UMLS", "C0011849")),
    "cci_diabetes_complication": (("ICD10CM", "E11.8"), ("UMLS", "C0011849")),
    "cci_paraplegia": (("ICD10CM", "G82.20"), ("UMLS", "C0030486")),
    "cci_renal": (("ICD10CM", "N18.9"), ("UMLS", "C0022661")),
    "cci_malignancy": (("ICD10CM", "C80.1"), ("UMLS", "C0006826")),
    "cci_severe_liver": (("ICD10CM", "K74.60"), ("UMLS", "C0023890")),
    "cci_metastatic_cancer": (("ICD10CM", "C79.9"), ("UMLS", "C0027627")),
    "cci_hiv": (("ICD10CM", "B20"), ("UMLS", "C0019693")),
    "elx_chf": (("ICD10CM", "I50.9"), ("UMLS", "C0018802")),
    "elx_cardiac_arrhythmia": (("ICD10CM", "I49.9"), ("UMLS", "C0003811")),
    "elx_valvular_disease": (("ICD10CM", "I38"), ("UMLS", "C0018824")),
    "elx_pulmonary_circulation": (("ICD10CM", "I27.20"), ("UMLS", "C0020542")),
    "elx_pvd": (("ICD10CM", "I73.9"), ("UMLS", "C0085096")),
    "elx_hypertension_uncomplicated": (("ICD10CM", "I10"), ("UMLS", "C0020538")),
    "elx_hypertension_complicated": (("ICD10CM", "I11.9"), ("UMLS", "C0020538")),
    "elx_paralysis": (("ICD10CM", "G83.9"), ("UMLS", "C0522224")),
    "elx_other_neurological": (("ICD10CM", "G96.9"), ("UMLS", "C0027765")),
    "elx_chronic_pulmonary": (
        ("ICD10CM", "J44.9"),
        ("UMLS", "C0024117"),
        ("ICD10CM", "J45.909"),
        ("UMLS", "C0004096"),
    ),
    "elx_diabetes_uncomplicated": (("ICD10CM", "E11.9"), ("UMLS", "C0011849")),
    "elx_diabetes_complicated": (("ICD10CM", "E11.8"), ("UMLS", "C0011849")),
    "elx_hypothyroidism": (("ICD10CM", "E03.9"), ("UMLS", "C0020676")),
    "elx_renal_failure": (("ICD10CM", "N18.9"), ("UMLS", "C0022661")),
    "elx_liver_disease": (("ICD10CM", "K76.9"), ("UMLS", "C0023895")),
    "elx_peptic_ulcer": (("ICD10CM", "K27.9"), ("UMLS", "C0030920")),
    "elx_aids_hiv": (("ICD10CM", "B20"), ("UMLS", "C0019693")),
    "elx_lymphoma": (("ICD10CM", "C85.90"), ("UMLS", "C0024299")),
    "elx_metastatic_cancer": (("ICD10CM", "C79.9"), ("UMLS", "C0027627")),
    "elx_solid_tumor_no_metastasis": (("ICD10CM", "C80.1"), ("UMLS", "C0006826")),
    "elx_rheumatoid_collagen": (("ICD10CM", "M06.9"), ("UMLS", "C0003873")),
    "elx_coagulopathy": (("ICD10CM", "D68.9"), ("UMLS", "C0005779")),
    "elx_obesity": (("ICD10CM", "E66.9"), ("UMLS", "C0028754")),
    "elx_weight_loss": (("ICD10CM", "R63.4"), ("UMLS", "C1262477")),
    "elx_fluid_electrolyte": (("ICD10CM", "E87.8"), ("UMLS", "C0042990")),
    "elx_blood_loss_anemia": (("ICD10CM", "D50.0"), ("UMLS", "C0002871")),
    "elx_deficiency_anemia": (("ICD10CM", "D50.9"), ("UMLS", "C0162316")),
    "elx_alcohol_abuse": (("ICD10CM", "F10.10"), ("UMLS", "C0001973")),
    "elx_drug_abuse": (("ICD10CM", "F19.10"), ("UMLS", "C0038586")),
    "elx_psychoses": (("ICD10CM", "F29"), ("UMLS", "C0033975")),
    "elx_depression": (("ICD10CM", "F33.9"), ("UMLS", "C0011581")),
}


def main() -> int:
    print("captured_at_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
    seen: dict[tuple[str, str], str] = {}
    bad = 0
    with EntityLinker() as linker:
        for feature, codes in CANDIDATES.items():
            for system, code in codes:
                key = (system, code)
                if key in seen:
                    print(f"{feature:34s} {system}:{code:8s} -> (dup) {seen[key]}")
                    continue
                try:
                    if system == "UMLS":
                        c = linker.umls.cui_lookup(code)
                        line = f"cui={c.cui} name={c.preferred_name!r} semtypes={c.semantic_types}"
                    else:
                        link = linker.resolve(code, system)  # type: ignore[arg-type]
                        if link.concept is None:
                            line = f"UNRESOLVED error={link.error!r}"
                            bad += 1
                        else:
                            line = (
                                f"cui={link.concept.cui} name={link.concept.preferred_name!r} "
                                f"confidence={link.confidence}"
                            )
                except Exception as exc:  # noqa: BLE001
                    line = f"RAISED {type(exc).__name__}: {exc}"
                    bad += 1
                seen[key] = line
                print(f"{feature:34s} {system}:{code:8s} -> {line}")
    print("unresolved_or_raised", bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
