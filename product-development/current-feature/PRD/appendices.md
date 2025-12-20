# Appendices

## Appendix A: Analyzed Brands

**Remibrutinib (CSU)**
- **Indication**: Chronic Spontaneous Urticaria
- **Mechanism**: BTK inhibitor
- **Market**: Dermatology/Immunology
- **Key Metrics**: Dermatologist prescribing patterns, patient persistence

**Fabhalta (PNH)**
- **Indication**: Paroxysmal Nocturnal Hemoglobinuria
- **Mechanism**: Factor B inhibitor
- **Market**: Rare hematology
- **Key Metrics**: Hematologist adoption, switch from Soliris/Ultomiris

**Kisqali (HR+/HER2- Breast Cancer)**
- **Indication**: HR+/HER2- Metastatic Breast Cancer
- **Mechanism**: CDK4/6 inhibitor (ribociclib)
- **Market**: Oncology (highly competitive)
- **Key Metrics**: Oncologist prescribing, payer coverage, patient duration

---

## Appendix B: Database Schema Summary

**37 Total Tables** across 5 categories:

**Core Data (8 tables)**
- `patient_journeys`: Patient treatment timelines
- `hcp_profiles`: Healthcare provider attributes
- `treatment_events`: Prescription and treatment records
- `market_dynamics`: Competitive landscape data
- `geography_hierarchy`: Territory/region structure
- `payer_coverage`: Insurance coverage data
- `digital_interactions`: HCP digital engagement
- `samples_data`: Sample drop tracking

**ML Foundation (17 tables)**
- `ml_experiments`: Experiment metadata
- `ml_model_registry`: Model versions and metadata
- `ml_deployments`: Production model tracking
- `ml_feature_store`: Feature definitions
- `ml_data_quality`: Data validation results
- `digital_twin_models`: Twin model metadata (v4.2)
- `twin_simulations`: Simulation results (v4.2)
- `twin_fidelity_tracking`: Twin validation metrics (v4.2)
- `tool_registry`: Available tools (v4.2)
- `tool_dependencies`: Tool relationships (v4.2)
- `composition_episodes`: Tool composition logs (v4.2)
- `shap_explanations`: SHAP audit trail (v4.1)
- Plus 5 additional ML tables

**Memory (7 tables)**
- `episodic_memories`: User interactions
- `procedural_memories`: Tool sequences
- `semantic_cache`: Graph embeddings cache
- `working_memory`: Session state (Redis)
- `composition_episodes`: Tool composition memory (v4.2)
- `classification_logs`: Query classification history (v4.2)
- `composition_metrics`: Composition performance (v4.2)

**Causal Validation (2 tables)**
- `causal_validations`: Refutation test results
- `expert_reviews`: Human validation for edge cases

**Supporting (3 tables)**
- `user_sessions`: Authentication and session management
- `data_source_tracking`: Data lineage
- `audit_logs`: Full audit trail

---

## Appendix C: Glossary

**ATE**: Average Treatment Effect - the average causal effect across all units

**CATE**: Conditional Average Treatment Effect - causal effect for specific subgroups

**DoWhy**: Python library for causal inference with multiple identification methods

**EconML**: Econometric ML library for heterogeneous treatment effects

**HCP**: Healthcare Provider (physician, nurse practitioner, etc.)

**KPI**: Key Performance Indicator

**NRx**: New Prescriptions (first fill for a patient)

**Refutation Tests**: Statistical tests to validate causal estimates

**SHAP**: SHapley Additive exPlanations - model interpretability method

**TRx**: Total Prescriptions (includes new + refills)

**Twin Fidelity**: Accuracy of digital twin predictions vs real-world outcomes

---

## Appendix D: Regulatory & Compliance Notes

**Scope Limitations**:
- ❌ NOT clinical decision support
- ❌ NOT medical literature search
- ❌ NOT drug safety monitoring
- ❌ NOT patient treatment recommendations

**Compliance Alignment**:
- ✅ GDPR: De-identified data, right to erasure, data portability
- ✅ HIPAA-aligned: Commercial data only (not PHI)
- ✅ 21 CFR Part 11: Audit trails, e-signature ready
- ✅ FDA Promotional Guidelines: No off-label promotion in data sources

**Audit Requirements**:
- Full query audit trail (who, what, when)
- Model prediction audit trail with explanations
- Data lineage tracking
- Version control for all models and configurations

---

## Appendix E: Technology Stack

**AI/ML**:
- LangGraph 0.2+, LangChain 0.3+, Claude (Anthropic)
- DoWhy 0.11+, EconML 0.15+, NetworkX 3.0+
- scikit-learn 1.5+, pandas 2.2+, numpy 1.26+

**MLOps**:
- MLflow 2.16+, Opik 0.2+, Optuna 3.6+
- SHAP 0.46+, BentoML 1.3+, Feast 0.40+
- Great Expectations 1.0+

**NLP/RAG**:
- fastText 0.9.3+, rapidfuzz 3.10+
- sentence-transformers 3.0+, tiktoken 0.8+

**Database & Storage**:
- Supabase 2.0+ (PostgreSQL + pgvector)
- Redis 5.0+, FalkorDB 1.0+

**API & Backend**:
- FastAPI 0.115+, Uvicorn 0.30+
- Pydantic 2.9+, python-dotenv 1.0+

**Development**:
- Python 3.12+, pytest 8.3+, black 24.8+
- Docker, Docker Compose

**Frontend (Planned)**:
- React 18+, TypeScript 5+, Redux Toolkit

---
