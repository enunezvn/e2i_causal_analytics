"""
E2I Real-Time SHAP Chat Integration
====================================
Integrates the /explain API with the existing NLV chat interface.

Components:
1. EXPLAIN intent classification
2. Entity extraction for explanation queries
3. Orchestrator tool for calling /explain API
4. Response formatter with SHAP visualization
5. Explainer agent narrative integration

Integration Points:
- nlp/intent_classifier.py → Add EXPLAIN intent
- nlp/entity_extractor.py → Extract patient_id, model_type
- agents/orchestrator/router.py → Route EXPLAIN to this handler
- visualization/shap_viz.py → Render charts in chat response

Author: E2I Causal Analytics Team
Version: 4.1.0
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import re
import httpx
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. INTENT CLASSIFICATION - Add to nlp/models/intent_models.py
# =============================================================================

class IntentType(str, Enum):
    """
    Query intent types for agent routing.
    
    V4.1 Addition: EXPLAIN intent for real-time SHAP explanations
    """
    # Existing intents
    CAUSAL = "causal"                    # "Why did X cause Y?"
    EXPLORATORY = "exploratory"          # "Show me trends in..."
    COMPARATIVE = "comparative"          # "Compare X vs Y"
    TREND = "trend"                      # "How has X changed?"
    WHAT_IF = "what_if"                  # "What if we increased...?"
    
    # V4 intents
    SCOPE = "scope"                      # "Define the problem..."
    QUALITY = "quality"                  # "Check data quality..."
    TRAINING = "training"                # "Train a model..."
    DEPLOY = "deploy"                    # "Deploy the model..."
    
    # V4.1 NEW: Explanation intent
    EXPLAIN = "explain"                  # "Why is this patient flagged?"


class ExplainSubIntent(str, Enum):
    """Sub-intents for EXPLAIN queries."""
    PATIENT_PREDICTION = "patient_prediction"      # "Why was patient X flagged?"
    MODEL_DECISION = "model_decision"              # "What drove this recommendation?"
    FEATURE_IMPORTANCE = "feature_importance"      # "Which features matter most?"
    PREDICTION_HISTORY = "prediction_history"      # "Show explanation history for..."
    COMPARE_EXPLANATIONS = "compare_explanations"  # "Compare why these patients differ"


# =============================================================================
# 2. INTENT PATTERNS - Add to intent_classifier.py
# =============================================================================

EXPLAIN_INTENT_PATTERNS = {
    # Direct explanation requests
    "why_patient": [
        r"why (?:is|was) (?:patient\s+)?(\S+) (?:flagged|recommended|scored|predicted)",
        r"explain (?:the )?(?:prediction|recommendation|score) for (?:patient\s+)?(\S+)",
        r"what (?:drove|caused|factors|contributed) (?:to )?(?:patient\s+)?(\S+)",
    ],
    
    # Feature importance queries
    "feature_importance": [
        r"(?:what|which) features? (?:are|is) (?:most )?important",
        r"(?:what|which) (?:drove|factors|variables) (?:drove|affect|impact)",
        r"why (?:did|does) the model (?:predict|recommend|flag)",
        r"explain (?:the )?model(?:'s)? (?:decision|prediction|output)",
    ],
    
    # SHAP-specific queries
    "shap_explicit": [
        r"(?:show|get|compute|display) shap",
        r"shap (?:values?|explanation|analysis)",
        r"(?:feature )?(?:contribution|attribution)s?",
        r"waterfall (?:chart|plot|diagram)",
    ],
    
    # History queries
    "explanation_history": [
        r"(?:show|get) (?:explanation|prediction) history",
        r"past (?:explanations?|predictions?) for",
        r"how (?:has|have) (?:the )?predictions? (?:changed|evolved)",
    ],
    
    # Comparison queries
    "compare_explanations": [
        r"compare (?:explanations?|predictions?) (?:for|between)",
        r"why (?:is|are) (?:patient\s+)?(\S+) different (?:from|than)",
        r"what(?:'s| is) the difference between",
    ]
}


def classify_explain_intent(query: str) -> Tuple[bool, Optional[ExplainSubIntent], float]:
    """
    Determine if query is an EXPLAIN intent and classify sub-intent.
    
    Returns:
        (is_explain_intent, sub_intent, confidence)
    """
    query_lower = query.lower().strip()
    
    # Check each pattern category
    for sub_intent_key, patterns in EXPLAIN_INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                sub_intent = ExplainSubIntent(sub_intent_key.replace("why_patient", "patient_prediction"))
                return (True, sub_intent, 0.85)
    
    # Fallback: keyword-based detection
    explain_keywords = [
        "why", "explain", "drove", "factors", "features", 
        "shap", "interpret", "reasoning", "decision"
    ]
    
    patient_context_keywords = [
        "patient", "flagged", "recommended", "predicted", 
        "scored", "propensity", "risk"
    ]
    
    has_explain_keyword = any(kw in query_lower for kw in explain_keywords)
    has_patient_context = any(kw in query_lower for kw in patient_context_keywords)
    
    if has_explain_keyword and has_patient_context:
        return (True, ExplainSubIntent.PATIENT_PREDICTION, 0.7)
    elif has_explain_keyword:
        return (True, ExplainSubIntent.MODEL_DECISION, 0.6)
    
    return (False, None, 0.0)


# =============================================================================
# 3. ENTITY EXTRACTION - Add to entity_extractor.py
# =============================================================================

class ExplanationEntities(BaseModel):
    """Entities extracted for explanation queries."""
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    patient_ids: List[str] = Field(default_factory=list, description="Multiple patient IDs for batch/compare")
    hcp_id: Optional[str] = Field(None, description="HCP context")
    model_type: Optional[str] = Field(None, description="Model type to explain")
    top_k: int = Field(default=5, description="Number of top features to return")
    include_narrative: bool = Field(default=True, description="Whether to include NL explanation")
    include_visualization: bool = Field(default=True, description="Whether to include chart")


# Patient ID patterns
PATIENT_ID_PATTERNS = [
    r"(?:patient\s+)?(?:id\s+)?([A-Z]{2,4}[-_]?\d{4}[-_]\d{4,8})",  # PAT-2024-001234
    r"(?:patient\s+)([A-Z0-9]{8,15})",                               # Alphanumeric IDs
    r"patient\s+#?(\d{6,12})",                                       # Numeric IDs
]

# HCP ID patterns  
HCP_ID_PATTERNS = [
    r"(?:hcp|doctor|physician|provider)\s+(?:id\s+)?([A-Z]{2,4}[-_]?[A-Z]{2}[-_]?\d{4,8})",
    r"(?:hcp|npi)\s+#?(\d{10})",
]

# Model type mapping
MODEL_TYPE_KEYWORDS = {
    "propensity": ["propensity", "likelihood", "probability", "chance"],
    "risk_stratification": ["risk", "stratification", "severity", "priority"],
    "next_best_action": ["nba", "next best", "recommendation", "suggest"],
    "churn_prediction": ["churn", "attrition", "dropout", "discontinuation"],
}


def extract_explanation_entities(query: str) -> ExplanationEntities:
    """
    Extract entities relevant to explanation queries.
    """
    query_lower = query.lower()
    entities = ExplanationEntities()
    
    # Extract patient IDs
    for pattern in PATIENT_ID_PATTERNS:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            if len(matches) == 1:
                entities.patient_id = matches[0].upper()
            else:
                entities.patient_ids = [m.upper() for m in matches]
            break
    
    # Extract HCP ID
    for pattern in HCP_ID_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            entities.hcp_id = match.group(1).upper()
            break
    
    # Detect model type
    for model_type, keywords in MODEL_TYPE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            entities.model_type = model_type
            break
    
    # Default model type if patient context detected but no explicit type
    if entities.patient_id and not entities.model_type:
        entities.model_type = "propensity"  # Most common use case
    
    # Detect visualization preference
    if "no chart" in query_lower or "text only" in query_lower:
        entities.include_visualization = False
    
    # Detect top_k preference
    top_k_match = re.search(r"(?:top|show)\s+(\d+)\s+(?:features?|factors?)", query_lower)
    if top_k_match:
        entities.top_k = min(int(top_k_match.group(1)), 10)
    
    return entities


# =============================================================================
# 4. ORCHESTRATOR TOOL - Add to agents/orchestrator/tools/
# =============================================================================

class ExplainAPITool:
    """
    Tool for the Orchestrator to call the Real-Time SHAP API.
    
    This tool:
    1. Calls /explain/predict endpoint
    2. Formats response for chat
    3. Triggers visualization generation
    4. Optionally calls explainer agent for narrative
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000/api/v1",
        timeout: float = 10.0
    ):
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def explain_patient_prediction(
        self,
        patient_id: str,
        model_type: str = "propensity",
        hcp_id: Optional[str] = None,
        top_k: int = 5,
        include_narrative: bool = True
    ) -> Dict[str, Any]:
        """
        Get real-time SHAP explanation for a patient prediction.
        
        Args:
            patient_id: Patient identifier
            model_type: Type of model to explain
            hcp_id: Optional HCP context
            top_k: Number of top features to return
            include_narrative: Whether to request NL explanation
        
        Returns:
            Explanation result with SHAP values and optional narrative
        """
        try:
            response = await self.client.post(
                f"{self.api_base_url}/explain/predict",
                json={
                    "patient_id": patient_id,
                    "hcp_id": hcp_id,
                    "model_type": model_type,
                    "top_k": top_k,
                    "format": "narrative" if include_narrative else "top_k",
                    "store_for_audit": True,
                    "include_base_value": True
                }
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        
        except httpx.HTTPError as e:
            logger.error(f"Explain API error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_explanation_history(
        self,
        patient_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get explanation history for a patient."""
        try:
            response = await self.client.get(
                f"{self.api_base_url}/explain/history/{patient_id}",
                params={"limit": limit}
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        
        except httpx.HTTPError as e:
            logger.error(f"History API error: {e}")
            return {"success": False, "error": str(e)}
    
    async def batch_explain(
        self,
        patient_ids: List[str],
        model_type: str = "propensity"
    ) -> Dict[str, Any]:
        """Get explanations for multiple patients."""
        try:
            requests = [
                {
                    "patient_id": pid,
                    "model_type": model_type,
                    "top_k": 5,
                    "store_for_audit": True
                }
                for pid in patient_ids[:50]  # Limit to 50
            ]
            
            response = await self.client.post(
                f"{self.api_base_url}/explain/predict/batch",
                json={"requests": requests, "parallel": True}
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        
        except httpx.HTTPError as e:
            logger.error(f"Batch explain API error: {e}")
            return {"success": False, "error": str(e)}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# =============================================================================
# 5. RESPONSE FORMATTER - For chat response with visualization
# =============================================================================

class ExplanationResponseFormatter:
    """
    Formats SHAP explanation results for chat display.
    
    Produces:
    1. Summary metrics (prediction, confidence)
    2. Feature contribution list
    3. Visualization config for shap_viz.py
    4. Natural language narrative
    """
    
    @staticmethod
    def format_chat_response(
        explanation: Dict[str, Any],
        include_visualization: bool = True,
        include_narrative: bool = True
    ) -> Dict[str, Any]:
        """
        Format explanation for chat display.
        
        Returns structure compatible with E2I chat response format.
        """
        data = explanation.get("data", explanation)
        
        # Build response sections
        response = {
            "type": "explanation",
            "metadata": {
                "explanation_id": data.get("explanation_id"),
                "timestamp": data.get("request_timestamp"),
                "model_version": data.get("model_version_id"),
                "computation_time_ms": data.get("computation_time_ms")
            }
        }
        
        # Summary section
        prediction_class = data.get("prediction_class", "unknown")
        probability = data.get("prediction_probability", 0)
        
        response["summary"] = {
            "patient_id": data.get("patient_id"),
            "prediction": prediction_class.replace("_", " ").title(),
            "confidence": f"{probability:.1%}",
            "confidence_value": probability
        }
        
        # Feature contributions
        features = data.get("top_features", [])
        response["features"] = [
            {
                "name": f["feature_name"].replace("_", " ").title(),
                "value": f["feature_value"],
                "shap": f["shap_value"],
                "direction": f["contribution_direction"],
                "rank": f["contribution_rank"]
            }
            for f in features
        ]
        
        # Visualization config (for shap_viz.py)
        if include_visualization:
            response["visualization"] = {
                "type": "shap_waterfall",
                "config": {
                    "base_value": data.get("base_value", 0.5),
                    "final_value": probability,
                    "features": [
                        {
                            "name": f["feature_name"],
                            "value": f["shap_value"],
                            "display_value": f["feature_value"]
                        }
                        for f in features
                    ],
                    "colors": {
                        "positive": "#48bb78",
                        "negative": "#fc8181",
                        "base": "#a0aec0",
                        "total": "#667eea"
                    }
                }
            }
        
        # Narrative
        if include_narrative:
            response["narrative"] = ExplanationResponseFormatter._generate_narrative(data)
        
        return response
    
    @staticmethod
    def _generate_narrative(data: Dict[str, Any]) -> str:
        """
        Generate natural language explanation.
        
        This is a template-based fallback; the explainer agent
        can enhance this with more sophisticated reasoning.
        """
        patient_id = data.get("patient_id", "This patient")
        prediction = data.get("prediction_class", "").replace("_", " ")
        probability = data.get("prediction_probability", 0)
        features = data.get("top_features", [])
        
        # Separate positive and negative contributors
        positive = [f for f in features if f["shap_value"] > 0]
        negative = [f for f in features if f["shap_value"] < 0]
        
        # Build narrative
        narrative = f"**{patient_id}** has been classified as **{prediction}** "
        narrative += f"with **{probability:.1%} confidence**.\n\n"
        
        if positive:
            narrative += "**Factors increasing the score:**\n"
            for f in positive[:3]:
                name = f["feature_name"].replace("_", " ")
                narrative += f"- {name} = {f['feature_value']} (+{f['shap_value']:.3f})\n"
        
        if negative:
            narrative += "\n**Factors decreasing the score:**\n"
            for f in negative[:3]:
                name = f["feature_name"].replace("_", " ")
                narrative += f"- {name} = {f['feature_value']} ({f['shap_value']:.3f})\n"
        
        return narrative
    
    @staticmethod
    def format_history_response(history_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format explanation history for chat display."""
        explanations = history_data.get("explanations", [])
        
        return {
            "type": "explanation_history",
            "patient_id": history_data.get("patient_id"),
            "total": len(explanations),
            "explanations": [
                {
                    "date": exp.get("request_timestamp", "")[:10],
                    "model": exp.get("model_type", ""),
                    "prediction": exp.get("prediction_class", "").replace("_", " ").title(),
                    "confidence": f"{exp.get('prediction_probability', 0):.1%}",
                    "explanation_id": exp.get("explanation_id")
                }
                for exp in explanations
            ],
            "visualization": {
                "type": "explanation_timeline",
                "config": {
                    "data": [
                        {
                            "date": exp.get("request_timestamp"),
                            "probability": exp.get("prediction_probability", 0),
                            "class": exp.get("prediction_class")
                        }
                        for exp in explanations
                    ]
                }
            }
        }
    
    @staticmethod
    def format_comparison_response(
        explanations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format comparison of multiple patient explanations."""
        return {
            "type": "explanation_comparison",
            "patients": [
                {
                    "patient_id": exp.get("patient_id"),
                    "prediction": exp.get("prediction_class", "").replace("_", " ").title(),
                    "confidence": f"{exp.get('prediction_probability', 0):.1%}",
                    "top_feature": exp.get("top_features", [{}])[0].get("feature_name", "N/A")
                }
                for exp in explanations
            ],
            "visualization": {
                "type": "shap_comparison",
                "config": {
                    "patients": [exp.get("patient_id") for exp in explanations],
                    "features": list(set(
                        f["feature_name"]
                        for exp in explanations
                        for f in exp.get("top_features", [])
                    ))[:10]
                }
            }
        }


# =============================================================================
# 6. ORCHESTRATOR ROUTING - Add to agents/orchestrator/router.py
# =============================================================================

class ExplainIntentHandler:
    """
    Handler for EXPLAIN intents in the orchestrator.
    
    Routing logic:
    1. Extract explanation entities
    2. Call ExplainAPITool
    3. Format response
    4. Optionally enhance with explainer agent
    """
    
    def __init__(self, api_tool: ExplainAPITool):
        self.api_tool = api_tool
        self.formatter = ExplanationResponseFormatter()
    
    async def handle(
        self,
        query: str,
        sub_intent: ExplainSubIntent,
        entities: ExplanationEntities,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an EXPLAIN intent query.
        """
        if sub_intent == ExplainSubIntent.PATIENT_PREDICTION:
            return await self._handle_patient_explanation(entities)
        
        elif sub_intent == ExplainSubIntent.PREDICTION_HISTORY:
            return await self._handle_history(entities)
        
        elif sub_intent == ExplainSubIntent.COMPARE_EXPLANATIONS:
            return await self._handle_comparison(entities)
        
        elif sub_intent == ExplainSubIntent.FEATURE_IMPORTANCE:
            return await self._handle_feature_importance(entities, query)
        
        else:
            # Default to patient explanation if we have a patient_id
            if entities.patient_id:
                return await self._handle_patient_explanation(entities)
            return self._handle_ambiguous(query)
    
    async def _handle_patient_explanation(
        self,
        entities: ExplanationEntities
    ) -> Dict[str, Any]:
        """Handle single patient explanation request."""
        if not entities.patient_id:
            return {
                "type": "clarification_needed",
                "message": "I'd be happy to explain a prediction. Which patient would you like me to explain?",
                "suggestions": [
                    "Try: 'Why is patient PAT-2024-001234 flagged?'",
                    "Or: 'Explain the propensity score for PAT-2024-005678'"
                ]
            }
        
        result = await self.api_tool.explain_patient_prediction(
            patient_id=entities.patient_id,
            model_type=entities.model_type or "propensity",
            hcp_id=entities.hcp_id,
            top_k=entities.top_k,
            include_narrative=entities.include_narrative
        )
        
        if result["success"]:
            return self.formatter.format_chat_response(
                result,
                include_visualization=entities.include_visualization,
                include_narrative=entities.include_narrative
            )
        else:
            return {
                "type": "error",
                "message": f"Unable to retrieve explanation: {result.get('error', 'Unknown error')}",
                "suggestion": "Please check the patient ID and try again."
            }
    
    async def _handle_history(
        self,
        entities: ExplanationEntities
    ) -> Dict[str, Any]:
        """Handle explanation history request."""
        if not entities.patient_id:
            return {
                "type": "clarification_needed",
                "message": "Which patient's explanation history would you like to see?"
            }
        
        result = await self.api_tool.get_explanation_history(
            patient_id=entities.patient_id,
            limit=10
        )
        
        if result["success"]:
            return self.formatter.format_history_response(result["data"])
        else:
            return {"type": "error", "message": f"Unable to retrieve history: {result.get('error')}"}
    
    async def _handle_comparison(
        self,
        entities: ExplanationEntities
    ) -> Dict[str, Any]:
        """Handle comparison of multiple patient explanations."""
        patient_ids = entities.patient_ids or ([entities.patient_id] if entities.patient_id else [])
        
        if len(patient_ids) < 2:
            return {
                "type": "clarification_needed",
                "message": "Please provide at least two patient IDs to compare.",
                "suggestions": [
                    "Try: 'Compare explanations for PAT-001 and PAT-002'",
                    "Or: 'Why is PAT-001 different from PAT-002?'"
                ]
            }
        
        result = await self.api_tool.batch_explain(
            patient_ids=patient_ids[:5],  # Limit to 5 for comparison
            model_type=entities.model_type or "propensity"
        )
        
        if result["success"]:
            explanations = result["data"].get("explanations", [])
            return self.formatter.format_comparison_response(explanations)
        else:
            return {"type": "error", "message": f"Unable to compare: {result.get('error')}"}
    
    async def _handle_feature_importance(
        self,
        entities: ExplanationEntities,
        query: str
    ) -> Dict[str, Any]:
        """Handle general feature importance query."""
        # If patient context, explain that patient
        if entities.patient_id:
            return await self._handle_patient_explanation(entities)
        
        # Otherwise, explain this is about global vs local importance
        return {
            "type": "guidance",
            "message": (
                "I can explain feature importance in two ways:\n\n"
                "1. **For a specific patient**: Tell me which patient, and I'll show "
                "which features drove that particular prediction.\n\n"
                "2. **Global importance**: Ask 'What features are most important for "
                "the propensity model overall?' to see aggregate feature rankings."
            ),
            "suggestions": [
                "Explain features for patient PAT-2024-001234",
                "What are the most important features for propensity?"
            ]
        }
    
    def _handle_ambiguous(self, query: str) -> Dict[str, Any]:
        """Handle ambiguous explanation request."""
        return {
            "type": "clarification_needed",
            "message": "I can help explain model predictions. What would you like to understand?",
            "suggestions": [
                "Why is patient [ID] flagged for Remibrutinib?",
                "Explain the risk score for patient [ID]",
                "What drove the recommendation for [patient ID]?",
                "Show explanation history for patient [ID]"
            ]
        }


# =============================================================================
# 7. INTEGRATION EXAMPLE - How to wire this into the orchestrator
# =============================================================================

"""
INTEGRATION INSTRUCTIONS
========================

1. Add to nlp/models/intent_models.py:
   - Add EXPLAIN to IntentType enum
   - Import ExplainSubIntent

2. Add to nlp/intent_classifier.py:
   - Import classify_explain_intent
   - Call it early in classification pipeline
   - Return EXPLAIN intent if matched

3. Add to nlp/entity_extractor.py:
   - Import ExplanationEntities, extract_explanation_entities
   - Call for EXPLAIN intents

4. Add to agents/orchestrator/router.py:
   
   ```python
   from src.agents.orchestrator.tools.explain_tool import (
       ExplainAPITool,
       ExplainIntentHandler,
       classify_explain_intent,
       extract_explanation_entities
   )
   
   class OrchestratorRouter:
       def __init__(self):
           # ... existing init ...
           self.explain_tool = ExplainAPITool()
           self.explain_handler = ExplainIntentHandler(self.explain_tool)
       
       async def route(self, parsed_query):
           intent = parsed_query.intent
           
           # Check for EXPLAIN intent
           if intent == IntentType.EXPLAIN:
               is_explain, sub_intent, confidence = classify_explain_intent(
                   parsed_query.raw_query
               )
               entities = extract_explanation_entities(parsed_query.raw_query)
               return await self.explain_handler.handle(
                   query=parsed_query.raw_query,
                   sub_intent=sub_intent,
                   entities=entities
               )
           
           # ... existing routing ...
   ```

5. Add visualization rendering to chat response handler:
   
   ```python
   # In chat response handler
   if response.get("visualization"):
       viz_type = response["visualization"]["type"]
       if viz_type == "shap_waterfall":
           # Call visualization/shap_viz.py to render
           chart_config = shap_viz.create_waterfall_chart(
               response["visualization"]["config"]
           )
           response["chart"] = chart_config
   ```

6. Add to domain_vocabulary.yaml:
   
   ```yaml
   intents:
     # ... existing ...
     - explain
   
   explain_sub_intents:
     - patient_prediction
     - model_decision
     - feature_importance
     - prediction_history
     - compare_explanations
   ```
"""


# =============================================================================
# 8. EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example of using the chat integration."""
    
    # Initialize
    api_tool = ExplainAPITool()
    handler = ExplainIntentHandler(api_tool)
    
    # Example queries
    queries = [
        "Why is patient PAT-2024-001234 flagged for Remibrutinib?",
        "Explain the propensity score for PAT-2024-005678",
        "What factors drove the recommendation for this patient?",
        "Show me the explanation history for PAT-2024-001234",
        "Compare why PAT-001 and PAT-002 have different scores"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        
        # Classify intent
        is_explain, sub_intent, confidence = classify_explain_intent(query)
        print(f"Intent: EXPLAIN={is_explain}, SubIntent={sub_intent}, Confidence={confidence}")
        
        # Extract entities
        entities = extract_explanation_entities(query)
        print(f"Entities: {entities}")
        
        # Handle (would make API call in production)
        # response = await handler.handle(query, sub_intent, entities)
        # print(f"Response type: {response.get('type')}")
    
    await api_tool.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
