# API/Backend Specialist Instructions

## Domain Scope
You are the API/Backend specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `src/api/` - FastAPI application
- `src/api/routes/` - API endpoints
- `src/api/middleware/` - Request processing
- `src/api/schemas/` - API request/response schemas

## Technology Stack
- **Framework**: FastAPI 0.100+
- **Validation**: Pydantic v2
- **Auth**: JWT tokens (Supabase Auth)
- **WebSocket**: For streaming responses

## API Structure

```
src/api/
├── __init__.py
├── main.py                 # FastAPI app factory
├── dependencies.py         # Dependency injection
├── routes/
│   ├── __init__.py
│   ├── chat.py            # /api/v1/chat endpoints
│   ├── agents.py          # /api/v1/agents endpoints
│   ├── kpis.py            # /api/v1/kpis endpoints
│   ├── splits.py          # /api/v1/splits endpoints
│   ├── visualizations.py  # /api/v1/viz endpoints
│   └── health.py          # /api/v1/health endpoints
├── middleware/
│   ├── __init__.py
│   ├── auth.py            # JWT validation
│   ├── logging.py         # Request logging
│   └── rate_limit.py      # Rate limiting
├── schemas/
│   ├── __init__.py
│   ├── chat.py            # Chat request/response
│   ├── agent.py           # Agent schemas
│   ├── kpi.py             # KPI schemas
│   └── visualization.py   # Viz schemas
└── services/
    ├── __init__.py
    ├── chat_service.py    # Chat orchestration
    ├── agent_service.py   # Agent coordination
    └── kpi_service.py     # KPI calculations
```

## Core Endpoints

### Chat Endpoints
```python
# routes/chat.py
from fastapi import APIRouter, Depends, WebSocket
from ..schemas.chat import ChatRequest, ChatResponse, StreamChunk

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """
    Process a chat message through the NLP → Agent pipeline.
    
    Flow:
    1. NLP: Parse query, extract entities, classify intent
    2. RAG: Retrieve relevant context
    3. Orchestrator: Route to appropriate agent(s)
    4. Agent(s): Generate analysis
    5. Response: Synthesize and return
    """
    return await chat_service.process(request)

@router.websocket("/stream")
async def stream_response(
    websocket: WebSocket,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Stream response chunks via WebSocket.
    
    Chunks include:
    - status: Processing stage updates
    - partial: Partial response text
    - visualization: Chart/graph data
    - complete: Final response
    """
    await websocket.accept()
    # ...
```

### Agent Endpoints
```python
# routes/agents.py
from fastapi import APIRouter, Depends
from ..schemas.agent import AgentInfo, AgentActivity

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/", response_model=List[AgentInfo])
async def list_agents(
    tier: Optional[int] = None,
    agent_service: AgentService = Depends(get_agent_service)
) -> List[AgentInfo]:
    """List all available agents, optionally filtered by tier."""
    return await agent_service.list_agents(tier=tier)

@router.get("/{agent_name}/activities", response_model=List[AgentActivity])
async def get_agent_activities(
    agent_name: str,
    limit: int = 10,
    agent_service: AgentService = Depends(get_agent_service)
) -> List[AgentActivity]:
    """Get recent activities for a specific agent."""
    return await agent_service.get_activities(agent_name, limit)
```

### KPI Endpoints
```python
# routes/kpis.py
from fastapi import APIRouter, Depends, Query
from ..schemas.kpi import KPIValue, KPITrend, KPIDefinition

router = APIRouter(prefix="/kpis", tags=["kpis"])

@router.get("/", response_model=List[KPIDefinition])
async def list_kpis(
    category: Optional[str] = None
) -> List[KPIDefinition]:
    """List all 46 KPI definitions."""
    pass

@router.get("/{kpi_name}", response_model=KPIValue)
async def get_kpi_value(
    kpi_name: str,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    time_range: Optional[str] = Query(None, regex="^\\d+d$")
) -> KPIValue:
    """Get current value for a specific KPI."""
    pass

@router.get("/{kpi_name}/trend", response_model=KPITrend)
async def get_kpi_trend(
    kpi_name: str,
    brand: Optional[str] = None,
    granularity: str = "weekly"
) -> KPITrend:
    """Get trend data for a KPI."""
    pass
```

### Split Endpoints (ML Compliance)
```python
# routes/splits.py
from fastapi import APIRouter, Depends
from ..schemas.split import SplitStats, LeakageAuditResult

router = APIRouter(prefix="/splits", tags=["splits"])

@router.get("/stats", response_model=SplitStats)
async def get_split_stats() -> SplitStats:
    """Get current split distribution statistics."""
    pass

@router.post("/audit", response_model=LeakageAuditResult)
async def run_leakage_audit() -> LeakageAuditResult:
    """Run leakage audit on current data."""
    pass
```

## Request/Response Schemas

### Chat Schemas
```python
# schemas/chat.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    brand_context: Optional[str] = None
    include_visualizations: bool = True

class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    response_text: str
    intent_detected: str
    agents_used: List[str]
    visualizations: Optional[List[VisualizationData]]
    confidence: float
    processing_time_ms: int
    created_at: datetime

class StreamChunk(BaseModel):
    chunk_type: Literal["status", "partial", "visualization", "complete"]
    content: str
    metadata: Optional[Dict[str, Any]]
```

### Agent Schemas
```python
# schemas/agent.py
class AgentInfo(BaseModel):
    name: str
    tier: int
    description: str
    intents: List[str]
    is_active: bool

class AgentActivity(BaseModel):
    id: str
    agent_name: str
    query: str
    analysis_results: Dict[str, Any]
    confidence: float
    processing_time_ms: int
    created_at: datetime
```

## Middleware

### Authentication
```python
# middleware/auth.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
import jwt

class JWTAuth(HTTPBearer):
    async def __call__(self, request: Request) -> str:
        credentials = await super().__call__(request)
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"]
            )
            return payload["sub"]  # user_id
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

### Request Logging
```python
# middleware/logging.py
from fastapi import Request
import structlog

logger = structlog.get_logger()

async def log_requests(request: Request, call_next):
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        user_id=getattr(request.state, "user_id", None)
    )
    response = await call_next(request)
    logger.info(
        "request_completed",
        status_code=response.status_code
    )
    return response
```

## Error Handling

```python
# main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="E2I Causal Analytics API", version="3.0.0")

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "type": "validation_error"}
    )

@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": "agent_error",
            "agent": exc.agent_name
        }
    )
```

## Integration Contracts

### Frontend Contract
```yaml
# API responses must include:
- conversation_id (for continuity)
- visualizations array (for chart rendering)
- agents_used (for transparency)
- confidence score (for UX indicators)
```

### Agent Contract
```python
# ChatService must:
# 1. Always return within 30 seconds
# 2. Stream partial responses for long operations
# 3. Include agent attribution in response
```

## Testing Requirements
- `tests/unit/test_api/`
- All endpoints must have OpenAPI documentation
- Response time < 2s for 95th percentile (non-streaming)
- WebSocket connections must handle disconnects gracefully

## Handoff Format
```yaml
api_handoff:
  endpoints_affected: [<list>]
  schemas_changed: [<list>]
  breaking_changes: <bool>
  openapi_updated: <bool>
  auth_requirements: <unchanged|modified>
```
