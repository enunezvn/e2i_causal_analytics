"""
E2I Causal Analytics - Audit Chain Service
Provides tamper-evident logging for agent actions with hash-linked chains.

This module implements a lightweight cryptographic audit trail inspired by
AgentField's W3C DID/VC approach, adapted for pharmaceutical compliance
requirements (HIPAA, FDA) without the full DID infrastructure complexity.

Version: 4.1
Date: December 2025
"""

import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import time

from supabase import Client
from pydantic import BaseModel, Field


class AgentTier(Enum):
    """
    Agent tier classifications matching domain_vocabulary.yaml.
    
    Tier 0: ML Foundation (scope_definer, data_preparer, model_selector, etc.)
    Tier 1: Coordination (orchestrator)
    Tier 2: Causal Analytics (causal_impact, gap_analyzer, heterogeneous_optimizer)
    Tier 3: Monitoring (drift_monitor, experiment_designer, health_score)
    Tier 4: ML Predictions (prediction_synthesizer, resource_optimizer)
    Tier 5: Self-Improvement (explainer, feedback_learner)
    """
    ML_FOUNDATION = 0
    COORDINATION = 1
    CAUSAL_ANALYTICS = 2
    MONITORING = 3
    ML_PREDICTIONS = 4
    SELF_IMPROVEMENT = 5


@dataclass
class RefutationResults:
    """Results from DoWhy refutation tests"""
    placebo_treatment: Optional[bool] = None
    random_common_cause: Optional[bool] = None
    data_subset: Optional[bool] = None
    unobserved_confound: Optional[bool] = None
    
    @property
    def all_passed(self) -> bool:
        """Check if all executed tests passed"""
        tests = [self.placebo_treatment, self.random_common_cause, 
                 self.data_subset, self.unobserved_confound]
        executed = [t for t in tests if t is not None]
        return all(executed) if executed else False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "placebo_treatment": self.placebo_treatment,
            "random_common_cause": self.random_common_cause,
            "data_subset": self.data_subset,
            "unobserved_confound": self.unobserved_confound
        }


@dataclass
class AuditChainEntry:
    """
    Single entry in the audit chain.
    
    Each entry contains:
    - Identification: entry_id, workflow_id, sequence_number
    - Agent info: agent_name, agent_tier, action_type
    - Timing: created_at, duration_ms
    - Payload hashes: input_hash, output_hash
    - Validation: validation_passed, confidence_score, refutation_results
    - Chain linking: previous_entry_id, previous_hash, entry_hash
    - Metadata: user_id, session_id, query_text, brand
    """
    entry_id: UUID
    workflow_id: UUID
    sequence_number: int
    agent_name: str
    agent_tier: int
    action_type: str
    created_at: datetime
    duration_ms: Optional[int] = None
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    validation_passed: Optional[bool] = None
    confidence_score: Optional[float] = None
    refutation_results: Optional[Dict] = None
    previous_entry_id: Optional[UUID] = None
    previous_hash: Optional[str] = None
    entry_hash: str = ""
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    query_text: Optional[str] = None
    brand: Optional[str] = None
    
    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        return {
            "entry_id": str(self.entry_id),
            "workflow_id": str(self.workflow_id),
            "sequence_number": self.sequence_number,
            "agent_name": self.agent_name,
            "agent_tier": self.agent_tier,
            "action_type": self.action_type,
            "created_at": self.created_at.isoformat(),
            "duration_ms": self.duration_ms,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "validation_passed": self.validation_passed,
            "confidence_score": self.confidence_score,
            "refutation_results": self.refutation_results,
            "previous_entry_id": str(self.previous_entry_id) if self.previous_entry_id else None,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "user_id": self.user_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "query_text": self.query_text,
            "brand": self.brand
        }


@dataclass
class ChainVerificationResult:
    """Result of chain integrity verification"""
    is_valid: bool
    entries_checked: int
    first_invalid_entry: Optional[UUID] = None
    error_message: Optional[str] = None
    verified_at: datetime = field(default_factory=datetime.utcnow)


class AuditChainService:
    """
    Service for creating and verifying tamper-evident audit chains.
    
    Each agent action in a workflow is linked via SHA-256 hash to the
    previous action, creating a verifiable chain of custody.
    
    Usage:
        service = AuditChainService(supabase_client)
        
        # Start a new workflow
        genesis = service.start_workflow(
            agent_name="causal_impact",
            agent_tier=AgentTier.CAUSAL_ANALYTICS,
            action_type="graph_builder",
            input_data={"treatment": "rep_visit", "outcome": "prescription"},
            user_id="analyst@pharma.com",
            brand="Kisqali"
        )
        
        # Add subsequent entries (automatically linked)
        with service.timed_entry(genesis.workflow_id, "causal_impact", 
                                  AgentTier.CAUSAL_ANALYTICS, "estimation") as entry:
            result = run_estimation()
            entry.output_hash = service.hash_payload(result)
            entry.confidence_score = result.confidence
        
        # Verify chain integrity
        verification = service.verify_workflow(genesis.workflow_id)
        assert verification.is_valid
    """
    
    def __init__(self, supabase_client: Client):
        self.db = supabase_client
        self._workflow_cache: Dict[UUID, AuditChainEntry] = {}
    
    # =========================================================================
    # Hash Computation
    # =========================================================================
    
    def _compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash of input string"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _compute_entry_hash(self, entry: AuditChainEntry) -> str:
        """
        Compute the chain hash for an entry.
        
        IMPORTANT: This must match the PostgreSQL compute_entry_hash function
        exactly to ensure verification works correctly.
        """
        components = [
            str(entry.entry_id),
            str(entry.workflow_id),
            str(entry.sequence_number),
            entry.agent_name,
            entry.action_type,
            entry.created_at.isoformat(),
            entry.input_hash or "",
            entry.output_hash or "",
            entry.previous_hash or "GENESIS"
        ]
        return self._compute_hash("".join(components))
    
    def hash_payload(self, payload: Any) -> str:
        """
        Hash any JSON-serializable payload for input/output tracking.
        
        Args:
            payload: Any JSON-serializable object
            
        Returns:
            SHA-256 hash of the serialized payload
        """
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return self._compute_hash(serialized)
    
    # =========================================================================
    # Workflow Management
    # =========================================================================
    
    def start_workflow(
        self,
        agent_name: str,
        agent_tier: AgentTier,
        action_type: str,
        input_data: Optional[Any] = None,
        user_id: Optional[str] = None,
        session_id: Optional[UUID] = None,
        query_text: Optional[str] = None,
        brand: Optional[str] = None,
        auto_commit: bool = True
    ) -> AuditChainEntry:
        """
        Start a new workflow audit chain (genesis block).
        
        Args:
            agent_name: Name of the initiating agent
            agent_tier: Tier classification of the agent
            action_type: Type of action being performed
            input_data: Optional input data to hash
            user_id: User who triggered the workflow
            session_id: Session reference
            query_text: Original user query
            brand: Brand context (Remibrutinib, Fabhalta, Kisqali)
            auto_commit: Whether to immediately persist to database
            
        Returns:
            The genesis entry for the workflow
        """
        workflow_id = uuid4()
        entry = AuditChainEntry(
            entry_id=uuid4(),
            workflow_id=workflow_id,
            sequence_number=1,
            agent_name=agent_name,
            agent_tier=agent_tier.value,
            action_type=action_type,
            created_at=datetime.utcnow(),
            input_hash=self.hash_payload(input_data) if input_data else None,
            previous_entry_id=None,  # Genesis has no previous
            previous_hash=None,      # Genesis has no previous hash
            user_id=user_id,
            session_id=session_id,
            query_text=query_text,
            brand=brand
        )
        entry.entry_hash = self._compute_entry_hash(entry)
        
        # Cache for chain continuation
        self._workflow_cache[workflow_id] = entry
        
        if auto_commit:
            self.commit_entry(entry)
        
        return entry
    
    def add_entry(
        self,
        workflow_id: UUID,
        agent_name: str,
        agent_tier: AgentTier,
        action_type: str,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
        duration_ms: Optional[int] = None,
        validation_passed: Optional[bool] = None,
        confidence_score: Optional[float] = None,
        refutation_results: Optional[RefutationResults] = None,
        auto_commit: bool = True
    ) -> AuditChainEntry:
        """
        Add a new entry to an existing workflow chain.
        
        Automatically links to the previous entry via hash.
        
        Args:
            workflow_id: ID of the workflow to add to
            agent_name: Name of the agent performing the action
            agent_tier: Tier classification of the agent
            action_type: Type of action being performed
            input_data: Optional input data to hash
            output_data: Optional output data to hash
            duration_ms: Execution time in milliseconds
            validation_passed: Whether validation tests passed
            confidence_score: Confidence level (0.0 to 1.0)
            refutation_results: DoWhy refutation test results
            auto_commit: Whether to immediately persist to database
            
        Returns:
            The new entry (already linked to previous)
            
        Raises:
            ValueError: If workflow_id is not found
        """
        # Get previous entry from cache or database
        previous = self._get_last_entry(workflow_id)
        
        if previous is None:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        entry = AuditChainEntry(
            entry_id=uuid4(),
            workflow_id=workflow_id,
            sequence_number=previous.sequence_number + 1,
            agent_name=agent_name,
            agent_tier=agent_tier.value,
            action_type=action_type,
            created_at=datetime.utcnow(),
            duration_ms=duration_ms,
            input_hash=self.hash_payload(input_data) if input_data else None,
            output_hash=self.hash_payload(output_data) if output_data else None,
            validation_passed=validation_passed,
            confidence_score=confidence_score,
            refutation_results=refutation_results.to_dict() if refutation_results else None,
            previous_entry_id=previous.entry_id,
            previous_hash=previous.entry_hash,  # THE KEY LINK
            user_id=previous.user_id,
            session_id=previous.session_id,
            brand=previous.brand
        )
        entry.entry_hash = self._compute_entry_hash(entry)
        
        # Update cache
        self._workflow_cache[workflow_id] = entry
        
        if auto_commit:
            self.commit_entry(entry)
        
        return entry
    
    @contextmanager
    def timed_entry(
        self,
        workflow_id: UUID,
        agent_name: str,
        agent_tier: AgentTier,
        action_type: str,
        input_data: Optional[Any] = None
    ):
        """
        Context manager for creating timed audit entries.
        
        Automatically records duration and commits on exit.
        
        Usage:
            with service.timed_entry(workflow_id, "causal_impact", 
                                     AgentTier.CAUSAL_ANALYTICS, "estimation") as entry:
                result = run_estimation()
                entry.output_hash = service.hash_payload(result)
                entry.confidence_score = result.confidence
        """
        start_time = time.time()
        
        # Create entry without auto-commit
        entry = self.add_entry(
            workflow_id=workflow_id,
            agent_name=agent_name,
            agent_tier=agent_tier,
            action_type=action_type,
            input_data=input_data,
            auto_commit=False
        )
        
        try:
            yield entry
        finally:
            # Calculate duration and recompute hash
            entry.duration_ms = int((time.time() - start_time) * 1000)
            entry.entry_hash = self._compute_entry_hash(entry)
            self.commit_entry(entry)
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def commit_entry(self, entry: AuditChainEntry) -> None:
        """
        Persist an entry to the database.
        
        Args:
            entry: The audit chain entry to persist
        """
        self.db.table("audit_chain_entries").insert(entry.to_db_dict()).execute()
    
    def _get_last_entry(self, workflow_id: UUID) -> Optional[AuditChainEntry]:
        """Get the last entry in a workflow chain"""
        # Check cache first
        if workflow_id in self._workflow_cache:
            return self._workflow_cache[workflow_id]
        
        # Query database
        result = self.db.table("audit_chain_entries")\
            .select("*")\
            .eq("workflow_id", str(workflow_id))\
            .order("sequence_number", desc=True)\
            .limit(1)\
            .execute()
        
        if not result.data:
            return None
        
        row = result.data[0]
        entry = AuditChainEntry(
            entry_id=UUID(row["entry_id"]),
            workflow_id=UUID(row["workflow_id"]),
            sequence_number=row["sequence_number"],
            agent_name=row["agent_name"],
            agent_tier=row["agent_tier"],
            action_type=row["action_type"],
            created_at=datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')),
            duration_ms=row.get("duration_ms"),
            input_hash=row.get("input_hash"),
            output_hash=row.get("output_hash"),
            validation_passed=row.get("validation_passed"),
            confidence_score=row.get("confidence_score"),
            refutation_results=row.get("refutation_results"),
            previous_entry_id=UUID(row["previous_entry_id"]) if row.get("previous_entry_id") else None,
            previous_hash=row.get("previous_hash"),
            entry_hash=row["entry_hash"],
            user_id=row.get("user_id"),
            session_id=UUID(row["session_id"]) if row.get("session_id") else None,
            brand=row.get("brand")
        )
        
        # Cache for future use
        self._workflow_cache[workflow_id] = entry
        return entry
    
    # =========================================================================
    # Verification
    # =========================================================================
    
    def verify_workflow(self, workflow_id: UUID, log_verification: bool = True) -> ChainVerificationResult:
        """
        Verify the integrity of a workflow's audit chain.
        
        Calls the PostgreSQL verify_chain_integrity function and optionally
        logs the verification result.
        
        Args:
            workflow_id: The workflow to verify
            log_verification: Whether to log the verification result
            
        Returns:
            ChainVerificationResult with validity status
        """
        result = self.db.rpc(
            "verify_chain_integrity",
            {"p_workflow_id": str(workflow_id)}
        ).execute()
        
        verification_data = result.data[0] if result.data else {}
        
        verification = ChainVerificationResult(
            is_valid=verification_data.get("is_valid", False),
            entries_checked=verification_data.get("entries_checked", 0),
            first_invalid_entry=UUID(verification_data["first_invalid_entry"]) 
                if verification_data.get("first_invalid_entry") else None,
            error_message=verification_data.get("error_message")
        )
        
        if log_verification:
            self._log_verification(workflow_id, verification)
        
        return verification
    
    def verify_workflow_local(self, workflow_id: UUID) -> ChainVerificationResult:
        """
        Verify chain integrity locally (without database function).
        
        Useful for offline verification or when database functions unavailable.
        """
        result = self.db.table("audit_chain_entries")\
            .select("*")\
            .eq("workflow_id", str(workflow_id))\
            .order("sequence_number")\
            .execute()
        
        if not result.data:
            return ChainVerificationResult(
                is_valid=False,
                entries_checked=0,
                error_message="Workflow not found"
            )
        
        prev_hash = None
        for i, row in enumerate(result.data):
            # Verify previous_hash links correctly
            if i > 0 and row.get("previous_hash") != prev_hash:
                return ChainVerificationResult(
                    is_valid=False,
                    entries_checked=i + 1,
                    first_invalid_entry=UUID(row["entry_id"]),
                    error_message=f"Previous hash mismatch at sequence {row['sequence_number']}"
                )
            
            # Recompute entry hash
            entry = AuditChainEntry(
                entry_id=UUID(row["entry_id"]),
                workflow_id=UUID(row["workflow_id"]),
                sequence_number=row["sequence_number"],
                agent_name=row["agent_name"],
                agent_tier=row["agent_tier"],
                action_type=row["action_type"],
                created_at=datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')),
                input_hash=row.get("input_hash"),
                output_hash=row.get("output_hash"),
                previous_hash=row.get("previous_hash"),
                entry_hash=row["entry_hash"]
            )
            
            expected_hash = self._compute_entry_hash(entry)
            if row["entry_hash"] != expected_hash:
                return ChainVerificationResult(
                    is_valid=False,
                    entries_checked=i + 1,
                    first_invalid_entry=UUID(row["entry_id"]),
                    error_message=f"Entry hash mismatch at sequence {row['sequence_number']}"
                )
            
            prev_hash = row["entry_hash"]
        
        return ChainVerificationResult(
            is_valid=True,
            entries_checked=len(result.data)
        )
    
    def _log_verification(self, workflow_id: UUID, result: ChainVerificationResult) -> None:
        """Log a verification result to the database"""
        self.db.table("audit_chain_verification_log").insert({
            "workflow_id": str(workflow_id),
            "entries_verified": result.entries_checked,
            "chain_valid": result.is_valid,
            "first_broken_entry": str(result.first_invalid_entry) if result.first_invalid_entry else None,
            "verified_by": "system",
            "verification_notes": result.error_message
        }).execute()
    
    # =========================================================================
    # Queries
    # =========================================================================
    
    def get_workflow_summary(self, workflow_id: UUID) -> Optional[Dict[str, Any]]:
        """Get summary of a workflow from the v_audit_chain_summary view"""
        result = self.db.table("v_audit_chain_summary")\
            .select("*")\
            .eq("workflow_id", str(workflow_id))\
            .execute()
        
        return result.data[0] if result.data else None
    
    def get_causal_validations(
        self, 
        brand: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get causal validation chain entries.
        
        Args:
            brand: Filter by brand (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of validation entries from v_causal_validation_chain
        """
        query = self.db.table("v_causal_validation_chain").select("*")
        
        if brand:
            query = query.eq("brand", brand)
        if start_date:
            query = query.gte("created_at", start_date.isoformat())
        if end_date:
            query = query.lte("created_at", end_date.isoformat())
        
        result = query.order("created_at", desc=True).execute()
        return result.data


# =============================================================================
# Convenience Functions
# =============================================================================

def create_audit_chain_service(supabase_url: str, supabase_key: str) -> AuditChainService:
    """Factory function to create AuditChainService with Supabase client"""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return AuditChainService(client)
