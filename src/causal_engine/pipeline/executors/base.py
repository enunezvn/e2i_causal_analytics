"""LibraryExecutor abstract base class.

Defines the contract every per-library executor (NetworkX, DoWhy, EconML,
CausalML) must satisfy. Locked in phase C-1 of GH #354 — Wave-1 dispatchers
(C-2..C-5) MUST NOT add/remove/rename abstract methods. Reopen via a separate
PR if the contract needs to change.

Contract:
- `library` (@property @abstractmethod) -> CausalLibrary
- `execute(state, config)` (async @abstractmethod) -> LibraryExecutionResult
- `validate_input(state)` (@abstractmethod) -> tuple[bool, str]
"""

from abc import ABC, abstractmethod

from ..router import CausalLibrary
from ..state import LibraryExecutionResult, PipelineConfig, PipelineState


class LibraryExecutor(ABC):
    """Abstract base class for library-specific executors."""

    @property
    @abstractmethod
    def library(self) -> CausalLibrary:
        """Return the library this executor handles."""
        pass

    @abstractmethod
    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute the library's analysis and return results.

        Args:
            state: Current pipeline state with input data
            config: Pipeline configuration

        Returns:
            LibraryExecutionResult with success/failure and result data
        """
        pass

    @abstractmethod
    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate that input state has required fields for this library.

        Args:
            state: Current pipeline state

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
