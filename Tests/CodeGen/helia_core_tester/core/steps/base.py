"""
Base classes for pipeline steps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class StepStatus(Enum):
    """Status of a pipeline step execution."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a pipeline step."""
    name: str
    status: StepStatus
    message: str
    duration_sec: Optional[float] = None
    outputs: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    should_continue: bool = True
    
    @property
    def success(self) -> bool:
        """Check if step succeeded."""
        return self.status == StepStatus.SUCCESS
    
    @property
    def skipped(self) -> bool:
        """Check if step was skipped."""
        return self.status == StepStatus.SKIPPED


@dataclass
class StepPlan:
    """Plan for a pipeline step (no execution)."""
    name: str
    will_run: bool
    reason: str
    commands: List[List[str]] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class StepBase(ABC):
    """
    Base class for all pipeline steps.
    
    Each step represents a single operation in the test pipeline:
    - generate: TFLite model generation
    - build: CMake build
    - run: FVP execution
    - clean: Cleanup
    """
    
    def __init__(self, config):
        """
        Initialize step with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = None  # Will be set by subclasses
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this step."""
        pass
    
    def execute(self) -> StepResult:
        """Execute this step (skip/dry_run/validate then _do_execute)."""
        if self.should_skip():
            return StepResult(
                name=self.name,
                status=StepStatus.SKIPPED,
                message=f"{self.name} step skipped by config"
            )
        if self.config.dry_run:
            return self.dry_run()
        error = self.validate()
        if error:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                message=error,
                should_continue=not self.config.fail_fast
            )
        start = time.perf_counter()
        result = self._do_execute()
        duration = time.perf_counter() - start
        result.duration_sec = duration
        if result.status == StepStatus.FAILED:
            result.should_continue = not self.config.fail_fast
        return result

    @abstractmethod
    def _do_execute(self) -> StepResult:
        """
        Perform the step work. Called by execute() after skip/dry_run/validate.
        
        Returns:
            StepResult indicating success or failure
        """
        pass

    def should_skip(self) -> bool:
        """
        Check if this step should be skipped.
        
        Returns:
            True if step should be skipped
        """
        return False
    
    def validate(self) -> Optional[str]:
        """
        Validate prerequisites for this step.
        
        Returns:
            None if valid, error message string if invalid
        """
        return None
    
    def dry_run(self) -> StepResult:
        """
        Perform a dry run of this step (show what would be done).
        
        Returns:
            StepResult with status SKIPPED and message describing what would be done
        """
        return StepResult(
            name=self.name,
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would execute {self.name} step"
        )

    def plan(self) -> StepPlan:
        """Return a plan for this step without executing it."""
        if self.should_skip():
            return StepPlan(
                name=self.name,
                will_run=False,
                reason="skipped by config"
            )
        validation_error = self.plan_validate()
        if validation_error:
            return StepPlan(
                name=self.name,
                will_run=False,
                reason=f"invalid: {validation_error}"
            )
        return self._plan_details()

    def plan_validate(self) -> Optional[str]:
        """Validation for plan mode. Override for lenient checks."""
        return self.validate()

    def _plan_details(self) -> StepPlan:
        """Provide commands/outputs for plan mode."""
        return StepPlan(
            name=self.name,
            will_run=True,
            reason="ready"
        )
