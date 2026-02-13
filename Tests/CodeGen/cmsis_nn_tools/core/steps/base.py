"""
Base classes for pipeline steps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class StepStatus(Enum):
    """Status of a pipeline step execution."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a pipeline step."""
    status: StepStatus
    message: str
    error: Optional[Exception] = None
    
    @property
    def success(self) -> bool:
        """Check if step succeeded."""
        return self.status == StepStatus.SUCCESS
    
    @property
    def skipped(self) -> bool:
        """Check if step was skipped."""
        return self.status == StepStatus.SKIPPED


class StepBase(ABC):
    """
    Base class for all pipeline steps.
    
    Each step represents a single operation in the test pipeline:
    - generate: TFLite model generation
    - runners: Unity test runner generation
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
    
    @abstractmethod
    def execute(self) -> StepResult:
        """
        Execute this step.
        
        Returns:
            StepResult indicating success, failure, or skip
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
            status=StepStatus.SKIPPED,
            message=f"DRY RUN: Would execute {self.name} step"
        )
