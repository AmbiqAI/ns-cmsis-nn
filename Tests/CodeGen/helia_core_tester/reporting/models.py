"""
Data models for test reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class TestStatus(Enum):
    """Test execution status."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"
    BUILD_FAILED = "BUILD_FAILED"
    GENERATION_FAILED = "GENERATION_FAILED"
    CONVERSION_FAILED = "CONVERSION_FAILED"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    status: TestStatus
    duration: float  # seconds
    cpu: str
    elf_path: str
    failure_reason: Optional[str] = None
    skip_reason: Optional[str] = None
    output_lines: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    memory_usage: Optional[int] = None  # bytes
    cycles: Optional[int] = None  # for PMU runs
    exit_code: Optional[int] = None
    error_type: Optional[str] = None  # "assertion", "timeout", "crash", etc.
    descriptor_name: Optional[str] = None  # Link back to descriptor
    expected_output: Optional[str] = None  # Golden reference output
    actual_output: Optional[str] = None  # Actual test output
    output_differences: List[str] = field(default_factory=list)  # List of difference details
    project_root: Optional[Path] = None  # For making paths relative
    
    def _make_path_relative(self, path: str) -> str:
        """Convert absolute path to relative path if project_root is set."""
        if not self.project_root:
            return path
        try:
            path_obj = Path(path)
            if path_obj.is_absolute():
                try:
                    return str(path_obj.relative_to(self.project_root))
                except ValueError:
                    # Path is not under project_root, return as-is
                    return path
            return path
        except Exception:
            return path
    
    def to_dict(self, descriptor_name: Optional[str] = None) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding null fields.
        
        Args:
            descriptor_name: If provided and matches test_name, test_name will be omitted
        """
        result = {
            "status": self.status.value,
            "duration": self.duration,
            "cpu": self.cpu,
            "elf_path": self._make_path_relative(str(self.elf_path)),
            "timestamp": self.timestamp.isoformat(),
        }
        
        # Only include test_name if it's different from descriptor_name
        if descriptor_name is None or self.test_name != descriptor_name:
            result["test_name"] = self.test_name
        
        # Only include non-null optional fields
        if self.failure_reason is not None:
            result["failure_reason"] = self.failure_reason
        if self.skip_reason is not None:
            result["skip_reason"] = self.skip_reason
        if self.output_lines:
            result["output_lines"] = self.output_lines
        if self.memory_usage is not None:
            result["memory_usage"] = self.memory_usage
        if self.cycles is not None:
            result["cycles"] = self.cycles
        if self.exit_code is not None:
            result["exit_code"] = self.exit_code
        if self.error_type is not None:
            result["error_type"] = self.error_type
        if self.descriptor_name is not None and (descriptor_name is None or self.descriptor_name != descriptor_name):
            result["descriptor_name"] = self.descriptor_name
        if self.expected_output is not None:
            result["expected_output"] = self.expected_output
        if self.actual_output is not None:
            result["actual_output"] = self.actual_output
        if self.output_differences:
            result["output_differences"] = self.output_differences
        
        return result


@dataclass
class DescriptorResult:
    """Result for a single descriptor."""
    descriptor_name: str
    descriptor_path: Path
    descriptor_content: Dict[str, Any]  
    status: TestStatus
    test_result: Optional[TestResult] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    project_root: Optional[Path] = None  # For making paths relative
    
    def _make_path_relative(self, path: Path) -> str:
        """Convert absolute path to relative path if project_root is set."""
        if not self.project_root:
            return str(path)
        try:
            if path.is_absolute():
                try:
                    return str(path.relative_to(self.project_root))
                except ValueError:
                    # Path is not under project_root, return as-is
                    return str(path)
            return str(path)
        except Exception:
            return str(path)
    
    def to_dict(self, include_descriptor_name: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Args:
            include_descriptor_name: If False, omit descriptor_name (it's the key in parent dict)
        """
        # Clean descriptor_content by removing redundant 'name' field if it matches descriptor_name
        descriptor_content = self.descriptor_content.copy()
        if not include_descriptor_name and 'name' in descriptor_content:
            if descriptor_content['name'] == self.descriptor_name:
                descriptor_content = {k: v for k, v in descriptor_content.items() if k != 'name'}
        
        result = {
            "descriptor_path": self._make_path_relative(self.descriptor_path),
            "descriptor_content": descriptor_content,
        }
        
        # Only include descriptor_name if explicitly requested (it\'s redundant with the key)
        if include_descriptor_name:
            result["descriptor_name"] = self.descriptor_name
        
        # Include test_result if present, passing descriptor_name to remove duplicate test_name
        if self.test_result:
            result["test_result"] = self.test_result.to_dict(descriptor_name=self.descriptor_name)
        
        # Only include failure_stage if not None
        if self.failure_stage is not None:
            result["failure_stage"] = self.failure_stage
        
        # Only include failure_reason if it's different from test_result.failure_reason
        # or if there's no test_result
        if self.failure_reason is not None:
            if not self.test_result or self.failure_reason != self.test_result.failure_reason:
                result["failure_reason"] = self.failure_reason
        
        return result


@dataclass
class TestReport:
    """Complete test run report - descriptor-centric."""
    run_id: str
    start_time: datetime
    end_time: datetime
    cpu: str
    descriptor_results: Dict[str, DescriptorResult] = field(default_factory=dict)
    all_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    duration: float = 0.0
    project_root: Optional[Path] = None  # For making paths relative
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    timed_out: int = 0
    errors: int = 0
    not_run: int = 0
    build_failed: int = 0
    generation_failed: int = 0
    conversion_failed: int = 0
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.duration = (self.end_time - self.start_time).total_seconds()
        self._calculate_counts()
        self._generate_summary()
    
    def _calculate_counts(self):
        """Calculate counts from descriptor_results."""
        self.total_tests = len(self.descriptor_results)
        
        for desc_result in self.descriptor_results.values():
            status = desc_result.status
            if status == TestStatus.PASS:
                self.passed += 1
            elif status == TestStatus.FAIL:
                self.failed += 1
            elif status == TestStatus.SKIP:
                self.skipped += 1
            elif status == TestStatus.TIMEOUT:
                self.timed_out += 1
            elif status == TestStatus.ERROR:
                self.errors += 1
            elif status == TestStatus.NOT_RUN:
                self.not_run += 1
            elif status == TestStatus.BUILD_FAILED:
                self.build_failed += 1
            elif status == TestStatus.GENERATION_FAILED:
                self.generation_failed += 1
            elif status == TestStatus.CONVERSION_FAILED:
                self.conversion_failed += 1
    
    def _generate_summary(self):
        """Generate human-readable summary."""
        total = self.total_tests
        if total == 0:
            self.summary = "No descriptors found"
            return
            
        parts = []
        if self.passed > 0:
            pass_rate = (self.passed / total) * 100
            parts.append(f"{self.passed} passed ({pass_rate:.1f}%)")
        if self.failed > 0:
            fail_rate = (self.failed / total) * 100
            parts.append(f"{self.failed} failed ({fail_rate:.1f}%)")
        if self.skipped > 0:
            skip_rate = (self.skipped / total) * 100
            parts.append(f"{self.skipped} skipped ({skip_rate:.1f}%)")
        if self.timed_out > 0:
            timeout_rate = (self.timed_out / total) * 100
            parts.append(f"{self.timed_out} timed out ({timeout_rate:.1f}%)")
        if self.errors > 0:
            error_rate = (self.errors / total) * 100
            parts.append(f"{self.errors} errors ({error_rate:.1f}%)")
        if self.not_run > 0:
            not_run_rate = (self.not_run / total) * 100
            parts.append(f"{self.not_run} not run ({not_run_rate:.1f}%)")
        if self.build_failed > 0:
            build_fail_rate = (self.build_failed / total) * 100
            parts.append(f"{self.build_failed} build failed ({build_fail_rate:.1f}%)")
        if self.generation_failed > 0:
            gen_fail_rate = (self.generation_failed / total) * 100
            parts.append(f"{self.generation_failed} generation failed ({gen_fail_rate:.1f}%)")
        if self.conversion_failed > 0:
            conv_fail_rate = (self.conversion_failed / total) * 100
            parts.append(f"{self.conversion_failed} conversion failed ({conv_fail_rate:.1f}%)")
        
        self.summary = f"Descriptors: {total} total, " + ", ".join(parts)
    
    def get_status_counts(self) -> Dict[str, int]:
        """Get count of descriptors by status."""
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "errors": self.errors,
            "not_run": self.not_run,
            "build_failed": self.build_failed,
            "generation_failed": self.generation_failed,
            "conversion_failed": self.conversion_failed
        }
    
    def get_failed_tests(self) -> List[DescriptorResult]:
        """Get list of failed descriptors."""
        return [dr for dr in self.descriptor_results.values() 
                if dr.status in [TestStatus.FAIL, TestStatus.ERROR, TestStatus.TIMEOUT, 
                                TestStatus.BUILD_FAILED, TestStatus.GENERATION_FAILED, 
                                TestStatus.CONVERSION_FAILED]]
    
    def get_passed_tests(self) -> List[DescriptorResult]:
        """Get list of passed descriptors."""
        return [dr for dr in self.descriptor_results.values() 
                if dr.status == TestStatus.PASS]
    
    def get_skipped_tests(self) -> List[DescriptorResult]:
        """Get list of skipped descriptors."""
        return [dr for dr in self.descriptor_results.values() 
                if dr.status == TestStatus.SKIP]
    
    def get_test_results(self) -> List[TestResult]:
        """Get all test results that were executed."""
        return [dr.test_result for dr in self.descriptor_results.values() 
                if dr.test_result is not None]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, excluding null fields.
        """
        result = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "cpu": self.cpu,
            "metadata": self.metadata,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "errors": self.errors,
            "not_run": self.not_run,
            "build_failed": self.build_failed,
            "generation_failed": self.generation_failed,
            "conversion_failed": self.conversion_failed,
            "descriptor_results": {name: dr.to_dict(include_descriptor_name=False) for name, dr in self.descriptor_results.items()},
            "summary": self.summary,
            "duration": self.duration
        }
        
        # Removed all_descriptors as it's redundant (descriptor content is in descriptor_results)
        
        return result
