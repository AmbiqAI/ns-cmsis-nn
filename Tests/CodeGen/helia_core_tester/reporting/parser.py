"""
Test result parser for Unity framework output.
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from helia_core_tester.reporting.models import TestResult, TestStatus


class TestResultParser:
    """Parser for Unity test framework output."""
    
    def __init__(self):
        self.test_start_pattern = re.compile(r'Running test: (.+)')
        self.test_pass_pattern = re.compile(r'(.+):test__([^:]+):PASS')
        self.test_fail_pattern = re.compile(r'(.+):test__([^:]+):FAIL')
        self.assertion_pattern = re.compile(r'TEST_ASSERT_EQUAL.*?MESSAGE\([^,]+,\s*[^,]+,\s*"([^"]+)"\)')
        self.failure_reason_pattern = re.compile(r'Expected\s+(\S+)\s+Was\s+(\S+)')
        self.timeout_pattern = re.compile(r'TIMEOUT running (.+)')
        self.error_pattern = re.compile(r'ERROR:\s*(.+)')
        self.cycles_pattern = re.compile(r'\[PERF\]\s+(\w+):\s+(\d+)\s+cycles')
        self.memory_pattern = re.compile(r'Memory usage:\s+(\d+)\s+bytes')
        # Patterns for extracting output differences
        self.expected_pattern = re.compile(r'(?:Expected|Golden|Reference)[:\s]+([^\n]+)', re.IGNORECASE)
        self.actual_pattern = re.compile(r'(?:Actual|Got|Output|Result)[:\s]+([^\n]+)', re.IGNORECASE)
        self.difference_pattern = re.compile(r'(?:Difference|Delta|Diff)[:\s]+([^\n]+)', re.IGNORECASE)
        self.index_pattern = re.compile(r'(?:Index|Position|Element)\s*[\[\(]?\s*(\d+)\s*[\]\)]?', re.IGNORECASE)
        self.value_comparison_pattern = re.compile(r'(\d+)\s*[!=<>]+\s*(\d+)')
        # Pattern for exact "0 Failures" match (not substring)
        self.zero_failures_pattern = re.compile(r'^0\s+Failures\s*$', re.MULTILINE | re.IGNORECASE)
        # Pattern for "X Failures" where X > 0
        self.nonzero_failures_pattern = re.compile(r'^(\d+)\s+Failures\s*$', re.MULTILINE | re.IGNORECASE)
        # Pattern for "Convolution failed" or API errors
        self.api_error_pattern = re.compile(r'Convolution failed with status\s+(-?\d+)', re.IGNORECASE)
        
    def parse_fvp_output(self, 
                        output: str, 
                        elf_path: Path, 
                        cpu: str, 
                        duration: float,
                        exit_code: Optional[int] = None,
                        descriptor_name: Optional[str] = None) -> TestResult:
        """
        Parse FVP output to extract test result.
        
        Args:
            output: Raw FVP output
            elf_path: Path to the ELF file
            cpu: Target CPU
            duration: Test execution duration in seconds
            exit_code: Process exit code
            descriptor_name: Optional descriptor name to link test to descriptor
            
        Returns:
            TestResult object
        """
        lines = output.split('\n')
        test_name = self._extract_test_name(elf_path)
        
        if descriptor_name is None:
            descriptor_name = test_name
        
        status, failure_reason, skip_reason, error_type = self._determine_status(
            output, lines, exit_code
        )
        
        cycles = self._extract_cycles(output)
        memory_usage = self._extract_memory_usage(output)
        
        relevant_lines = self._extract_relevant_lines(lines)
        
        # Extract output differences if test failed
        expected_output = None
        actual_output = None
        output_differences = []
        if status == TestStatus.FAIL:
            expected_output, actual_output, output_differences = self._extract_output_differences(output, lines)
        
        return TestResult(
            test_name=test_name,
            status=status,
            duration=duration,
            cpu=cpu,
            elf_path=elf_path,
            failure_reason=failure_reason,
            skip_reason=skip_reason,
            output_lines=relevant_lines,
            timestamp=datetime.now(),
            memory_usage=memory_usage,
            cycles=cycles,
            exit_code=exit_code,
            error_type=error_type,
            descriptor_name=descriptor_name,
            expected_output=expected_output,
            actual_output=actual_output,
            output_differences=output_differences
        )
    
    def _extract_test_name(self, elf_path: Path) -> str:
        """Extract test name from ELF file path."""
        return elf_path.stem
    
    def _determine_status(self, 
                         output: str, 
                         lines: List[str], 
                         exit_code: Optional[int]) -> Tuple[TestStatus, Optional[str], Optional[str], Optional[str]]:
        """Determine test status and extract failure/skip reasons."""
        
        if exit_code == 124 or "TIMEOUT" in output:
            return TestStatus.TIMEOUT, "Test execution timed out", None, "timeout"
        
        pass_matches = self.test_pass_pattern.findall(output)
        fail_matches = self.test_fail_pattern.findall(output)
        
        if fail_matches:
            failure_reason = self._extract_failure_reason(output, lines)
            return TestStatus.FAIL, failure_reason, None, "assertion"
        elif pass_matches:
            return TestStatus.PASS, None, None, None
        elif exit_code and exit_code != 0:
            error_msg = f"Process exited with code {exit_code}"
            return TestStatus.ERROR, error_msg, None, "crash"
        else:
            # Check for API errors first (e.g., "Convolution failed with status -1")
            api_error_match = self.api_error_pattern.search(output)
            if api_error_match:
                status_code = api_error_match.group(1)
                return TestStatus.FAIL, f"Convolution API error (status {status_code})", None, "api_error"
            
            # Check for exact "0 Failures" match (not substring)
            if self.zero_failures_pattern.search(output):
                return TestStatus.PASS, None, None, None
            
            # Check for "X Failures" where X > 0 (output mismatch)
            nonzero_match = self.nonzero_failures_pattern.search(output)
            if nonzero_match:
                failure_count = int(nonzero_match.group(1))
                return TestStatus.FAIL, f"Output mismatch: {failure_count} element(s) differ from expected", None, "output_mismatch"
            
            # If no recognizable pattern, mark as error
            return TestStatus.ERROR, "Unknown test status", None, "unknown"
    
    def _extract_failure_reason(self, output: str, lines: List[str]) -> str:
        """Extract detailed failure reason from Unity output."""
        assertion_matches = self.assertion_pattern.findall(output)
        if assertion_matches:
            return assertion_matches[-1]  # Get the last assertion message
        
        failure_matches = self.failure_reason_pattern.findall(output)
        if failure_matches:
            expected, actual = failure_matches[-1]
            return f"Expected {expected} but got {actual}"
        
        error_matches = self.error_pattern.findall(output)
        if error_matches:
            return error_matches[-1]
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['fail', 'error', 'assert']):
                return line.strip()
        
        return "Test failed (no specific reason found)"
    
    def _extract_cycles(self, output: str) -> Optional[int]:
        """Extract cycle count from performance output."""
        cycles_matches = self.cycles_pattern.findall(output)
        if cycles_matches:
            try:
                return int(cycles_matches[-1][1])
            except (ValueError, IndexError):
                pass
        return None
    
    def _extract_memory_usage(self, output: str) -> Optional[int]:
        """Extract memory usage from output."""
        memory_matches = self.memory_pattern.findall(output)
        if memory_matches:
            try:
                return int(memory_matches[-1])
            except (ValueError, IndexError):
                pass
        return None
    
    def _extract_relevant_lines(self, lines: List[str]) -> List[str]:
        """Extract relevant output lines for debugging."""
        relevant = []
        in_test_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.lower() for keyword in ['test', 'fail', 'pass', 'error', 'assert']):
                in_test_section = True
            
            if in_test_section:
                relevant.append(line)
                
                if len(relevant) > 50:
                    relevant.append("... (truncated)")
                    break
        
        return relevant
    
    def _extract_output_differences(self, output: str, lines: List[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Extract expected vs actual output differences from test output.
        
        Returns:
            Tuple of (expected_output, actual_output, differences_list)
        """
        expected = None
        actual = None
        differences = []
        
        # Look for explicit "Expected" and "Actual" patterns
        expected_matches = self.expected_pattern.findall(output)
        actual_matches = self.actual_pattern.findall(output)
        
        if expected_matches:
            expected = expected_matches[-1].strip()
        if actual_matches:
            actual = actual_matches[-1].strip()
        
        # Look for "Expected X Was Y" patterns (already handled by failure_reason_pattern)
        failure_matches = self.failure_reason_pattern.findall(output)
        if failure_matches:
            for exp, act in failure_matches:
                differences.append(f"Expected: {exp}, Actual: {act}")
        
        # Look for index-based differences (e.g., "Index [0]: Expected 5, Got 3")
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['expected', 'actual', 'got', 'golden', 'reference', 'difference']):
                # Check if this line contains value comparisons
                if self.value_comparison_pattern.search(line):
                    differences.append(line.strip())
                # Check if this line contains index information
                elif self.index_pattern.search(line):
                    differences.append(line.strip())
                # Check if this line contains difference information
                elif self.difference_pattern.search(line):
                    diff_match = self.difference_pattern.search(line)
                    if diff_match:
                        differences.append(f"Difference: {diff_match.group(1).strip()}")
        
        # Look for array/tensor output differences
        # Common patterns: "Output[0] = X (expected Y)" or "Element N: got X, expected Y"
        array_diff_pattern = re.compile(r'(?:Output|Element|Value|Index)\s*[\[\(]?\s*(\d+)\s*[\]\)]?\s*[:=]?\s*(?:got|was|actual)?\s*(\d+)\s*(?:expected|should be|golden)?\s*(\d+)', re.IGNORECASE)
        array_diffs = array_diff_pattern.findall(output)
        for idx, got, exp in array_diffs:
            differences.append(f"Index [{idx}]: Got {got}, Expected {exp}")
        
        # Limit differences to prevent overwhelming output
        if len(differences) > 50:
            differences = differences[:50]
            differences.append("... (more differences truncated)")
        
        return expected, actual, differences
    
    def parse_multiple_tests(self, 
                           outputs: List[Tuple[str, Path, str, float, Optional[int]]]) -> List[TestResult]:
        """
        Parse multiple test outputs.
        
        Args:
            outputs: List of tuples (output, elf_path, cpu, duration, exit_code)
            
        Returns:
            List of TestResult objects
        """
        results = []
        for output, elf_path, cpu, duration, exit_code in outputs:
            result = self.parse_fvp_output(output, elf_path, cpu, duration, exit_code)
            results.append(result)
        return results
