"""
Report generator for multiple output formats.
"""

import json
import re
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

from helia_core_tester.reporting.models import TestReport, TestStatus


class ReportGenerator:
    """Generate test reports in multiple formats."""
    
    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _report_path(self, report: TestReport, extension: str) -> Path:
        """Return path for report file; ensure output dir exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{report.cpu}_{timestamp}.{extension}"
        return self.output_dir / filename

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#x27;"))
    
    def generate_reports(self, 
                        report: TestReport, 
                        formats: List[str] = None) -> Dict[str, Path]:
        """
        Generate reports in specified formats.
        
        Args:
            report: TestReport object
            formats: List of formats to generate (json, html, md)
            
        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = ["json", "html", "md"]
        
        generated_files = {}
        
        for format_type in formats:
            if format_type == "json":
                file_path = self._generate_json_report(report)
                generated_files["json"] = file_path
            elif format_type == "html":
                file_path = self._generate_html_report(report)
                generated_files["html"] = file_path
            elif format_type == "md":
                file_path = self._generate_markdown_report(report)
                generated_files["md"] = file_path
            elif format_type == "junit":
                file_path = self._generate_junit_report(report)
                generated_files["junit"] = file_path
        
        return generated_files
    
    def _generate_json_report(self, report: TestReport) -> Path:
        """Generate JSON report with compact array formatting."""
        file_path = self._report_path(report, "json")
        json_str = json.dumps(report.to_dict(), indent=2)
        
        # Post-process to format arrays on one line
        # Pattern for shape arrays (numbers only) - most common case
        # Matches: "field_name": [\n  1,\n  2,\n  3\n]
        json_str = re.sub(
            r'("(?:input|filter|strides|padding|shape|axis|num_tensors|input_\d+_shape|output_shape)[^"]*":\s*)\[\s*\n\s*((?:\d+\s*,\s*\n\s*)*\d+)\s*\n\s*\]',
            lambda m: m.group(1) + '[' + re.sub(r'\s+', ' ', m.group(2).strip()) + ']',
            json_str,
            flags=re.MULTILINE
        )
        
        # General pattern for any array with numbers only (fallback)
        json_str = re.sub(
            r'(\s+)"([^"]+)":\s*\[\s*\n\s*((?:\d+\s*,\s*\n\s*)*\d+)\s*\n\s*\]',
            lambda m: m.group(1) + f'"{m.group(2)}": [' + re.sub(r'\s+', ' ', m.group(3).strip()) + ']',
            json_str,
            flags=re.MULTILINE
        )
        
        # Add blank line before "descriptor_results"
        json_str = re.sub(
            r'(\n\s*)"descriptor_results":\s*\{',
            r'\1\n\1"descriptor_results": {',
            json_str
        )
        
        # Add blank lines between descriptor entries
        # Match: closing brace of a descriptor, comma, newline, then next descriptor key
        json_str = re.sub(
            r'(\s+)\},\s*\n(\s+)"([^"]+)":\s*\{',
            r'\1},\n\n\2"\3": {',
            json_str
        )
        
        with open(file_path, 'w') as f:
            f.write(json_str)
        
        return file_path
    
    def _generate_html_report(self, report: TestReport) -> Path:
        """Generate HTML report."""
        file_path = self._report_path(report, "html")
        html_content = self._create_html_content(report)
        
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        return file_path
    
    def _generate_markdown_report(self, report: TestReport) -> Path:
        """Generate Markdown report."""
        file_path = self._report_path(report, "md")
        md_content = self._create_markdown_content(report)
        
        with open(file_path, 'w') as f:
            f.write(md_content)
        
        return file_path

    def _generate_junit_report(self, report: TestReport) -> Path:
        """Generate JUnit XML report."""
        file_path = self.output_dir / "junit.xml"

        testsuite = ET.Element(
            "testsuite",
            {
                "name": f"helia_core_tester_{report.cpu}",
                "tests": str(report.total_tests),
                "failures": str(report.failed + report.build_failed + report.generation_failed + report.conversion_failed),
                "errors": str(report.errors + report.timed_out),
                "skipped": str(report.skipped + report.not_run),
                "time": f"{report.duration:.2f}",
            },
        )

        if report.metadata:
            props = ET.SubElement(testsuite, "properties")
            for key, value in report.metadata.items():
                if value is None:
                    continue
                ET.SubElement(props, "property", {"name": str(key), "value": str(value)})

        for desc_name, desc_result in report.descriptor_results.items():
            duration = desc_result.test_result.duration if desc_result.test_result else 0.0
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {
                    "classname": "helia_core_tester",
                    "name": desc_name,
                    "time": f"{duration:.2f}",
                },
            )

            status = desc_result.status
            reason = desc_result.failure_reason or ""
            if status == TestStatus.SKIP or status == TestStatus.NOT_RUN:
                skipped = ET.SubElement(testcase, "skipped")
                if reason:
                    skipped.set("message", reason)
            elif status in (
                TestStatus.FAIL,
                TestStatus.BUILD_FAILED,
                TestStatus.GENERATION_FAILED,
                TestStatus.CONVERSION_FAILED,
            ):
                failure = ET.SubElement(testcase, "failure", {"message": reason or status.value})
                if reason:
                    failure.text = reason
            elif status in (TestStatus.ERROR, TestStatus.TIMEOUT):
                err = ET.SubElement(testcase, "error", {"message": reason or status.value})
                if reason:
                    err.text = reason

        tree = ET.ElementTree(testsuite)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        return file_path
    
    def _create_html_content(self, report: TestReport) -> str:
        """Create HTML content for the report - descriptor-centric."""
        status_counts = report.get_status_counts()
        
        total = report.total_tests
        pass_rate = (status_counts["passed"] / total * 100) if total > 0 else 0
        fail_rate = (status_counts["failed"] / total * 100) if total > 0 else 0
        
        status_cards = []
        for status_name, count in status_counts.items():
            if count > 0:
                rate = (count / total * 100) if total > 0 else 0
                status_class = status_name.replace("_", "-")
                status_cards.append(f"""
        <div class="summary-card {status_class}">
            <h3>{count}</h3>
            <p>{status_name.replace('_', ' ').title()}<br>({rate:.1f}%)</p>
        </div>""")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Helia-Core Tester Report - {report.cpu}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .summary-card {{ 
            background-color: #e8f4fd; 
            padding: 15px; 
            border-radius: 5px; 
            text-align: center;
            min-width: 120px;
        }}
        .passed {{ background-color: #d4edda; }}
        .failed {{ background-color: #f8d7da; }}
        .skipped {{ background-color: #fff3cd; }}
        .not-run {{ background-color: #e9ecef; }}
        .build-failed {{ background-color: #f8d7da; }}
        .generation-failed {{ background-color: #f8d7da; }}
        .conversion-failed {{ background-color: #f8d7da; }}
        .timeout {{ background-color: #e7d3ff; }}
        .error {{ background-color: #f8d7da; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .results-table th, .results-table td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }}
        .results-table th {{ background-color: #f2f2f2; }}
        .status-pass {{ color: green; font-weight: bold; }}
        .status-fail {{ color: red; font-weight: bold; }}
        .status-skip {{ color: orange; font-weight: bold; }}
        .status-timeout {{ color: purple; font-weight: bold; }}
        .status-error {{ color: darkred; font-weight: bold; }}
        .status-not-run {{ color: gray; font-weight: bold; }}
        .status-build-failed {{ color: red; font-weight: bold; }}
        .status-generation-failed {{ color: red; font-weight: bold; }}
        .status-conversion-failed {{ color: red; font-weight: bold; }}
        .descriptor-details {{ 
            background-color: #f8f9fa; 
            padding: 10px; 
            border-left: 4px solid #007bff;
            margin: 5px 0;
        }}
        .test-details {{ 
            background-color: #fff3cd; 
            padding: 10px; 
            border-left: 4px solid #ffc107;
            margin: 5px 0;
        }}
        .failure-details {{ 
            background-color: #f8d7da; 
            padding: 10px; 
            border-left: 4px solid #dc3545;
            margin: 5px 0;
        }}
        .expandable {{ cursor: pointer; background-color: #007bff; color: white; border: none; padding: 5px 10px; border-radius: 3px; }}
        .expandable:hover {{ background-color: #0056b3; }}
        .hidden {{ display: none; }}
        .yaml-content {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; font-family: monospace; white-space: pre-wrap; }}
    </style>
    <script>
        function toggleDetails(descName, type) {{
            var details = document.getElementById(type + '-' + descName);
            details.classList.toggle('hidden');
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>Helia-Core Tester Report</h1>
        <p><strong>CPU:</strong> {report.cpu}</p>
        <p><strong>Run ID:</strong> {report.run_id}</p>
        <p><strong>Start Time:</strong> {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {report.duration:.2f} seconds</p>
        <p><strong>Summary:</strong> {report.summary}</p>
        {"".join([f"<p><strong>{k}:</strong> {v}</p>" for k, v in (report.metadata or {}).items() if v is not None])}
    </div>
    
    <div class="summary">
{''.join(status_cards)}
    </div>
    
    <h2>Descriptor Results</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>Descriptor Name</th>
                <th>Operator</th>
                <th>DTypes</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Failure Reason</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
"""
        
        sorted_descriptors = sorted(report.descriptor_results.items())
        
        for desc_name, desc_result in sorted_descriptors:
            desc_content = desc_result.descriptor_content
            operator = desc_content.get('operator', 'N/A')
            activation_dtype = desc_content.get('activation_dtype', 'N/A')
            weight_dtype = desc_content.get('weight_dtype', 'N/A')
            dtypes = f"{activation_dtype}/{weight_dtype}"
            
            status_class = f"status-{desc_result.status.value.lower().replace('_', '-')}"
            duration = desc_result.test_result.duration if desc_result.test_result else 0.0
            failure_reason = desc_result.failure_reason or ""
            
            safe_desc_name = desc_name.replace("'", "\\'").replace('"', '\\"')
            
            test_button = ""
            if desc_result.test_result:
                test_button = f'<button class="expandable" onclick="toggleDetails(\'{safe_desc_name}\', \'test\')">Show Test</button>'
            
            html += f"""
            <tr>
                <td><strong>{desc_name}</strong></td>
                <td>{operator}</td>
                <td>{dtypes}</td>
                <td class="{status_class}">{desc_result.status.value}</td>
                <td>{duration:.2f}</td>
                <td>{failure_reason[:50]}{'...' if len(failure_reason) > 50 else ''}</td>
                <td>
                    <button class="expandable" onclick="toggleDetails('{safe_desc_name}', 'desc')">
                        Show Descriptor
                    </button>
                    {test_button}
                </td>
            </tr>
            <tr id="desc-{safe_desc_name}" class="hidden">
                <td colspan="7">
                    <div class="descriptor-details">
                        <h4>Descriptor Content (YAML)</h4>
                        <div class="yaml-content">{self._escape_html(yaml.dump(desc_content, default_flow_style=False))}</div>
                        <p><strong>Descriptor Path:</strong> {desc_result.descriptor_path}</p>
                    </div>
                </td>
            </tr>"""
            
            if desc_result.test_result:
                test_result = desc_result.test_result
                html += f"""
            <tr id="test-{safe_desc_name}" class="hidden">
                <td colspan="7">
                    <div class="test-details">
                        <h4>Test Result Details</h4>
                        <p><strong>Test Name:</strong> {test_result.test_name}</p>
                        <p><strong>Status:</strong> {test_result.status.value}</p>
                        <p><strong>Duration:</strong> {test_result.duration:.2f} seconds</p>
                        <p><strong>ELF Path:</strong> {test_result.elf_path}</p>
                        <p><strong>Timestamp:</strong> {test_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>"""
            
                if test_result.cycles:
                    html += f"<p><strong>Cycles:</strong> {test_result.cycles:,}</p>"
                if test_result.memory_usage:
                    html += f"<p><strong>Memory Usage:</strong> {test_result.memory_usage:,} bytes</p>"
                if test_result.exit_code is not None:
                    html += f"<p><strong>Exit Code:</strong> {test_result.exit_code}</p>"
                if test_result.failure_reason:
                    html += f"<p><strong>Failure Reason:</strong> {test_result.failure_reason}</p>"
                if test_result.output_lines:
                    html += "<p><strong>Output:</strong></p><pre>"
                    for line in test_result.output_lines[:20]:
                        html += f"{self._escape_html(line)}\n"
                    if len(test_result.output_lines) > 20:
                        html += "... (truncated)"
                html += "</pre>"
            
            html += """
                    </div>
                </td>
            </tr>"""
        
        html += """
        </tbody>
    </table>
</body>
</html>
"""
        return html
    
    def _create_markdown_content(self, report: TestReport) -> str:
        """Create Markdown content for the report - descriptor-centric."""
        status_counts = report.get_status_counts()
        
        total = report.total_tests

        metadata_lines = "".join(
            f"- **{k}:** {v}\n" for k, v in (report.metadata or {}).items() if v is not None
        )
        
        md = f"""# Helia-Core Tester Report

## Summary

- **CPU:** {report.cpu}
- **Run ID:** {report.run_id}
- **Start Time:** {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration:** {report.duration:.2f} seconds
- **Total Descriptors:** {report.total_tests}
{metadata_lines}

## Results Overview

| Status | Count | Percentage |
|--------|-------|------------|
"""
        
        for status_name, count in status_counts.items():
            if count > 0:
                rate = (count / total * 100) if total > 0 else 0
                status_display = status_name.replace('_', ' ').title()
                md += f"| {status_display} | {count} | {rate:.1f}% |\n"
        
        md += "\n## Descriptor Results\n\n"
        
        sorted_descriptors = sorted(report.descriptor_results.items())
        
        for desc_name, desc_result in sorted_descriptors:
            desc_content = desc_result.descriptor_content
            operator = desc_content.get('operator', 'N/A')
            activation_dtype = desc_content.get('activation_dtype', 'N/A')
            weight_dtype = desc_content.get('weight_dtype', 'N/A')
            
            md += f"### {desc_name}\n\n"
            md += f"- **Status:** {desc_result.status.value}\n"
            md += f"- **Operator:** {operator}\n"
            md += f"- **Activation DType:** {activation_dtype}\n"
            md += f"- **Weight DType:** {weight_dtype}\n"
        
            if desc_result.failure_stage:
                md += f"- **Failure Stage:** {desc_result.failure_stage}\n"
            if desc_result.failure_reason:
                md += f"- **Failure Reason:** {desc_result.failure_reason}\n"
            
            md += f"- **Descriptor Path:** `{desc_result.descriptor_path}`\n\n"
        
            md += "**Descriptor Content (YAML):**\n```yaml\n"
            md += yaml.dump(desc_content, default_flow_style=False)
            md += "```\n\n"
            
            if desc_result.test_result:
                test_result = desc_result.test_result
                md += "**Test Result:**\n\n"
                md += f"- **Test Name:** {test_result.test_name}\n"
                md += f"- **Status:** {test_result.status.value}\n"
                md += f"- **Duration:** {test_result.duration:.2f} seconds\n"
                md += f"- **ELF Path:** `{test_result.elf_path}`\n"
                md += f"- **Timestamp:** {test_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                if test_result.cycles:
                    md += f"- **Cycles:** {test_result.cycles:,}\n"
                if test_result.memory_usage:
                    md += f"- **Memory Usage:** {test_result.memory_usage:,} bytes\n"
                if test_result.exit_code is not None:
                    md += f"- **Exit Code:** {test_result.exit_code}\n"
                if test_result.failure_reason:
                    md += f"- **Failure Reason:** {test_result.failure_reason}\n"
                
                if test_result.output_lines:
                    md += "\n**Test Output:**\n```\n"
                    for line in test_result.output_lines[:30]:
                        md += f"{line}\n"
                    if len(test_result.output_lines) > 30:
                        md += "... (truncated)\n"
                    md += "```\n"
            else:
                md += "**Test Result:** Not executed\n"
            
            md += "\n---\n\n"
        
        return md
