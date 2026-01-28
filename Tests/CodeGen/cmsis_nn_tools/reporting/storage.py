"""
Report storage and retrieval functionality.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import TestReport, TestResult, TestStatus, DescriptorResult


class ReportStorage:
    """Handle storage and retrieval of test reports."""
    
    def __init__(self, reports_dir: Path = Path("reports")):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
    
    def save_report(self, report: TestReport) -> Path:
        """
        Save a test report to disk.
        
        Args:
            report: TestReport to save
            
        Returns:
            Path to saved report file
        """
        timestamp = report.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{report.cpu}_{timestamp}.json"
        file_path = self.reports_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return file_path
    
    def load_report(self, file_path: Path) -> TestReport:
        """
        Load a test report from disk.
        
        Args:
            file_path: Path to report file
            
        Returns:
            TestReport object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_report(data)
    
    def list_reports(self, cpu: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List available reports.
        
        Args:
            cpu: Filter by CPU type
            limit: Maximum number of reports to return
            
        Returns:
            List of report metadata dictionaries
        """
        reports = []
        
        for file_path in sorted(self.reports_dir.glob("test_report_*.json"), reverse=True):
            if limit > 0 and len(reports) >= limit:
                break
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if cpu and data.get('cpu') != cpu:
                    continue
                
                reports.append({
                    'file_path': file_path,
                    'cpu': data.get('cpu', 'unknown'),
                    'start_time': data.get('start_time'),
                    'duration': data.get('duration', 0),
                    'total_tests': data.get('total_tests', 0),
                    'passed': data.get('passed', 0),
                    'failed': data.get('failed', 0),
                    'skipped': data.get('skipped', 0)
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        return reports
    
    def get_latest_report(self, cpu: Optional[str] = None) -> Optional[TestReport]:
        """
        Get the most recent report.
        
        Args:
            cpu: Filter by CPU type
            
        Returns:
            Latest TestReport or None if no reports found
        """
        reports = self.list_reports(cpu=cpu, limit=1)
        if not reports:
            return None
        
        return self.load_report(reports[0]['file_path'])
    
    def get_report_summary(self, cpu: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """
        Get summary statistics for recent reports.
        
        Args:
            cpu: Filter by CPU type
            days: Number of days to look back
            
        Returns:
            Summary statistics dictionary
        """
        reports = self.list_reports(cpu=cpu, limit=100)
        
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_reports = []
        
        for report in reports:
            try:
                report_time = datetime.fromisoformat(report['start_time'].replace('Z', '+00:00'))
                if report_time.timestamp() >= cutoff_date:
                    recent_reports.append(report)
            except (ValueError, TypeError):
                continue
        
        if not recent_reports:
            return {
                'total_runs': 0,
                'avg_duration': 0,
                'avg_pass_rate': 0,
                'total_tests': 0,
                'trend': 'no_data'
            }
        
        total_runs = len(recent_reports)
        total_duration = sum(r['duration'] for r in recent_reports)
        avg_duration = total_duration / total_runs if total_runs > 0 else 0
        
        total_tests = sum(r['total_tests'] for r in recent_reports)
        total_passed = sum(r['passed'] for r in recent_reports)
        avg_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        trend = 'stable'
        if len(recent_reports) >= 4:
            mid_point = len(recent_reports) // 2
            first_half = recent_reports[mid_point:]
            second_half = recent_reports[:mid_point]
            
            first_half_pass_rate = sum(r['passed'] for r in first_half) / sum(r['total_tests'] for r in first_half) * 100
            second_half_pass_rate = sum(r['passed'] for r in second_half) / sum(r['total_tests'] for r in second_half) * 100
            
            if second_half_pass_rate > first_half_pass_rate + 5:
                trend = 'improving'
            elif second_half_pass_rate < first_half_pass_rate - 5:
                trend = 'declining'
        
        return {
            'total_runs': total_runs,
            'avg_duration': avg_duration,
            'avg_pass_rate': avg_pass_rate,
            'total_tests': total_tests,
            'trend': trend,
            'date_range': {
                'start': recent_reports[-1]['start_time'] if recent_reports else None,
                'end': recent_reports[0]['start_time'] if recent_reports else None
            }
        }
    
    def cleanup_old_reports(self, keep_days: int = 30) -> int:
        """
        Remove old report files.
        
        Args:
            keep_days: Number of days of reports to keep
            
        Returns:
            Number of files removed
        """
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
        removed_count = 0
        
        for file_path in self.reports_dir.glob("test_report_*.json"):
            try:
                if file_path.stat().st_mtime < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
            except OSError:
                continue
        
        return removed_count
    
    def _dict_to_report(self, data: Dict[str, Any]) -> TestReport:
        """Convert dictionary to TestReport object."""
        if 'descriptor_results' in data:
            descriptor_results = {}
            for desc_name, desc_data in data.get('descriptor_results', {}).items():
                test_result = None
                if desc_data.get('test_result'):
                    tr_data = desc_data['test_result']
                    test_result = TestResult(
                        test_name=tr_data['test_name'],
                        status=TestStatus(tr_data['status']),
                        duration=tr_data['duration'],
                        cpu=tr_data['cpu'],
                        elf_path=Path(tr_data['elf_path']),
                        failure_reason=tr_data.get('failure_reason'),
                        skip_reason=tr_data.get('skip_reason'),
                        output_lines=tr_data.get('output_lines', []),
                        timestamp=datetime.fromisoformat(tr_data['timestamp']),
                        memory_usage=tr_data.get('memory_usage'),
                        cycles=tr_data.get('cycles'),
                        exit_code=tr_data.get('exit_code'),
                        error_type=tr_data.get('error_type'),
                        descriptor_name=tr_data.get('descriptor_name')
                    )
                
                descriptor_results[desc_name] = DescriptorResult(
                    descriptor_name=desc_data['descriptor_name'],
                    descriptor_path=Path(desc_data['descriptor_path']),
                    descriptor_content=desc_data['descriptor_content'],
                    status=TestStatus(desc_data['status']),
                    test_result=test_result,
                    failure_stage=desc_data.get('failure_stage'),
                    failure_reason=desc_data.get('failure_reason')
                )
            
            report = TestReport(
                run_id=data['run_id'],
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']),
                cpu=data['cpu'],
                descriptor_results=descriptor_results,
                all_descriptors=data.get('all_descriptors', [])
            )
        else:
            results = []
            for result_data in data.get('results', []):
                result = TestResult(
                    test_name=result_data['test_name'],
                    status=TestStatus(result_data['status']),
                    duration=result_data['duration'],
                    cpu=result_data['cpu'],
                    elf_path=Path(result_data['elf_path']),
                    failure_reason=result_data.get('failure_reason'),
                    skip_reason=result_data.get('skip_reason'),
                    output_lines=result_data.get('output_lines', []),
                    timestamp=datetime.fromisoformat(result_data['timestamp']),
                    memory_usage=result_data.get('memory_usage'),
                    cycles=result_data.get('cycles'),
                    exit_code=result_data.get('exit_code'),
                    error_type=result_data.get('error_type'),
                    descriptor_name=result_data.get('descriptor_name')
                )
                results.append(result)

            descriptor_results = {}
            for result in results:
                desc_name = result.descriptor_name or result.test_name
                desc_content = {'name': desc_name}
                
                descriptor_results[desc_name] = DescriptorResult(
                    descriptor_name=desc_name,
                    descriptor_path=Path(f"descriptors/{desc_name}.yaml"),
                    descriptor_content=desc_content,
                    status=result.status,
                    test_result=result,
                    failure_stage="execution" if result.status != TestStatus.PASS else None,
                    failure_reason=result.failure_reason
                )
            report = TestReport(
                run_id=data['run_id'],
                start_time=datetime.fromisoformat(data['start_time']),
                end_time=datetime.fromisoformat(data['end_time']),
                cpu=data['cpu'],
                descriptor_results=descriptor_results,
                all_descriptors=list(descriptor_results.values())
            )
        
        return report
