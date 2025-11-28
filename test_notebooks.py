#!/usr/bin/env python3
"""
Automated test script to execute Jupyter notebooks and verify successful execution.
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime

class NotebookTester:
    def __init__(self, notebook_dir="notebooks"):
        self.notebook_dir = Path(notebook_dir)
        self.results = []
        
    def execute_notebook(self, notebook_path):
        """Execute a Jupyter notebook and return execution status."""
        print(f"\n{'='*70}")
        print(f"Executing: {notebook_path.name}")
        print(f"{'='*70}")
        
        # Use absolute paths
        notebook_path = notebook_path.absolute()
        output_name = f"{notebook_path.stem}_executed{notebook_path.suffix}"
        output_path = notebook_path.parent / output_name
        
        try:
            # Execute notebook using nbconvert
            cmd = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=600",  # 10 minute timeout per cell
                "--output", output_name,  # Just the filename
                str(notebook_path)  # Full path to input notebook
            ]
            
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(notebook_path.parent)  # Run in notebook directory
            )
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            success = result.returncode == 0
            
            # Parse executed notebook to count cells
            cell_count = 0
            if output_path.exists():
                with open(output_path, 'r') as f:
                    nb_data = json.load(f)
                    cell_count = len([c for c in nb_data.get('cells', []) 
                                     if c.get('cell_type') == 'code'])
            
            test_result = {
                'notebook': notebook_path.name,
                'success': success,
                'execution_time': execution_time,
                'cell_count': cell_count,
                'output_path': str(output_path) if success else None,
                'error': result.stderr if not success else None
            }
            
            self.results.append(test_result)
            
            if success:
                print(f"✅ SUCCESS: Executed {cell_count} cells in {execution_time:.2f}s")
                print(f"   Output saved to: {output_path}")
            else:
                print(f"❌ FAILED: Execution failed after {execution_time:.2f}s")
                print(f"   Error: {result.stderr[:200]}")
            
            return test_result
            
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            test_result = {
                'notebook': notebook_path.name,
                'success': False,
                'execution_time': 0,
                'cell_count': 0,
                'output_path': None,
                'error': str(e)
            }
            self.results.append(test_result)
            return test_result
    
    def run_all_tests(self):
        """Execute all notebooks in the notebook directory."""
        print("\n" + "="*70)
        print("JUPYTER NOTEBOOK AUTOMATED TEST SUITE")
        print("="*70)
        print(f"Notebook directory: {self.notebook_dir.absolute()}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Find all notebooks
        notebooks = sorted(self.notebook_dir.glob("*.ipynb"))
        notebooks = [nb for nb in notebooks if not nb.name.endswith("_executed.ipynb")]
        
        if not notebooks:
            print("\n⚠️  No notebooks found!")
            return False
        
        print(f"\nFound {len(notebooks)} notebook(s) to test:")
        for nb in notebooks:
            print(f"  - {nb.name}")
        
        # Execute each notebook
        for notebook in notebooks:
            self.execute_notebook(notebook)
        
        # Print summary
        self.print_summary()
        
        # Return overall success
        return all(r['success'] for r in self.results)
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed
        total_time = sum(r['execution_time'] for r in self.results)
        total_cells = sum(r['cell_count'] for r in self.results)
        
        print(f"\nTotal notebooks: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Total cells executed: {total_cells}")
        
        if failed > 0:
            print("\n❌ FAILED NOTEBOOKS:")
            for result in self.results:
                if not result['success']:
                    print(f"\n  {result['notebook']}:")
                    print(f"    Error: {result['error'][:200] if result['error'] else 'Unknown'}")
        
        print("\n" + "="*70)
        if passed == total:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED")
        print("="*70)
        
        # Save detailed report
        self.save_report()
    
    def save_report(self):
        """Save detailed test report to JSON."""
        report_path = self.notebook_dir / "test_report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_notebooks': len(self.results),
            'passed': sum(1 for r in self.results if r['success']),
            'failed': sum(1 for r in self.results if not r['success']),
            'total_execution_time': sum(r['execution_time'] for r in self.results),
            'total_cells': sum(r['cell_count'] for r in self.results),
            'results': self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_path}")

def main():
    """Main entry point."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run tests
    tester = NotebookTester(notebook_dir="notebooks")
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
