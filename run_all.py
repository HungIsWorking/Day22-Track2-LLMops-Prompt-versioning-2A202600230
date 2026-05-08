"""
Run all steps sequentially: Step 1 → Step 2 → Step 3 → Step 4
"""

from datetime import datetime
import subprocess
import sys
from pathlib import Path


EVIDENCE_DIR = Path(__file__).parent / "evidence"


def _log_path_for(script_name: str) -> Path:
    mapping = {
        "01_langsmith_rag_pipeline.py": "01_langsmith_traces_log.txt",
        "02_prompt_hub_ab_routing.py": "02_ab_routing_log.txt",
        "03_ragas_evaluation.py": "03_ragas_scores_log.txt",
        "04_guardrails_validator.py": "04_pii_demo_log.txt",
    }
    return EVIDENCE_DIR / mapping.get(script_name, f"{Path(script_name).stem}.log")


def _write_log(log_path: Path, content: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")


def run_script(script_name: str):
    """Run a Python script and report results."""
    print(f"\n{'='*70}")
    print(f"Running {script_name}...")
    print('='*70)
    log_path = _log_path_for(script_name)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        combined_output = ""
        if result.stdout:
            combined_output += result.stdout
        if result.stderr:
            combined_output += ("\n" if combined_output else "") + result.stderr
        _write_log(log_path, combined_output)

        if combined_output:
            print(combined_output, end="" if combined_output.endswith("\n") else "\n")
        print(f"\n📝 Log saved to {log_path.relative_to(Path(__file__).parent)}")
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️  {script_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {str(e)}")
        return False

def main():
    print("\n" + "="*70)
    print("Day 22 RAG Lab — All Steps (Gemini Edition)")
    print("="*70)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        f"Day 22 RAG Lab run at {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    
    steps = [
        ("01_langsmith_rag_pipeline.py", "LangSmith RAG Pipeline"),
        ("02_prompt_hub_ab_routing.py", "Prompt Hub & A/B Routing"),
        ("03_ragas_evaluation.py", "RAGAS Evaluation"),
        ("04_guardrails_validator.py", "Guardrails Validators"),
    ]
    
    results = {}
    for script, description in steps:
        print(f"\n▶️  Step: {description}")
        success = run_script(script)
        results[description] = success
        summary_lines.append(f"{description}: {'PASS' if success else 'FAIL'}")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    for description, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{description:<40} {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_steps = len(results)
    
    if total_passed == total_steps:
        print(f"\n🎉 All {total_steps} steps completed successfully!")
        print("\nNext steps:")
        print("1. Review LangSmith traces: https://smith.langchain.com")
        print("2. Check Prompt Hub: https://smith.langchain.com/hub")
        print("3. Review RAGAS report: data/ragas_report.json")
        print("4. Commit to GitHub with evidence/ folder")
    else:
        print(f"\n⚠️  {total_steps - total_passed} step(s) failed. Check errors above.")

    summary_lines.extend([
        "",
        f"Total passed: {total_passed}/{total_steps}",
        f"Overall: {'PASS' if total_passed == total_steps else 'FAIL'}",
    ])
    _write_log(EVIDENCE_DIR / "run_all_summary.txt", "\n".join(summary_lines) + "\n")
    print(f"\n📝 Summary log saved to evidence/run_all_summary.txt")
    
    return total_passed == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
