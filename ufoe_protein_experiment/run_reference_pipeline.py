#!/usr/bin/env python3
"""
기준 파이프라인(Claude Code) 실행 스크립트.

claude code/ufoe_protein_experiment 에서 run_mock_experiment 또는 run_quick_experiment를
실행하고 요약을 출력합니다. 워크스페이스 루트 기준으로 상대 경로를 사용합니다.
"""

import subprocess
import sys
from pathlib import Path

# 프로젝트 루트 = ufoe_protein_experiment 의 상위
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CLAUDE_PROJECT = WORKSPACE / "claude code" / "ufoe_protein_experiment"


def main():
    if not CLAUDE_PROJECT.is_dir():
        print(f"기준 코드 경로가 없습니다: {CLAUDE_PROJECT}")
        print('"claude code/ufoe_protein_experiment" 폴더를 확인하세요.')
        return 1

    mode = "mock"
    n = 10
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    if len(sys.argv) > 2:
        try:
            n = int(sys.argv[2])
        except ValueError:
            pass

    if mode == "quick":
        code = """
from src.pipeline.pipeline import run_quick_experiment, Phase2Pipeline
results = run_quick_experiment()
Phase2Pipeline.print_summary(results)
"""
    else:
        code = f"""
from src.pipeline.pipeline import run_mock_experiment, Phase2Pipeline
results = run_mock_experiment(n={n})
Phase2Pipeline.print_summary(results)
"""

    cmd = [sys.executable, "-c", code]
    result = subprocess.run(
        cmd,
        cwd=str(CLAUDE_PROJECT),
        timeout=120,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
