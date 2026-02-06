#!/usr/bin/env python3
"""
Colab에서 전체 파이프라인을 한 번에 실행하는 진입점.

사용 (Colab에서 환경 설정 셀 실행 후):
  !python run_colab.py
  !python run_colab.py mock 5   # mock, n=5
  !python run_colab.py quick 3 # quick 모드, n=3 (실제 MD/API 사용 시)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CLAUDE_DIR = REPO_ROOT / "claude code" / "ufoe_protein_experiment"


def main():
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "mock").strip()
    n = 10
    if len(sys.argv) > 2:
        try:
            n = int(sys.argv[2])
        except ValueError:
            pass

    if CLAUDE_DIR.is_dir():
        sys.path.insert(0, str(CLAUDE_DIR))
        import os
        os.chdir(CLAUDE_DIR)
        from src.pipeline.pipeline import Phase2Pipeline, PipelineConfig, run_mock_experiment

        if mode == "mock":
            results = run_mock_experiment(n=n, output_dir="output/colab_run")
        else:
            config = PipelineConfig(
                output_dir="output/colab_run",
                run_mode="quick" if mode == "quick" else "full",
                esmfold_mode="api" if "api" in mode else "mock",
            )
            pipeline = Phase2Pipeline(config)
            results = pipeline.run_full()
            pipeline.save_results(results)
        Phase2Pipeline.print_summary(results)
        return 0

    # claude code 없을 때: ufoe_protein_experiment 단순 파이프라인
    sys.path.insert(0, str(ROOT))
    from src.pipeline.pipeline import run_mock_experiment
    results = run_mock_experiment(n=n)
    print("접힘 성공률:", results.get("fold_success_rate"), "| K/M:", results.get("K_over_M"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
