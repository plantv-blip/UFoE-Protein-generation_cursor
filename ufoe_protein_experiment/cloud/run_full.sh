#!/bin/bash
# UFoE Full 실험 one-click 실행
# 사용: ./run_full.sh [출력디렉터리]
# 예: ./run_full.sh /opt/ufoe/output/run_001

set -e
OUTPUT_DIR="${1:-output/full_run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== UFoE Full 파이프라인 ==="
echo "출력: $OUTPUT_DIR"
echo "프로젝트: $PROJECT_ROOT"

# 기준 파이프라인(Claude Code)이 있으면 해당 경로에서 실행
CLAUDE_DIR="$PROJECT_ROOT/../claude code/ufoe_protein_experiment"
if [ -d "$CLAUDE_DIR" ]; then
  echo "기준 파이프라인 사용: $CLAUDE_DIR"
  cd "$CLAUDE_DIR"
  python3 -c "
from src.pipeline.pipeline import Phase2Pipeline, PipelineConfig
config = PipelineConfig(output_dir='$OUTPUT_DIR', run_mode='full', esmfold_mode='api')
pipeline = Phase2Pipeline(config)
results = pipeline.run_full()
pipeline.save_results(results)
Phase2Pipeline.print_summary(results)
"
else
  echo "로컬 pipeline 사용"
  cd "$PROJECT_ROOT"
  python3 -c "
import sys
sys.path.insert(0, '.')
from src.pipeline.pipeline import Phase2Pipeline, PipelineConfig
config = PipelineConfig(run_mode='full', esmfold_mode='api')
# output_dir는 pipeline 내부 기본값 사용
pipeline = Phase2Pipeline(config)
results = pipeline.run_full()
pipeline.save_results(results)
print('K/M:', results.elimination_ufoe_2a.self_consistency_ratio if results.elimination_ufoe_2a else 'N/A')
"
fi

echo "=== 완료 ==="
echo "결과: $OUTPUT_DIR"
