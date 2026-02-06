# 기준 파이프라인 (Claude Code)

Option C Step 1에서 완료된 **Phase 2 전체 파이프라인** 기준 코드가  
워크스페이스의 **`claude code/ufoe_protein_experiment/`** 폴더에 있습니다.

## 위치

```
단백질 구조 생성 실험_Cursor/
├── claude code/ufoe_protein_experiment/   ← 기준 구현 (74 tests 통과)
│   ├── src/
│   │   ├── utils/       (pdb_writer, pdb_parser, constants)
│   │   ├── filters/     (ufoef_filters)
│   │   ├── generator/   (ufoef_generator, control_generators)
│   │   ├── folding/     (md_simulator, esmfold_predictor, folding_metrics)
│   │   ├── pipeline/    (pipeline, elimination_loop)
│   │   └── analysis/    (statistics)
│   └── tests/           (test_filters, test_generator, test_phase2)
└── ufoe_protein_experiment/               ← 현재 프로젝트 (데이터·실험 스크립트)
```

## 기준 코드 실행 방법

### 1. 테스트 (74개 전체)

```bash
cd "claude code/ufoe_protein_experiment"
python3 -m pytest tests/ -v
```

### 2. Mock 실험 (합성 데이터, ~수 초)

```bash
cd "claude code/ufoe_protein_experiment"
python3 -c "
from src.pipeline.pipeline import run_mock_experiment
results = run_mock_experiment(n=10)
from src.pipeline.pipeline import Phase2Pipeline
Phase2Pipeline.print_summary(results)
"
```

### 3. Quick 실험 (n=5, 1ns MD)

```bash
cd "claude code/ufoe_protein_experiment"
python3 -c "
from src.pipeline.pipeline import run_quick_experiment
results = run_quick_experiment()
from src.pipeline.pipeline import Phase2Pipeline
Phase2Pipeline.print_summary(results)
"
"
```

### 4. 전체 실험 (OpenMM 필요)

```bash
cd "claude code/ufoe_protein_experiment"
python3 -c "
from src.pipeline.pipeline import Phase2Pipeline, PipelineConfig
config = PipelineConfig(run_mode='full', esmfold_mode='api')
pipeline = Phase2Pipeline(config)
results = pipeline.run_full()
pipeline.save_results(results)
Phase2Pipeline.print_summary(results)
"
```

## 설계 요약 (기준 코드)

- **Mock 모드**: OpenMM/ESMFold 미설치 환경에서 합성 데이터로 전체 흐름 검증
- **접힘 성공**: RMSD 수렴(σ<2Å), Rg 안정(변화<30%), 에너지 수렴(CV<5%) 중 2개 이상 통과
- **자기일관성 K/M**: 접힘 후 UFoE 필터 재통과 비율. **K/M ≥ 70%** 이면 자기일관적 설계 원리로 인정
- **대조군**: Random, Shuffled, Inverted 3종

## Step 2·3

- **Step 2**: 이 프로젝트(`ufoe_protein_experiment`) 또는 다른 에이전트가 위 기준 파이프라인을 동일 설정으로 실행
- **Step 3**: 결과 CSV/JSON을 모아 통계 비교 및 통합
