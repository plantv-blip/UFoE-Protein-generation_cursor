# UFoE 단백질 구조 생성 실험 (Phase 0–5)

UFoE 기반 소거 특허의 5개 기하학적 제약을 **생성 제약**으로 뒤집어, 안정 구조를 설계할 수 있는지 검증하는 실험입니다.  
**설계 종합·프레임 전환·Phase 2–5 설계:** [DESIGN.md](DESIGN.md) 참고.  
**기준 파이프라인(Claude Code, 74 tests):** [REFERENCE_PIPELINE.md](REFERENCE_PIPELINE.md) — `claude code/ufoe_protein_experiment/` 폴더 사용법 및 실행 방법.

## 구조

```
ufoe_protein_experiment/
├── src/
│   ├── filters/          # Phase 0: 5-필터
│   ├── generator/        # Phase 1: Type B 서열 생성기
│   ├── phase2/           # Phase 2: 2a(OpenMM), 2b(ESMFold+MD), 2c(대조군) 입출력·러너
│   ├── phase3/           # Phase 3: 소거 루프 (K/M, 탈락 분석)
│   └── utils/
├── data/
│   ├── phase2_input/     # Phase 2 입력 서열 CSV
│   ├── phase2a_output/   # 2a 결과
│   ├── phase2b_output/   # 2b 결과
│   ├── phase2c_output/   # 2c 결과
│   └── phase3_output/   # Phase 3 보고서
├── scripts/collect_metrics.py   # 검증 지표 수집
├── run_phase0_validation.py    # Phase 0 전체 PDB 검증
├── run_phase2a.py / run_phase2b.py / run_phase2c.py
├── run_phase3.py               # Phase 3 소거 루프
├── run_reference_pipeline.py   # 기준 파이프라인(Claude Code) 실행
├── run_colab.py                # Colab 한 번에 실행 진입점
├── DESIGN.md
├── REFERENCE_PIPELINE.md       # 기준 코드 위치 및 실행 가이드
├── REPORT_Phase0-2_Summary.md  # Phase 0-2 종합 보고서
├── run_phase5.py               # Phase 5 PDB 비교 (TM-align)
├── tests/
└── requirements.txt
```

## Google Colab에서 바로 실행

- **Colab A100 · n=5 한 번에 실행:** [UFoE_Colab_A100_RunAll.ipynb](https://colab.research.google.com/github/plantv-blip/UFoE-Protein-generation_cursor/blob/main/ufoe_protein_experiment/notebooks/UFoE_Colab_A100_RunAll.ipynb) — 셀 순서대로: 설치 → 생성(n=5) → MD → 결과.
- **전체 실험 노트북:** [UFoE_Full_Experiment.ipynb](https://colab.research.google.com/github/plantv-blip/UFoE-Protein-generation_cursor/blob/main/ufoe_protein_experiment/notebooks/UFoE_Full_Experiment.ipynb) — **Runtime → Run all** 또는 셀 하나씩 실행. 환경 설정 후 `!python run_colab.py` 로 파이프라인만 실행 가능.

## 설치 (로컬)

```bash
pip install -r requirements.txt
```

## Phase 0: 5-필터

- **Empty Center**: 중심 5Å 내 잔기 비율 (Primary <30%, Strict <25%)
- **Fibonacci Ratio**: 거리 Q3/Q1 (Primary [1.0, 3.0], Strict [1.2, 2.5])
- **Zone Balance**: EC/TZ/BZ 각 ≥20%
- **Density Gradient**: 내부→외부 밀도 감소
- **Hydrophobic Core**: 내부 평균 소수성 > 외부

검증: `data/pdb_representative/`에 PDB를 넣고 `notebooks/phase0_validation.ipynb` 실행 → **80%+ 통과** 목표.

## Phase 1: Type B 생성기

- Type B 분포: EC ~17%, TZ ~47%, BZ ~36%
- Zone별 소수성 규칙, Ramachandran·충돌 제약 적용
- `batch_generate(n=1000, type='B')` → 후보 (sequence, structure) 리스트

## 테스트

```bash
python3 -m pytest tests/ -v
```

## 기준 파이프라인 실행 (Claude Code)

`claude code/ufoe_protein_experiment` 폴더의 기준 구현을 이 프로젝트에서 실행:

```bash
# Mock 실험 (n=10, 합성 데이터, 수 초)
python3 run_reference_pipeline.py mock

# Mock n=5
python3 run_reference_pipeline.py mock 5

# Quick 실험 (n=5, 1ns MD)
python3 run_reference_pipeline.py quick
```

자세한 사용법·테스트·전체 실험: [REFERENCE_PIPELINE.md](REFERENCE_PIPELINE.md).

## Phase 2–3 (로컬 스텁)

- **Phase 2:** `python3 run_phase2a.py` → 입력 CSV·스텁 결과 생성. 실제 MD는 기준 코드(Claude) 또는 GPU에서 실행.
- **Phase 3:** `python3 run_phase3.py --results data/phase2a_output/phase2a_results.csv` → K/M, 탈락 필터 분석.
- **지표 수집:** `python3 scripts/collect_metrics.py` → Phase 2/3 요약.
- **Phase 5:** `python3 run_phase5.py` → 자기일관적 구조 vs PDB TM-align 비교.
- **리포트 생성:** `python3 scripts/generate_report.py` → `data/REPORT_Phase0-5_Results.md` 생성.

## 사용 예

```python
# 필터 (residue dict 리스트 또는 BioPython Structure)
from src.filters.ufoef_filters import apply_all_filters, batch_validate
results = apply_all_filters(structure, strict=False)

# 생성기
from src.generator.ufoef_generator import generate_candidate, batch_generate
seq, structure = generate_candidate(length=100, type="B")
candidates = batch_generate(n=1000, type="B", length=100)

# Phase 3 소거 루프
from src.phase3.elimination_loop import run_elimination_loop
result = run_elimination_loop(Path("data/phase2a_output/phase2a_results.csv"))
# result["K_over_M"], result["failed_by_filter"]
```
