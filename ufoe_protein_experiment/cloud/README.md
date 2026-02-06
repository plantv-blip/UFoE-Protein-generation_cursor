# UFoE Full 실험 — 클라우드 실행 가이드

GPU 환경(AWS, GCP, Colab)에서 Full 파이프라인을 one-click으로 실행하기 위한 스크립트와 문서입니다.

---

## 1. Google Colab

1. **노트북 열기:** `notebooks/UFoE_Full_Experiment.ipynb` 를 Colab에 업로드 또는 Drive에 배치 후 열기.
2. **런타임:** GPU 런타임 선택 (런타임 → 런타임 유형 변경 → GPU).
3. **실행 순서:**
   - 셀 1: 프로젝트 클론 또는 Drive 마운트
   - 셀 2: 의존성 설치 (`pip install ...`, OpenMM 선택)
   - 셀 3: (선택) 후보 서열 생성 `generate_candidates.py`
   - 셀 4: Full 파이프라인 실행
   - 셀 5: 결과 시각화

**주의:** Colab 기본 환경에는 OpenMM가 없을 수 있음. 설치 실패 시 `run_mode='mock'` 또는 `run_mode='quick'` 로 동작 확인.

---

## 2. AWS / GCP VM

### 2.1 환경 설정

```bash
# 프로젝트 루트로 이동 (ufoe_protein_experiment 상위 또는 내부)
cd /path/to/ufoe_protein_experiment
chmod +x cloud/setup.sh cloud/run_full.sh
./cloud/setup.sh
```

### 2.2 Full 실험 실행

```bash
# 기본 출력 디렉터리: output/full_run
./cloud/run_full.sh

# 출력 디렉터리 지정
./cloud/run_full.sh /opt/ufoe/output/run_001
```

### 2.3 Docker (선택)

프로젝트에 `Dockerfile` 이 있다면:

```bash
docker build -t ufoe-full .
docker run --gpus all -v $(pwd)/output:/app/output ufoe-full ./cloud/run_full.sh /app/output/run_001
```

현재는 `Dockerfile` 미포함. 필요 시 `setup.sh` 내용을 기반으로 Dockerfile 작성.

---

## 3. 실행 모드

| 모드 | 설명 | OpenMM | ESMFold |
|------|------|--------|---------|
| **mock** | 합성 궤적, 전체 흐름 검증 | 불필요 | mock |
| **quick** | n=5, 1ns MD | 필요 | api/mock |
| **full** | n=100, 100ns MD | 필요 (GPU 권장) | api 또는 local |

---

## 4. 출력 구조

- `output/<run_dir>/results/` — 접힘 메트릭 CSV, elimination JSON, summary.json
- `output/<run_dir>/pdb/` — UFoE·대조군 PDB
- `output/<run_dir>/md/` — MD 궤적 (설정에 따라)

---

## 5. 문제 해결

- **OpenMM 설치 실패:** `run_mode='mock'` 또는 `run_mode='quick'` 로 파이프라인만 검증.
- **ESMFold API 제한:** `esmfold_mode='mock'` 사용 또는 로컬 ESMFold 설치.
- **기준 파이프라인 미사용:** `claude code/ufoe_protein_experiment` 가 없으면 `run_full.sh` 는 현재 프로젝트의 `src.pipeline` 을 사용합니다.
