# UFoE 단백질 구조 생성 실험

UFoE 기하학적 조건을 **생성 제약**으로 사용해 안정 단백질 구조를 설계할 수 있는지 검증하는 실험 프로젝트입니다.

## 저장소 구조

| 폴더 | 설명 |
|------|------|
| **ufoe_protein_experiment/** | 메인 실험 코드·파이프라인·보고서 (진입점) |
| **pdb_complexes/** | Phase 0/5 검증용 PDB (복합체) |
| **pdb_p53/** | Phase 0/5 검증용 PDB (p53) |
| **pdb_large_scale/** | 대용량 PDB (`.gitignore`로 제외, 필요 시 별도 배치) |
| **claude code/ufoe_protein_experiment/** | 기준 파이프라인 참조용 |

## 빠른 시작

```bash
cd ufoe_protein_experiment
pip install -r requirements.txt
python3 run_reference_pipeline.py mock 10
```

자세한 설명·설계·실행 방법은 **[ufoe_protein_experiment/README.md](ufoe_protein_experiment/README.md)** 를 참고하세요.
