# UFoE 후보 서열 보고서

**타입:** B (Balanced, EC 21% / TZ 47% / BZ 32%)
**생성 수:** 10

## 요약 테이블

| ID | Length | Primary | Strict | Rg (Å) | Hydro (mean) | pLDDT | PDB |
|----|--------|---------|--------|--------|---------------|-------|-----|
| UFoE_B_001 | 60 | ✓ | ✗ | 16.34 | -1.18 | 85.0 | UFoE_B_001.pdb |
| UFoE_B_002 | 61 | ✗ | ✗ | 16.68 | -1.11 | 85.0 | UFoE_B_002.pdb |
| UFoE_B_003 | 62 | ✗ | ✗ | 16.73 | -1.37 | 85.0 | UFoE_B_003.pdb |
| UFoE_B_004 | 63 | ✗ | ✗ | 15.89 | -1.44 | 85.0 | UFoE_B_004.pdb |
| UFoE_B_005 | 64 | ✓ | ✓ | 16.19 | -0.7 | 85.0 | UFoE_B_005.pdb |
| UFoE_B_006 | 65 | ✗ | ✗ | 15.09 | -1.13 | 85.0 | UFoE_B_006.pdb |
| UFoE_B_007 | 66 | ✗ | ✗ | 15.88 | -1.23 | 85.0 | UFoE_B_007.pdb |
| UFoE_B_008 | 67 | ✗ | ✗ | 16.64 | -1.34 | 85.0 | UFoE_B_008.pdb |
| UFoE_B_009 | 68 | ✗ | ✗ | 15.92 | -1.1 | 85.0 | UFoE_B_009.pdb |
| UFoE_B_010 | 69 | ✓ | ✗ | 16.45 | -0.78 | 85.0 | UFoE_B_010.pdb |

## 필터 설명
- **Primary:** empty_center<30%, fibonacci [1.0,3.0], zone_balance≥20%, density_gradient, hydrophobic_core
- **Strict:** empty_center<25%, fibonacci [1.2,2.5], 동일

## 출력 파일
- `candidates/sequences.fasta` — 통합 FASTA
- `candidates/structures/*.pdb` — ESMFold 예측 구조 (또는 mock)
- `candidates/candidate_report.md` — 본 보고서
