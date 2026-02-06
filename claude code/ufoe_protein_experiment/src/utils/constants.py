"""
UFoE 단백질 구조 생성 실험 — 상수 정의

아미노산 소수성, Ramachandran 허용 범위, Zone 경계, 필터 임계값 등
"""

import numpy as np

# =============================================================================
# 아미노산 소수성 지수 (Kyte-Doolittle scale)
# 양수 = 소수성, 음수 = 친수성
# =============================================================================
HYDROPHOBICITY_KD = {
    "ALA": 1.8,   "ARG": -4.5,  "ASN": -3.5,  "ASP": -3.5,
    "CYS": 2.5,   "GLN": -3.5,  "GLU": -3.5,  "GLY": -0.4,
    "HIS": -3.2,  "ILE": 4.5,   "LEU": 3.8,   "LYS": -3.9,
    "MET": 1.9,   "PHE": 2.8,   "PRO": -1.6,  "SER": -0.8,
    "THR": -0.7,  "TRP": -0.9,  "TYR": -1.3,  "VAL": 4.2,
}

# 3글자 → 1글자 코드
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}

# 표준 아미노산 3글자 코드 집합
STANDARD_AA = set(HYDROPHOBICITY_KD.keys())

# 소수성/친수성 분류
HYDROPHOBIC_AA = {aa for aa, h in HYDROPHOBICITY_KD.items() if h > 0}
HYDROPHILIC_AA = {aa for aa, h in HYDROPHOBICITY_KD.items() if h < -1.0}
NEUTRAL_AA = STANDARD_AA - HYDROPHOBIC_AA - HYDROPHILIC_AA

# =============================================================================
# Zone 경계 (Å)
# =============================================================================
EC_RADIUS = 5.0    # Empty Center 경계 반경
TZ_RADIUS = 13.0   # Transition Zone 외부 경계

# =============================================================================
# UFoE 필터 임계값 — Primary (특허) / Strict (실험 설계)
# =============================================================================
FILTER_THRESHOLDS = {
    "primary": {
        "empty_center_max_ratio": 0.30,       # < 30%
        "fibonacci_ratio_range": (1.0, 3.0),  # Q3/Q1 ∈ [1.0, 3.0]
        "zone_balance_min_ratio": 0.20,        # ≥ 20%
        # density_gradient와 hydrophobic_core는 방향성 조건이므로 범위 없음
    },
    "strict": {
        "empty_center_max_ratio": 0.25,       # < 25%
        "fibonacci_ratio_range": (1.2, 2.5),  # Q3/Q1 ∈ [1.2, 2.5]
        "zone_balance_min_ratio": 0.20,        # ≥ 20%
    },
}

# =============================================================================
# 단백질 Type별 Zone 분포 타겟 (단백질 주기율표 v2)
# =============================================================================
TYPE_TARGETS = {
    "A": {  # Boundary-Heavy
        "center": 0.120,
        "transition": 0.349,
        "boundary": 0.531,
        "description": "Boundary-Heavy — 표면 상호작용 중시, DNA-binding 등",
    },
    "B": {  # Balanced
        "center": 0.169,
        "transition": 0.467,
        "boundary": 0.363,
        "description": "Balanced — 가장 일반적인 유형 (51.4%)",
    },
    "C": {  # Core-Heavy
        "center": 0.378,
        "transition": 0.450,
        "boundary": 0.172,
        "description": "Core-Heavy — 밀집된 코어, 효소/활성 부위",
    },
}

# UFoE 필터-호환 타겟: Zone Balance(min ≥ 20%) 충족하도록 조정
# 관측된 Type 분포에서 EC 비율이 20% 미만인 경우, 필터 통과를 위해 보정
TYPE_TARGETS_FILTER_COMPATIBLE = {
    "A": {  # EC를 20%로 올리고, 나머지를 비례 조정
        "center": 0.210,
        "transition": 0.330,
        "boundary": 0.460,
        "description": "Boundary-Heavy (filter-compatible)",
    },
    "B": {  # EC를 21%로 올리고, TZ/BZ를 비례 조정
        "center": 0.210,
        "transition": 0.450,
        "boundary": 0.340,
        "description": "Balanced (filter-compatible)",
    },
    "C": {  # 이미 EC > 20%, BZ를 20%로 올림
        "center": 0.360,
        "transition": 0.430,
        "boundary": 0.210,
        "description": "Core-Heavy (filter-compatible)",
    },
}

# Zone 분포 허용 오차 (±)
ZONE_TOLERANCE = 0.05

# =============================================================================
# Ramachandran plot 허용 범위 (degrees)
# 주요 2차 구조별 (phi, psi) 범위
# =============================================================================
RAMACHANDRAN_REGIONS = {
    "alpha_helix": {
        "phi": (-80.0, -48.0),
        "psi": (-59.0, -27.0),
    },
    "beta_sheet": {
        "phi": (-150.0, -90.0),
        "psi": (90.0, 150.0),
    },
    "left_alpha": {
        "phi": (50.0, 70.0),
        "psi": (25.0, 55.0),
    },
    "polyproline_II": {
        "phi": (-80.0, -60.0),
        "psi": (120.0, 160.0),
    },
}

# 일반적인 Ramachandran 허용 범위 (glycine 제외)
RAMA_GENERAL_ALLOWED = {
    "phi": (-180.0, 0.0),   # 대부분의 잔기는 phi < 0
    "psi": (-180.0, 180.0),  # psi는 넓은 범위 허용
}

# Glycine은 특별히 넓은 범위 허용
RAMA_GLYCINE_ALLOWED = {
    "phi": (-180.0, 180.0),
    "psi": (-180.0, 180.0),
}

# Proline은 제한적
RAMA_PROLINE_ALLOWED = {
    "phi": (-80.0, -55.0),
    "psi": (-60.0, 160.0),
}

# =============================================================================
# 구조 생성 제약 조건
# =============================================================================
# 이상적인 결합 길이/각도
BOND_LENGTH_CA_CA = 3.8       # Å (Cα-Cα 평균 거리)
BOND_LENGTH_TOLERANCE = 0.2   # Å
MIN_CONTACT_DISTANCE = 2.0    # Å (잔기 간 최소 거리 — 충돌 방지)

# 2차 구조 파라미터
HELIX_RISE_PER_RESIDUE = 1.5  # Å
HELIX_RADIUS = 2.3            # Å
HELIX_RESIDUES_PER_TURN = 3.6

SHEET_RISE_PER_RESIDUE = 3.3  # Å
SHEET_CA_DISTANCE = 3.5       # Å

# =============================================================================
# PDB 관련 상수
# =============================================================================
PDB_ATOM_CA = "CA"  # Cα 원자 이름
PDB_CHAIN_DEFAULT = "A"

# =============================================================================
# Phase 2: MD 시뮬레이션 파라미터
# =============================================================================

# 시뮬레이션 공통 설정
MD_TEMPERATURE = 300.0           # K (시뮬레이션 온도)
MD_PRESSURE = 1.0                # atm
MD_TIMESTEP_FS = 2.0             # fs (적분 시간 단계)
MD_FRICTION_COEFF = 1.0          # ps^-1 (Langevin 마찰 계수)
MD_NONBONDED_CUTOFF = 10.0       # Å (비결합 상호작용 컷오프)

# Phase 2a: 순수 MD 시뮬레이션
MD_SIMULATION_TIME_NS = 100.0    # ns (각 구조당 시뮬레이션 시간)
MD_EQUILIBRATION_NS = 1.0        # ns (평형화 시간)
MD_REPORT_INTERVAL_PS = 10.0     # ps (좌표 저장 간격)
MD_ENERGY_INTERVAL_PS = 1.0      # ps (에너지 저장 간격)
MD_N_SAMPLES_2A = 100            # Phase 2a 샘플 수
MD_LENGTH_RANGE = (60, 100)      # 서열 길이 범위 (잔기 수)

# Phase 2b: ESMFold + MD 정제
ESMFOLD_MODEL_NAME = "facebook/esmfold_v1"
MD_REFINEMENT_TIME_NS = 10.0     # ns (ESMFold 후 MD 정제 시간)
MD_N_SAMPLES_2B = 100            # Phase 2b 샘플 수

# Phase 2c: 대조군
MD_N_CONTROL_RANDOM = 100        # Random 대조군 수
MD_N_CONTROL_RFDIFF = 100        # RFdiffusion 대조군 수

# =============================================================================
# Phase 2: 접힘 성공 판정 기준
# =============================================================================
FOLDING_CRITERIA = {
    # RMSD 수렴: 마지막 10ns의 RMSD 표준편차
    "rmsd_convergence_max_std": 2.0,       # Å (수렴 판정 최대 표준편차)
    "rmsd_convergence_window_ns": 10.0,    # ns (수렴 판정 윈도우)

    # 반경: 전체 시뮬레이션에서 Rg(회전 반경) 변화
    "rg_stability_max_change": 0.3,         # 비율 (최대 30% 변화 허용)

    # 에너지: 포텐셜 에너지 안정화
    "energy_convergence_max_std_ratio": 0.05,  # 표준편차/평균 < 5%

    # 2차 구조 보존율
    "ss_preservation_min": 0.50,            # 최소 50% 보존

    # Rosetta-like scoring (근사)
    "contact_order_max": 0.30,              # 최대 접촉 순서
}

# =============================================================================
# Phase 3: 자기일관성 비율 (K/M)
# =============================================================================
SELF_CONSISTENCY_THRESHOLD = 0.70   # K/M ≥ 70%이면 성공

# =============================================================================
# Phase 4: 통계 분석 파라미터
# =============================================================================
STATS_SIGNIFICANCE_LEVEL = 0.05     # α = 0.05
STATS_COHENS_D_LARGE = 0.8         # 큰 효과 크기 기준
