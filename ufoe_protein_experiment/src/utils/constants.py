"""
UFoE 실험 공통 상수: 아미노산 소수성, Zone 정의 등.
"""

# Kyte-Doolittle hydrophobicity scale (내부 = 소수성 높을수록, 외부 = 친수성)
# 값이 클수록 소수성(내부 선호)
AMINO_ACID_HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# 3-letter to 1-letter
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_3 = {v: k for k, v in AA3_TO_1.items()}

# Zone 반경 (Å)
EC_RADIUS = 5.0
TZ_OUTER_RADIUS = 13.0

# Primary vs Strict 임계값
EMPTY_CENTER_PRIMARY = 0.30   # < 30%
EMPTY_CENTER_STRICT = 0.25   # < 25%
FIBONACCI_RANGE_PRIMARY = (1.0, 3.0)
FIBONACCI_RANGE_STRICT = (1.2, 2.5)
ZONE_BALANCE_MIN_RATIO = 0.10  # min(EC, TZ, BZ) >= 10%

# Type B 타겟 분포 (비율)
TYPE_B_TARGET = {
    "EC": 0.17,   # Empty Center ~17%
    "TZ": 0.47,   # Transition Zone ~47%
    "BZ": 0.36,   # Boundary Zone ~36%
}
# Type B 필터 호환 (Zone Balance min≥20% 충족용)
TYPE_B_FILTER_COMPATIBLE = {
    "EC": 0.21,
    "TZ": 0.47,
    "BZ": 0.32,
}

# Ramachandran 허용 범위 (phi, psi in degrees) — 일반적 허용 영역
RAMACHANDRAN_ALLOWED = {
    "phi": (-180, 180),
    "psi": (-180, 180),
}
# 엄격한 허용: alpha-helix, beta-sheet 등
RAMACHANDRAN_ALPHA = {"phi": (-70, -35), "psi": (-60, -30)}
RAMACHANDRAN_BETA = {"phi": (-180, -90), "psi": (90, 180)}
RAMACHANDRAN_EXTENDED = {"phi": (-180, 0), "psi": (100, 180)}

# 잔기 간 최소 거리 (Å) — 충돌 방지
MIN_CA_CA_DISTANCE = 2.0
