"""
Phase 2 입출력 규격.
Claude Code 기준 파이프라인과 동일한 형식으로 결과를 교환하기 위한 스키마.
"""

from pathlib import Path
from typing import List, Optional, TypedDict

# Phase 2 입력: 서열 리스트 (그룹 식별자 포함)
PHASE2_INPUT_COLUMNS = ["id", "group", "sequence", "length"]
# group: "UFoE", "Random", "RFdiffusion"

# Phase 2 출력 (각 구조별 1행)
PHASE2_OUTPUT_COLUMNS = [
    "id",
    "group",
    "sequence",
    "fold_success",       # bool: RMSD 수렴 < 3Å
    "final_rmsd",        # float (Å)
    "energy_score",      # float (e.g. Rosetta kcal/mol)
    "output_pdb_path",   # str
    "zone_EC",           # float: 접힘 후 EC 비율
    "zone_TZ",
    "zone_BZ",
    "runtime_sec",
    "phase",             # "2a" | "2b" | "2c"
]

# 디렉터리 규칙 (기준 코드와 통일)
PHASE2_DIRS = {
    "input_sequences": "data/phase2_input",
    "output_2a": "data/phase2a_output",
    "output_2b": "data/phase2b_output",
    "output_2c": "data/phase2c_output",
    "structures": "data/phase2_structures",
}


def get_phase2_structure_dir(base: Path, phase: str) -> Path:
    """Phase 2 결과 PDB가 저장되는 디렉터리."""
    d = base / PHASE2_DIRS["structures"] / phase
    d.mkdir(parents=True, exist_ok=True)
    return d
