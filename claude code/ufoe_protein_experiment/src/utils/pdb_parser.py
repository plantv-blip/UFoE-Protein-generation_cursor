"""
PDB 구조 파싱 유틸리티

BioPython PDB 모듈을 감싸서 UFoE 필터에 필요한 잔기 수준 정보를 추출한다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.constants import (
    HYDROPHOBICITY_KD,
    PDB_ATOM_CA,
    STANDARD_AA,
)

# BioPython PDB 파서의 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="Bio")


@dataclass
class Residue:
    """단일 잔기의 필수 정보를 담는 데이터 클래스."""

    id: int                   # 잔기 번호
    name: str                 # 3글자 코드 (ALA, GLY, ...)
    chain: str                # 체인 ID
    ca_coord: np.ndarray      # Cα 좌표 (3,)
    hydrophobicity: float     # Kyte-Doolittle 소수성 지수

    def distance_to(self, point: np.ndarray) -> float:
        """주어진 3D 좌표까지의 유클리드 거리."""
        return float(np.linalg.norm(self.ca_coord - point))


@dataclass
class ProteinStructure:
    """단백질 구조의 잔기 수준 표현."""

    pdb_id: str
    residues: list[Residue] = field(default_factory=list)
    _center: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def ca_coords(self) -> np.ndarray:
        """모든 Cα 좌표 배열 (N, 3)."""
        return np.array([r.ca_coord for r in self.residues])

    @property
    def geometric_center(self) -> np.ndarray:
        """Cα 좌표의 기하학적 중심."""
        if self._center is None:
            self._center = self.ca_coords.mean(axis=0)
        return self._center

    @geometric_center.setter
    def geometric_center(self, value: np.ndarray):
        self._center = value

    def residue_distances_from_center(self) -> dict[int, float]:
        """각 잔기의 기하학적 중심으로부터의 거리."""
        center = self.geometric_center
        return {r.id: r.distance_to(center) for r in self.residues}

    def get_hydrophobicity_array(self) -> np.ndarray:
        """잔기 순서대로 소수성 지수 배열."""
        return np.array([r.hydrophobicity for r in self.residues])


def parse_pdb_file(filepath: str | Path, chain_id: str | None = None) -> ProteinStructure:
    """PDB 파일을 파싱하여 ProteinStructure 반환.

    Parameters
    ----------
    filepath : str or Path
        PDB 파일 경로
    chain_id : str, optional
        특정 체인만 파싱. None이면 첫 번째 체인 사용.

    Returns
    -------
    ProteinStructure
    """
    from Bio.PDB import PDBParser

    filepath = Path(filepath)
    pdb_id = filepath.stem[:4]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(filepath))

    model = structure[0]  # 첫 번째 모델
    residues = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue

        for residue in chain:
            resname = residue.get_resname().strip()

            # 표준 아미노산만 처리
            if resname not in STANDARD_AA:
                continue

            # hetero flag가 있는 잔기 건너뛰기 (물, 리간드 등)
            if residue.id[0] != " ":
                continue

            # Cα 원자 추출
            if PDB_ATOM_CA not in residue:
                continue

            ca_atom = residue[PDB_ATOM_CA]
            ca_coord = np.array(ca_atom.get_vector().get_array(), dtype=np.float64)

            residues.append(Residue(
                id=residue.id[1],
                name=resname,
                chain=chain.id,
                ca_coord=ca_coord,
                hydrophobicity=HYDROPHOBICITY_KD.get(resname, 0.0),
            ))

        # chain_id를 지정하지 않았으면 첫 체인만
        if chain_id is None:
            break

    return ProteinStructure(pdb_id=pdb_id, residues=residues)


def parse_pdb_from_coords(
    pdb_id: str,
    ca_coords: np.ndarray,
    residue_names: list[str],
    chain_id: str = "A",
) -> ProteinStructure:
    """좌표 배열과 잔기 이름 리스트로 ProteinStructure를 직접 생성.

    Phase 1 생성기에서 사용.

    Parameters
    ----------
    pdb_id : str
    ca_coords : np.ndarray of shape (N, 3)
    residue_names : list[str] of length N (3글자 코드)
    chain_id : str

    Returns
    -------
    ProteinStructure
    """
    if len(ca_coords) != len(residue_names):
        raise ValueError(
            f"좌표 수({len(ca_coords)})와 잔기 이름 수({len(residue_names)})가 불일치"
        )

    residues = []
    for i, (coord, name) in enumerate(zip(ca_coords, residue_names)):
        residues.append(Residue(
            id=i + 1,
            name=name,
            chain=chain_id,
            ca_coord=np.array(coord, dtype=np.float64),
            hydrophobicity=HYDROPHOBICITY_KD.get(name, 0.0),
        ))

    return ProteinStructure(pdb_id=pdb_id, residues=residues)
