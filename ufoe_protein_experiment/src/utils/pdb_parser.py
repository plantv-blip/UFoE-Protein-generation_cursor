"""
PDB 파싱 유틸: 구조 → 좌표/잔기 목록 추출.
"""

from pathlib import Path
from typing import List, Tuple, Optional

try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    Structure = None
    Atom = None
    Residue = None

from .constants import AA3_TO_1, AMINO_ACID_HYDROPHOBICITY


def load_structure(path: str):
    """PDB/CIF 파일에서 Structure 로드. 첫 번째 모델 반환."""
    if not HAS_BIOPYTHON:
        raise ImportError("BioPython required. pip install biopython")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    return structure


def get_ca_atoms(structure) -> List[Tuple[str, int, Tuple[float, float, float], str]]:
    """
    구조에서 CA 원자만 추출.
    Returns: list of (resname_3, resseq, (x,y,z), resname_1)
    """
    out = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname()
                if resname not in AA3_TO_1:
                    continue
                try:
                    ca = residue["CA"]
                except KeyError:
                    continue
                x, y, z = ca.get_coord()
                out.append((resname, residue.id[1], (float(x), float(y), float(z)), AA3_TO_1[resname]))
        break
    return out


def structure_to_residue_list(structure):
    """
    Structure → 리스트 형태 (residue_id, x, y, z, resname_1letter).
    residue_id = (chain_id, resseq) 또는 단순 resseq.
    """
    rows = []
    for model in structure:
        for chain in model:
            cid = chain.id
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname()
                if resname not in AA3_TO_1:
                    continue
                try:
                    ca = residue["CA"]
                except KeyError:
                    continue
                x, y, z = ca.get_coord()
                rid = (cid, residue.id[1])
                rows.append({
                    "residue_id": rid,
                    "resseq": residue.id[1],
                    "x": float(x), "y": float(y), "z": float(z),
                    "resname_1": AA3_TO_1[resname],
                    "resname_3": resname,
                })
        break
    return rows


def get_hydrophobicity(resname_1: str) -> float:
    """1-letter 아미노산 코드 → 소수성 값."""
    return AMINO_ACID_HYDROPHOBICITY.get(resname_1, 0.0)
