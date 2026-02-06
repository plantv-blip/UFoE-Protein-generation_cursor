"""
Cα 좌표 → PDB / FASTA 파일 변환.
Claude 기준 파이프라인과 동일 입출력 규격.
"""

from pathlib import Path
from typing import List, Dict, Any, Union


def structure_to_pdb(
    residues: List[Dict[str, Any]],
    path: Union[str, Path],
    chain_id: str = "A",
) -> Path:
    """
    residue dict 리스트 (x, y, z, resname_1, residue_id 또는 resseq) → PDB 파일.
    CA만 기록.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, r in enumerate(residues):
        resseq = r.get("resseq", i + 1)
        if isinstance(resseq, tuple):
            resseq = resseq[1]
        name = (r.get("resname_3") or r.get("resname_1") or "G").upper()
        if len(name) == 1:
            from .constants import AA1_TO_3
            name = AA1_TO_3.get(name, "GLY")
        x, y, z = r["x"], r["y"], r["z"]
        lines.append(
            f"ATOM  {i+1:5d}  CA  {name:3s} {chain_id}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("END")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def structure_to_fasta(sequence: str, path: Union[str, Path], id_: str = "seq") -> Path:
    """서열 → FASTA 파일."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{id_}\n{sequence}\n", encoding="utf-8")
    return path


def coords_to_residues(
    coords: List[tuple],
    sequence: str,
    chain_id: str = "A",
) -> List[Dict[str, Any]]:
    """(x,y,z) 리스트 + 서열 → residue dict 리스트."""
    from .constants import AA1_TO_3
    residues = []
    for i, (x, y, z) in enumerate(coords):
        resseq = i + 1
        aa = sequence[i] if i < len(sequence) else "G"
        residues.append({
            "residue_id": (chain_id, resseq),
            "resseq": resseq,
            "x": float(x), "y": float(y), "z": float(z),
            "resname_1": aa,
            "resname_3": AA1_TO_3.get(aa, "GLY") if len(aa) == 1 else aa,
        })
    return residues
