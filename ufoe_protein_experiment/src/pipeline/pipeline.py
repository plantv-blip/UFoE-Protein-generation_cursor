"""
Phase 2a/2b/2c/3/4 전체 오케스트레이터.
run_mock_experiment(n=10), run_quick_experiment(), Phase2Pipeline(config).run_full(), save_results().
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import csv
import sys

def _root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class PipelineConfig:
    run_mode: str = "mock"   # "mock" | "full"
    esmfold_mode: str = "mock"  # "mock" | "api" | "local"
    n_ufoe: int = 100
    n_control_per_group: int = 100
    md_ns_2a: float = 100.0
    md_ns_quick: float = 1.0


def run_mock_experiment(n: int = 10) -> Dict[str, Any]:
    """빠른 테스트: mock 데이터, n개 서열, ~수 초."""
    root = _root()
    sys.path.insert(0, str(root))
    from src.generator.ufoef_generator import generate_candidate
    from src.folding.md_simulator import MDSimulator
    from src.folding.folding_metrics import folding_success_criteria, compute_rmsd_convergence, compute_rg_stability, compute_energy_convergence
    from src.filters.ufoef_filters import apply_all_filters
    from src.utils.pdb_parser import load_structure

    results_2a = []
    out_dir = root / "data" / "pipeline_mock" / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = MDSimulator(run_mode="mock")
    for i in range(n):
        length = 60 + (i % 41)
        seq, structure = generate_candidate(length=length, type="B")
        md_result = sim.run(structure, out_dir, f"ufoe_{i:04d}", length_ns=0.1)
        results_2a.append({
            "id": f"ufoe_{i:04d}",
            "group": "UFoE",
            "sequence": seq,
            "fold_success": md_result.fold_success,
            "final_rmsd": md_result.final_rmsd,
            "energy_score": md_result.energy_score,
            "output_pdb_path": md_result.output_pdb_path,
            "zone_EC": md_result.zone_EC,
            "zone_TZ": md_result.zone_TZ,
            "zone_BZ": md_result.zone_BZ,
            "runtime_sec": md_result.runtime_sec,
            "phase": "2a",
        })

    phase3_csv = root / "data" / "phase2a_output" / "phase2a_results.csv"
    phase3_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(phase3_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "group", "sequence", "fold_success", "final_rmsd", "energy_score",
            "output_pdb_path", "zone_EC", "zone_TZ", "zone_BZ", "runtime_sec", "phase",
        ], extrasaction="ignore")
        w.writeheader()
        w.writerows(results_2a)

    elim = run_elimination_loop(phase3_csv)
    return {
        "phase2a": results_2a,
        "phase3": elim,
        "n": n,
        "fold_success_rate": sum(1 for r in results_2a if r.get("fold_success")) / n if n else 0,
        "K_over_M": elim["K_over_M"],
        "self_consistent": elim.get("self_consistent", False),
    }


def run_quick_experiment() -> Dict[str, Any]:
    """축소 실험: n=5, 1ns MD."""
    config = PipelineConfig(run_mode="mock", n_ufoe=5, md_ns_2a=1.0)
    pipeline = Phase2Pipeline(config)
    return pipeline.run_full()


def run_elimination_loop(results_csv: Path, structures_dir: Optional[Path] = None, strict: bool = False) -> Dict[str, Any]:
    from .elimination_loop import run_elimination_loop as _run
    return _run(results_csv, structures_dir=structures_dir, strict=strict)


class Phase2Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.root = _root()
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

    def run_full(self) -> Dict[str, Any]:
        """Phase 2a (및 선택 2b/2c) + Phase 3 실행."""
        n = self.config.n_ufoe
        if self.config.run_mode == "mock":
            return run_mock_experiment(n=n)
        from src.generator.ufoef_generator import generate_candidate
        from src.folding.md_simulator import MDSimulator
        from src.folding.esmfold_predictor import ESMFoldPredictor
        out_2a = self.root / "data" / "phase2a_output"
        out_2a.mkdir(parents=True, exist_ok=True)
        struct_dir = out_2a / "structures"
        struct_dir.mkdir(parents=True, exist_ok=True)
        sim = MDSimulator(run_mode=self.config.run_mode)
        results = []
        for i in range(n):
            length = 60 + (i % 41)
            seq, structure = generate_candidate(length=length, type="B")
            ns = self.config.md_ns_quick if n <= 10 else self.config.md_ns_2a
            md_result = sim.run(structure, struct_dir, f"ufoe_{i:04d}", length_ns=ns)
            results.append({
                "id": f"ufoe_{i:04d}",
                "group": "UFoE",
                "sequence": seq,
                "fold_success": md_result.fold_success,
                "final_rmsd": md_result.final_rmsd,
                "energy_score": md_result.energy_score,
                "output_pdb_path": md_result.output_pdb_path,
                "zone_EC": md_result.zone_EC,
                "zone_TZ": md_result.zone_TZ,
                "zone_BZ": md_result.zone_BZ,
                "runtime_sec": md_result.runtime_sec,
                "phase": "2a",
            })
        csv_path = out_2a / "phase2a_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "id", "group", "sequence", "fold_success", "final_rmsd", "energy_score",
                "output_pdb_path", "zone_EC", "zone_TZ", "zone_BZ", "runtime_sec", "phase",
            ], extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        elim = run_elimination_loop(csv_path, structures_dir=struct_dir)
        return {
            "phase2a": results,
            "phase3": elim,
            "n": n,
            "fold_success_rate": sum(1 for r in results if r.get("fold_success")) / n if n else 0,
            "K_over_M": elim["K_over_M"],
            "self_consistent": elim.get("self_consistent", False),
        }

    def save_results(self, results: Dict[str, Any]) -> List[Path]:
        """결과를 data/pipeline_output/ 에 저장."""
        out = self.root / "data" / "pipeline_output"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        if "phase2a" in results:
            p = out / "phase2a_results.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "id", "group", "sequence", "fold_success", "final_rmsd", "energy_score",
                    "output_pdb_path", "zone_EC", "zone_TZ", "zone_BZ", "runtime_sec", "phase",
                ], extrasaction="ignore")
                w.writeheader()
                w.writerows(results["phase2a"])
            paths.append(p)
        if "phase3" in results:
            from .elimination_loop import write_phase3_report
            p = out / "phase3_report.csv"
            write_phase3_report(results["phase3"], p)
            paths.append(p)
        return paths
