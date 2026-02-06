#!/usr/bin/env python3
"""
Phase 0 검증: pdb_complexes, pdb_p53, data/pdb_representative 전체 PDB에 5-필터 적용.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

# 전체 PDB 수집
FOLDERS = [
    WORKSPACE / "pdb_complexes",
    WORKSPACE / "pdb_p53",
    ROOT / "data" / "pdb_representative",
]


def main():
    pdb_list = []
    for folder in FOLDERS:
        if folder.exists():
            pdb_list.extend(folder.glob("*.pdb"))
            pdb_list.extend(folder.glob("*.cif"))
    pdb_list = sorted(set(str(p) for p in pdb_list))

    if not pdb_list:
        print("PDB 파일이 없습니다.")
        return 1

    from src.filters.ufoef_filters import batch_validate
    import pandas as pd

    print("Phase 0 필터 적용 중... (n=%d)" % len(pdb_list))
    df = batch_validate(pdb_list)

    def source(path):
        if "pdb_complexes" in path:
            return "pdb_complexes"
        if "pdb_p53" in path:
            return "pdb_p53"
        return "pdb_representative"

    df["source"] = df["path"].map(source)
    n = len(df)
    all_p = df["all_passed"].sum()
    pass_rate = 100 * all_p / n if n else 0

    print()
    print("=== Phase 0 검증 결과 (전체 %d개) ===" % n)
    print("5개 필터 모두 통과: %d / %d (%.1f%%)" % (all_p, n, pass_rate))
    print()
    print("필터별 통과율:")
    for c in sorted(df.columns):
        if c.endswith("_passed"):
            print("  %s: %.1f%%" % (c, 100 * df[c].mean()))
    print()
    print("폴더별 통과:")
    for src in ["pdb_complexes", "pdb_p53", "pdb_representative"]:
        sub = df[df["source"] == src]
        if len(sub) > 0:
            sp = sub["all_passed"].sum()
            print("  %s: %d / %d (%.1f%%)" % (src, sp, len(sub), 100 * sp / len(sub)))

    if "error" in df.columns:
        errs = df.loc[df["error"].notna()]
        if len(errs) > 0:
            print()
            print("오류 발생 %d건:" % len(errs))
            for _, r in errs.iterrows():
                print("  %s: %s" % (r["pdb"], r["error"]))

    out_csv = ROOT / "data" / "phase0_validation_results.csv"
    df.to_csv(out_csv, index=False)
    print()
    print("저장: %s" % out_csv)
    print("80%%+ 목표: %s" % ("달성" if pass_rate >= 80 else "미달성"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
