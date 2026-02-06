#!/bin/bash
# UFoE Full 실험 — AWS/GCP 인스턴스 환경 설정
# Ubuntu 22.04 기준. GPU 인스턴스 권장 (OpenMM, ESMFold 로컬 사용 시)

set -e

echo "=== UFoE 환경 설정 ==="

# Python 3.10+
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv

# 가상환경 (선택)
# python3 -m venv venv && source venv/bin/activate

pip install --upgrade pip
pip install biopython numpy scipy pandas requests pytest

# OpenMM (Full MD)
pip install openmm

# ESMFold 로컬 사용 시 (선택, 용량 큼)
# pip install 'fair-esm[esmfold]'
# 또는 API만 사용 시 생략

# 프로젝트 디렉터리
mkdir -p /opt/ufoe 2>/dev/null || true
echo "설치 완료. 프로젝트를 /opt/ufoe 또는 작업 디렉터리에 배치한 뒤 run_full.sh 실행."
