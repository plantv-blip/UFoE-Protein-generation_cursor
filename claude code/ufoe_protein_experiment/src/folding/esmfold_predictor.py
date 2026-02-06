"""
ESMFold 구조 예측 모듈

Phase 2b: UFoE 생성 서열 → ESMFold 구조 예측 → MD 정제

ESMFold는 Meta AI의 단일 서열 기반 단백질 구조 예측 모델로,
MSA 없이 서열만으로 3D 구조를 예측할 수 있다.

의존성:
    - torch
    - transformers (ESMFold)
    또는
    - ESMFold API (requests)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.constants import ESMFOLD_MODEL_NAME, AA_1TO3
from src.utils.pdb_writer import write_pdb_from_coords, read_pdb_coords

logger = logging.getLogger(__name__)


# =============================================================================
# 데이터 구조
# =============================================================================

@dataclass
class ESMFoldPrediction:
    """ESMFold 구조 예측 결과."""

    candidate_id: str
    sequence: str               # 입력 1글자 서열
    n_residues: int

    # 예측 좌표
    ca_coords: np.ndarray       # (N, 3) Cα 좌표
    plddt_scores: np.ndarray    # (N,) per-residue confidence (0-100)

    # 메타데이터
    mean_plddt: float           # 평균 pLDDT
    pdb_path: Path | None = None  # 저장된 PDB 파일 경로
    prediction_time_s: float = 0.0

    # 오류
    success: bool = True
    error_message: str = ""

    metadata: dict = field(default_factory=dict)


# =============================================================================
# ESMFold 예측기
# =============================================================================

class ESMFoldPredictor:
    """ESMFold 기반 구조 예측기.

    두 가지 모드를 지원:
    1. 로컬 모드: transformers 라이브러리를 통한 로컬 추론
    2. API 모드: ESMFold API 서버를 통한 원격 추론

    사용법:
        predictor = ESMFoldPredictor(mode="api")
        result = predictor.predict("MVLSPADKTNVKAAWGK...")
    """

    def __init__(
        self,
        mode: str = "api",
        model_name: str = ESMFOLD_MODEL_NAME,
        api_url: str = "https://api.esmatlas.com/foldSequence/v1/pdb/",
        output_dir: str | Path = "output/esmfold",
        device: str = "auto",
    ):
        """
        Parameters
        ----------
        mode : str — 'local' or 'api'
        model_name : str — HuggingFace 모델 이름 (로컬 모드)
        api_url : str — ESMFold API URL (API 모드)
        output_dir : str or Path
        device : str — 'auto', 'cuda', 'cpu'
        """
        self.mode = mode
        self.model_name = model_name
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_local_model(self):
        """로컬 ESMFold 모델을 로드한다."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, EsmForProteinFolding

            logger.info(f"ESMFold 모델 로딩 중: {self.model_name}")

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = EsmForProteinFolding.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()

            logger.info(f"ESMFold 모델 로드 완료 (device: {self.device})")

        except ImportError:
            raise ImportError(
                "ESMFold 로컬 모드에 torch와 transformers가 필요합니다. "
                "pip install torch transformers"
            )

    def predict(
        self,
        sequence: str,
        candidate_id: str = "unknown",
        save_pdb: bool = True,
    ) -> ESMFoldPrediction:
        """서열에서 3D 구조를 예측한다.

        Parameters
        ----------
        sequence : str — 1글자 아미노산 서열
        candidate_id : str
        save_pdb : bool

        Returns
        -------
        ESMFoldPrediction
        """
        if self.mode == "local":
            return self._predict_local(sequence, candidate_id, save_pdb)
        elif self.mode == "api":
            return self._predict_api(sequence, candidate_id, save_pdb)
        elif self.mode == "mock":
            return self._predict_mock(sequence, candidate_id, save_pdb)
        else:
            raise ValueError(f"알 수 없는 모드: {self.mode}")

    def _predict_local(
        self,
        sequence: str,
        candidate_id: str,
        save_pdb: bool,
    ) -> ESMFoldPrediction:
        """로컬 ESMFold 모델로 예측한다."""
        import torch

        self._load_local_model()

        start = time.time()

        try:
            with torch.no_grad():
                tokenized = self._tokenizer(
                    [sequence],
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                tokenized = {
                    k: v.to(self.device) for k, v in tokenized.items()
                }

                output = self._model(**tokenized)

            # Cα 좌표 추출 (atom37 표현의 CA 인덱스 = 1)
            positions = output["positions"][-1]  # 마지막 recycling 단계
            ca_coords = positions[0, :, 1, :].cpu().numpy()  # (N, 3)

            # pLDDT 추출
            plddt = output["plddt"][0].cpu().numpy()  # (N,)

            elapsed = time.time() - start

            # PDB 저장
            pdb_path = None
            if save_pdb:
                residue_names = [
                    AA_1TO3.get(aa, "UNK") for aa in sequence
                ]
                pdb_path = write_pdb_from_coords(
                    self.output_dir / f"{candidate_id}_esmfold.pdb",
                    ca_coords,
                    residue_names,
                    b_factors=plddt,
                )

            return ESMFoldPrediction(
                candidate_id=candidate_id,
                sequence=sequence,
                n_residues=len(sequence),
                ca_coords=ca_coords,
                plddt_scores=plddt,
                mean_plddt=float(np.mean(plddt)),
                pdb_path=pdb_path,
                prediction_time_s=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"ESMFold 로컬 예측 오류: {candidate_id} | {e}")
            return ESMFoldPrediction(
                candidate_id=candidate_id,
                sequence=sequence,
                n_residues=len(sequence),
                ca_coords=np.zeros((len(sequence), 3)),
                plddt_scores=np.zeros(len(sequence)),
                mean_plddt=0.0,
                prediction_time_s=elapsed,
                success=False,
                error_message=str(e),
            )

    def _predict_api(
        self,
        sequence: str,
        candidate_id: str,
        save_pdb: bool,
    ) -> ESMFoldPrediction:
        """ESMFold API를 통해 예측한다."""
        import requests

        start = time.time()

        try:
            # API 요청
            response = requests.post(
                self.api_url,
                data=sequence,
                headers={"Content-Type": "text/plain"},
                timeout=300,
            )
            response.raise_for_status()

            pdb_text = response.text
            elapsed = time.time() - start

            # PDB 텍스트에서 좌표 추출
            pdb_path = self.output_dir / f"{candidate_id}_esmfold.pdb"
            pdb_path.parent.mkdir(parents=True, exist_ok=True)

            with open(pdb_path, "w") as f:
                f.write(pdb_text)

            # Cα 좌표 및 pLDDT 추출
            ca_coords, plddt = self._parse_esmfold_pdb(pdb_text)

            return ESMFoldPrediction(
                candidate_id=candidate_id,
                sequence=sequence,
                n_residues=len(sequence),
                ca_coords=ca_coords,
                plddt_scores=plddt,
                mean_plddt=float(np.mean(plddt)) if len(plddt) > 0 else 0.0,
                pdb_path=pdb_path if save_pdb else None,
                prediction_time_s=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"ESMFold API 오류: {candidate_id} | {e}")
            return ESMFoldPrediction(
                candidate_id=candidate_id,
                sequence=sequence,
                n_residues=len(sequence),
                ca_coords=np.zeros((len(sequence), 3)),
                plddt_scores=np.zeros(len(sequence)),
                mean_plddt=0.0,
                prediction_time_s=elapsed,
                success=False,
                error_message=str(e),
            )

    def _predict_mock(
        self,
        sequence: str,
        candidate_id: str,
        save_pdb: bool,
    ) -> ESMFoldPrediction:
        """테스트용 합성 예측 결과를 생성한다."""
        logger.warning(f"Mock ESMFold 예측: {candidate_id}")

        n = len(sequence)
        rng = np.random.default_rng(hash(candidate_id) % (2**31))

        # 합성 Cα 좌표: 나선형 구조 + 노이즈
        t = np.linspace(0, 4 * np.pi, n)
        ca_coords = np.column_stack([
            5.0 * np.cos(t) + rng.normal(0, 0.5, n),
            5.0 * np.sin(t) + rng.normal(0, 0.5, n),
            1.5 * t / (2 * np.pi) + rng.normal(0, 0.5, n),
        ])

        # 합성 pLDDT: 60-90 범위
        plddt = rng.uniform(60.0, 90.0, n)

        pdb_path = None
        if save_pdb:
            residue_names = [AA_1TO3.get(aa, "ALA") for aa in sequence]
            pdb_path = write_pdb_from_coords(
                self.output_dir / f"{candidate_id}_esmfold.pdb",
                ca_coords,
                residue_names,
                b_factors=plddt,
            )

        return ESMFoldPrediction(
            candidate_id=candidate_id,
            sequence=sequence,
            n_residues=n,
            ca_coords=ca_coords,
            plddt_scores=plddt,
            mean_plddt=float(np.mean(plddt)),
            pdb_path=pdb_path,
            prediction_time_s=0.01,
            metadata={"mock": True},
        )

    @staticmethod
    def _parse_esmfold_pdb(pdb_text: str) -> tuple[np.ndarray, np.ndarray]:
        """ESMFold PDB 출력에서 Cα 좌표와 pLDDT(B-factor)를 추출한다."""
        coords = []
        plddts = []

        for line in pdb_text.split("\n"):
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    bfactor = float(line[60:66])
                    coords.append([x, y, z])
                    plddts.append(bfactor)

        return np.array(coords), np.array(plddts)

    def batch_predict(
        self,
        sequences: list[tuple[str, str]],
        save_pdb: bool = True,
    ) -> list[ESMFoldPrediction]:
        """여러 서열을 배치로 예측한다.

        Parameters
        ----------
        sequences : list of (candidate_id, sequence)
        save_pdb : bool

        Returns
        -------
        list[ESMFoldPrediction]
        """
        results = []
        for i, (cid, seq) in enumerate(sequences):
            logger.info(f"ESMFold 예측 {i + 1}/{len(sequences)}: {cid}")
            result = self.predict(seq, cid, save_pdb)
            results.append(result)

        return results
