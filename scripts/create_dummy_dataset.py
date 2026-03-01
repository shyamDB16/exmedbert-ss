#!/usr/bin/env python
"""
Create a minimal dummy PatientDataset for smoke-testing pre-training.

No real patient data. Generates small vocabs and synthetic patient records
so you can run the minimal GPU test without confidential data.

Usage:
    python scripts/create_dummy_dataset.py --output-dir pretrain_data

Then run:
    TRAIN_PT=pretrain_data/dummy_train.pt VAL_PT=pretrain_data/dummy_val.pt \\
    ./scripts/run_minimal_pretrain_test.sh
"""

import logging
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import torch
import typer
from typer import Option

from exmed_bert.data.dataset import PatientDataset
from exmed_bert.data.encoding import AgeDict, CodeDict, EndpointDict, SexDict, StateDict
from exmed_bert.data.patient import Patient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimal code sets (no real codes)
DUMMY_ICD = [f"I{i:02d}" for i in range(1, 16)]  # 15 codes
DUMMY_ATC = [f"A{i:02d}A{i:02d}" for i in range(1, 11)]  # 10 codes
DUMMY_STATES = ["CA", "NY"]


def make_dummy_patient(
    patient_id: int,
    code_dict: CodeDict,
    seed: Optional[int] = None,
    num_diagnoses: int = 6,
    num_drugs: int = 3,
    base_date: date = date(2020, 1, 1),
) -> dict:
    """One synthetic patient dict (diagnoses, drugs, dates, demographics)."""
    rng = random.Random(seed if seed is not None else patient_id)
    diagnoses = rng.choices(DUMMY_ICD, k=num_diagnoses)
    drugs = rng.choices(DUMMY_ATC, k=num_drugs)
    days_off = [rng.randint(0, 365) for _ in range(num_diagnoses)]
    diag_dates = [(base_date + timedelta(days=d)).strftime("%Y%m%d") for d in days_off]
    drug_days = [rng.randint(0, 365) for _ in range(num_drugs)]
    drug_dates = [(base_date + timedelta(days=d)).strftime("%Y%m%d") for d in drug_days]
    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses,
        "drugs": drugs,
        "procedures": [],
        "diagnosis_dates": diag_dates,
        "prescription_dates": drug_dates,
        "procedure_dates": [],
        "birth_year": 1950 + rng.randint(0, 50),
        "sex": rng.choice(["Male", "Female"]),
        "patient_state": rng.choice(DUMMY_STATES),
    }


def main(
    output_dir: Path = Option(Path("pretrain_data"), "--output-dir", "-o", help="Output directory"),
    num_patients: int = Option(40, "--num-patients", "-n", help="Number of dummy patients"),
    max_length: int = Option(128, "--max-length", help="Max sequence length"),
    train_ratio: float = Option(0.8, "--train-ratio", help="Train fraction"),
    seed: int = Option(42, "--seed", help="Random seed"),
):
    """Create dummy_train.pt and dummy_val.pt in output_dir."""
    from datetime import datetime

    random.seed(seed)

    logger.info("Building minimal vocabs...")
    code_dict = CodeDict(
        atc_codes=DUMMY_ATC,
        icd_codes=DUMMY_ICD,
        cpt_codes=[],
        rx_to_atc_map=None,
    )
    age_dict = AgeDict(max_age=110, min_age=0, binsize=1.0)
    sex_dict = SexDict()
    state_dict = StateDict(states=DUMMY_STATES)
    endpoint_dict = EndpointDict([])

    logger.info("Generating %s dummy patient records...", num_patients)
    patients_data = [
        make_dummy_patient(i, code_dict, seed=seed + i) for i in range(num_patients)
    ]

    logger.info("Converting to Patient objects...")
    patients = []
    for i, pd in enumerate(patients_data):
        try:
            diagnosis_dates = [datetime.strptime(d, "%Y%m%d").date() for d in pd["diagnosis_dates"]]
            prescription_dates = [datetime.strptime(d, "%Y%m%d").date() for d in pd["prescription_dates"]]
            procedure_dates = []
            patient = Patient(
                patient_id=pd["patient_id"],
                diagnoses=pd["diagnoses"],
                drugs=pd["drugs"],
                procedures=pd["procedures"],
                diagnosis_dates=diagnosis_dates,
                prescription_dates=prescription_dates,
                procedure_dates=procedure_dates,
                birth_year=pd["birth_year"],
                sex=pd["sex"],
                patient_state=pd["patient_state"],
                max_length=max_length,
                code_embed=code_dict,
                sex_embed=sex_dict,
                age_embed=age_dict,
                state_embed=state_dict,
                mask_drugs=True,
                delete_temporary_variables=True,
                split_sequence=True,
                drop_duplicates=True,
                converted_codes=True,
                convert_rxcui_to_atc=False,
                keep_min_unmasked=1,
                max_masked_tokens=20,
                masked_lm_prob=0.15,
                truncate="right",
                dynamic_masking=True,
                min_observations=5,
                age_usage="year",
                use_cls=False,
                use_sep=False,
                endpoint_labels=None,
            )
            patients.append(patient)
        except Exception as e:
            logger.debug("Skip patient %s: %s", pd["patient_id"], e)

    if len(patients) < 2:
        raise RuntimeError("Need at least 2 valid patients; try increasing --num-patients")

    # Train/val split
    random.shuffle(patients)
    split_idx = int(len(patients) * train_ratio)
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "dummy_train.pt"
    val_path = output_dir / "dummy_val.pt"

    train_dataset = PatientDataset(
        code_embed=code_dict,
        age_embed=age_dict,
        sex_embed=sex_dict,
        state_embed=state_dict,
        endpoint_dict=endpoint_dict,
        patients=train_patients,
        max_length=max_length,
        do_eval=True,
        mask_substances=True,
        dynamic_masking=True,
    )
    val_dataset = PatientDataset(
        code_embed=code_dict,
        age_embed=age_dict,
        sex_embed=sex_dict,
        state_embed=state_dict,
        endpoint_dict=endpoint_dict,
        patients=val_patients,
        max_length=max_length,
        do_eval=True,
        mask_substances=True,
        dynamic_masking=True,
    )

    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)

    logger.info("Done. Saved %s (%s) and %s (%s).",
                train_path, len(train_dataset), val_path, len(val_dataset))
    logger.info("Run: TRAIN_PT=%s VAL_PT=%s ./scripts/run_minimal_pretrain_test.sh",
                train_path, val_path)


if __name__ == "__main__":
    typer.run(main)
