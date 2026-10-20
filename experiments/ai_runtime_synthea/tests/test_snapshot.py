from datetime import datetime, timedelta, timezone

from snapshot import gap_bucket, patient_sequence, split_for_patient, stable_hash


def test_stable_hash_and_split_are_deterministic():
    assert stable_hash("patient-1", 42) == stable_hash("patient-1", 42)
    assert split_for_patient("patient-1", 42, 0.8) == split_for_patient("patient-1", 42, 0.8)


def test_gap_buckets_cover_boundaries():
    assert [gap_bucket(value) for value in (None, 0.5, 1, 8, 31, 181, 366)] == [0, 1, 2, 3, 4, 5, 6]


def test_patient_sequence_keeps_recent_events_and_restarts_gap_history():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    visits = [
        (start, "e1", ["SNOMED:A", "SNOMED:B"]),
        (start + timedelta(days=10), "e2", ["SNOMED:C", "SNOMED:D"]),
    ]
    codes, gaps = patient_sequence(visits, max_sequence_length=3)
    assert codes == ["SNOMED:B", "SNOMED:C", "SNOMED:D"]
    assert gaps == [0, 3, 3]
