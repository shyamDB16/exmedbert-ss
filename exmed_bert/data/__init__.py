from .data_exceptions import NonePatient, PatientNotValid
from .dataset import (
    AgeDict,
    CodeDict,
    EndpointDict,
    ExtendedPatientDataset,
    Patient,
    PatientDataset,
    SexDict,
    StateDict,
)

# Optionally import streaming dataset if available
try:
    from .streaming_dataset import StreamingPatientDataset, create_streaming_datasets
    __all__ = [
        "NonePatient", "PatientNotValid",
        "AgeDict", "CodeDict", "EndpointDict", "ExtendedPatientDataset",
        "Patient", "PatientDataset", "SexDict", "StateDict",
        "StreamingPatientDataset", "create_streaming_datasets"
    ]
except ImportError:
    __all__ = [
        "NonePatient", "PatientNotValid",
        "AgeDict", "CodeDict", "EndpointDict", "ExtendedPatientDataset",
        "Patient", "PatientDataset", "SexDict", "StateDict"
    ]
