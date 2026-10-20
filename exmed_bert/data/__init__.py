"""Data contracts with lazy access to optional legacy dataset implementations."""

from importlib import import_module

from .data_exceptions import NonePatient, PatientNotValid
from .encoding import AgeDict, CodeDict, EndpointDict, SexDict, StateDict

__all__ = [
    "NonePatient",
    "PatientNotValid",
    "AgeDict",
    "CodeDict",
    "EndpointDict",
    "SexDict",
    "StateDict",
    "Patient",
    "PatientDataset",
    "ExtendedPatientDataset",
    "StreamingPatientDataset",
    "create_streaming_datasets",
]


def __getattr__(name):
    if name == "Patient":
        return getattr(import_module("exmed_bert.data.patient"), name)
    if name in {"PatientDataset", "ExtendedPatientDataset"}:
        return getattr(import_module("exmed_bert.data.dataset"), name)
    if name in {"StreamingPatientDataset", "create_streaming_datasets"}:
        return getattr(import_module("exmed_bert.data.streaming_dataset"), name)
    raise AttributeError(name)
