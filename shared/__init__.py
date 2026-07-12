"""Wire contract shared by independently deployed Edge and Cloud packages."""

from .protocol import (
    FinalizeRequest,
    InitRequest,
    ProposalRequest,
    ProtocolError,
    VerificationResponse,
)
from .version import PROTOCOL_VERSION

__all__ = [
    "FinalizeRequest",
    "InitRequest",
    "PROTOCOL_VERSION",
    "ProposalRequest",
    "ProtocolError",
    "VerificationResponse",
]
