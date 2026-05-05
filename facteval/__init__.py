"""FactEval – Find exactly which parts of your LLM output are hallucinated."""

# Suppress known harmless warnings from dependencies before any imports
import os as _os
import sys as _sys
import warnings as _warnings
import logging as _logging
import contextlib as _contextlib
import io as _io

# Suppress safetensors / accelerate noise
_os.environ.setdefault("SAFETENSORS_LOG_LEVEL", "error")
_os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")
_logging.getLogger("safetensors").setLevel(_logging.ERROR)
_logging.getLogger("accelerate").setLevel(_logging.ERROR)

# Suppress HF Hub unauthenticated request warnings
_logging.getLogger("huggingface_hub.utils._http").setLevel(_logging.ERROR)
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

# Suppress transformers info-level noise
_logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)
_logging.getLogger("transformers.generation.configuration_utils").setLevel(_logging.ERROR)

# Suppress FutureWarning about clean_up_tokenization_spaces
_warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


@_contextlib.contextmanager
def suppress_loading_noise():
    """Suppress stdout + stderr noise during model loading (LOAD REPORT, sharding info)."""
    old_stdout, old_stderr = _sys.stdout, _sys.stderr
    _sys.stdout = _io.StringIO()
    _sys.stderr = _io.StringIO()
    try:
        yield
    finally:
        _sys.stdout = old_stdout
        _sys.stderr = old_stderr


# Backward compat alias
suppress_stdout = suppress_loading_noise


# ── Public API ───────────────────────────────────────────────────────────────
from facteval.core import check, verify
from facteval.models import Claim, Evidence, ClaimWithEvidence
from facteval.verifier import FactLabel, VerificationResult

__version__ = "0.1.0"
__all__ = [
    "check",
    "verify",
    "Claim",
    "Evidence",
    "ClaimWithEvidence",
    "FactLabel",
    "VerificationResult",
    "suppress_loading_noise",
]
