"""Domain errors raised by the extraction services.

Routes translate these into HTTP responses, so services stay
independent of FastAPI.
"""


class DocumentProcessingError(Exception):
    """Base class for document extraction errors."""


class InvalidDocumentError(DocumentProcessingError):
    """The file is corrupted or is not really the format it claims to be."""


class EncryptedDocumentError(DocumentProcessingError):
    """The document is password-protected."""
