"models.py"

from typing import Mapping
import enum
import pydantic


class Language(enum.Enum):
    DE = "German"
    EN = "English"

class StatementType(enum.Enum):
    DESCRIPTIVE = "descriptive"
    ASCRIPTIVE = "ascriptive"
    NORMATIVE = "normative"


class Document(pydantic.BaseModel):
    text: str
    uid: str
    metadata: dict = {}


class CheckedClaim(pydantic.BaseModel):
    text: str
    negation: str | None = None
    uid: str
    statement_type: StatementType | None = None
    n_evidence: int | None = None
    average_confirmation: float | None = None
    evidential_uncertainty: float | None = None
    verbalized_confirmation: str | None = None
    documents: list[Document] | None = None
    confirmation_by_document: Mapping[str, float] | None = None
    metadata: dict = {}
