from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str


class AgentMetadata(BaseModel):
    """Metadata for one registered FeatureAgent, as reported by the live registry."""

    id: str
    name: str
    version: str
    # None only for "sequence" agents: sequence length is a per-molecule
    # result property, not a fixed agent-level dimension.
    output_dim: int | None
    requires_3d: bool
    value_type: Literal["binary", "count", "continuous", "categorical"]
    output_structure: Literal["vector", "sequence"]
    feature_names: list[str] | None = None


class ValidateRequest(BaseModel):
    smiles: list[str] = Field(min_length=1)


class ValidationResult(BaseModel):
    smiles: str
    valid: bool
    error: str | None = None


class ValidateResponse(BaseModel):
    results: list[ValidationResult]


class ComputeRequest(BaseModel):
    smiles: list[str] = Field(min_length=1)
    agent_ids: list[str] = Field(min_length=1)


class VectorFeatureOutput(BaseModel):
    """Output of a "vector" agent: a fixed-size numeric array."""

    output_structure: Literal["vector"] = "vector"
    agent_id: str
    agent_version: str
    values: list[float]
    dim: int

    @model_validator(mode="after")
    def _check_dim_matches_values(self) -> "VectorFeatureOutput":
        if self.dim != len(self.values):
            raise ValueError(
                f"dim ({self.dim}) does not match len(values) ({len(self.values)})"
            )
        return self


class SequenceFeatureOutput(BaseModel):
    """Output of a "sequence" agent: a variable-length ordered token list.

    Deliberately has no `values`/`dim` fields -- a token sequence is not a
    numeric vector, and fabricating dim=len(tokens) as if it meant the same
    thing as a vector's dim would misrepresent the representation.
    """

    output_structure: Literal["sequence"] = "sequence"
    agent_id: str
    agent_version: str
    tokens: list[str]
    length: int

    @model_validator(mode="after")
    def _check_length_matches_tokens(self) -> "SequenceFeatureOutput":
        if self.length != len(self.tokens):
            raise ValueError(
                f"length ({self.length}) does not match len(tokens) ({len(self.tokens)})"
            )
        return self


# Discriminated union: the wire format is chosen generically by each
# feature's own `output_structure`, never by inspecting agent_id.
FeatureOutput = Annotated[
    Union[VectorFeatureOutput, SequenceFeatureOutput],
    Field(discriminator="output_structure"),
]


class MoleculeResult(BaseModel):
    smiles: str
    valid: bool
    error: str | None = None
    features: list[FeatureOutput] = Field(default_factory=list)


class ComputeResponse(BaseModel):
    results: list[MoleculeResult]
