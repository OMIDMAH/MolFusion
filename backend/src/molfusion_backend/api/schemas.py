from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str


class AgentMetadata(BaseModel):
    """Metadata for one registered FeatureAgent, as reported by the live registry."""

    id: str
    name: str
    version: str
    output_dim: int
    requires_3d: bool
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


class FeatureVector(BaseModel):
    agent_id: str
    agent_version: str
    values: list[float]
    dim: int

    @model_validator(mode="after")
    def _check_dim_matches_values(self) -> "FeatureVector":
        if self.dim != len(self.values):
            raise ValueError(
                f"dim ({self.dim}) does not match len(values) ({len(self.values)})"
            )
        return self


class MoleculeResult(BaseModel):
    smiles: str
    valid: bool
    error: str | None = None
    features: list[FeatureVector] = Field(default_factory=list)


class ComputeResponse(BaseModel):
    results: list[MoleculeResult]
