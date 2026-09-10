from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str


class AgentAvailability(BaseModel):
    """Whether one agent can currently accept compute requests.

    Answers "can this agent run at all here?", never "will this molecule
    succeed?". A molecule-specific failure is reported per molecule in
    MoleculeResult.feature_errors and leaves the agent available.

    `code` is a stable, generic category (`artifact_missing`,
    `artifact_checksum_error`, ...) so a client can branch on the kind of
    problem without knowing which representation the agent implements.
    Both fields are null when available.
    """

    available: bool
    code: str | None = None
    message: str | None = None


class UnavailableAgent(BaseModel):
    """One selected agent that cannot run, as reported by request preflight."""

    agent_id: str
    code: str | None = None
    message: str | None = None


class UnavailableAgentsDetail(BaseModel):
    """Body of the error returned when preflight rejects a compute request.

    Structured rather than a bare string because the caller needs to know
    *which* of the agents it selected are unusable, and a machine-readable
    code lets a client distinguish "deploy the artifact" from "fix the
    configuration" without parsing prose.
    """

    message: str
    agents: list[UnavailableAgent]


class AgentMetadata(BaseModel):
    """Metadata for one registered FeatureAgent, as reported by the live registry.

    Identity (id, version, output shape) is static. `availability` is not:
    it is evaluated per request, so an agent whose prerequisite is missing
    stays listed here -- a consumer needs to know the representation exists
    and is currently unusable, which is different from it not existing.
    """

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
    # Added in Phase 5I. Defaults to available so a metadata dict built
    # without it (e.g. in a test fixture) still validates.
    availability: AgentAvailability = Field(default_factory=lambda: AgentAvailability(available=True))


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


class FeatureComputationError(BaseModel):
    """One selected agent that could not produce a representation for a
    molecule RDKit parsed successfully.

    Deliberately one generic model rather than per-representation variants:
    a consumer should be able to report "this agent failed" without knowing
    whether the agent emits vectors or token sequences, and adding
    `selfies_error`/`tfidf_error` fields would push representation
    knowledge into every client.

    `error` carries the agent's own message and nothing else -- no
    traceback, no exception class, no internal state. Callers get enough to
    identify what failed and why; they do not get the server's internals.
    """

    agent_id: str
    agent_version: str
    error: str


class MoleculeResult(BaseModel):
    """One molecule's outcome across every selected agent.

    Three distinct situations, and conflating any two of them loses real
    information:

      * `valid=False` with `error` set -- RDKit could not parse the SMILES.
        No representation was attempted, so `features` and `feature_errors`
        are both empty.
      * `valid=True` with entries in `feature_errors` -- the molecule is
        fine, but one or more representations could not be computed for it.
        Every agent that *did* succeed still appears in `features`.
      * `valid=True` with a feature present and no error -- success, which
        includes legitimately empty results such as an all-zero TF-IDF
        vector for a molecule containing no vocabulary n-gram. That is a
        computed representation, not a failure.

    `error` is therefore molecule-level only. A per-agent failure never
    populates it, and never removes another agent's successful output.
    """

    smiles: str
    valid: bool
    error: str | None = None
    features: list[FeatureOutput] = Field(default_factory=list)
    # Added in Phase 5H. Defaults to empty, so a response with no failures
    # is shaped exactly as before and existing consumers are unaffected.
    feature_errors: list[FeatureComputationError] = Field(default_factory=list)


class ComputeResponse(BaseModel):
    results: list[MoleculeResult]
