from fastapi import APIRouter, HTTPException
from rdkit import Chem

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.agents.registry import UnknownAgentError
from molfusion_backend.api.schemas import (
    AgentMetadata,
    ComputeRequest,
    ComputeResponse,
    FeatureComputationError,
    FeatureOutput,
    HealthResponse,
    MoleculeResult,
    SequenceFeatureOutput,
    ValidateRequest,
    ValidateResponse,
    ValidationResult,
    VectorFeatureOutput,
)
from molfusion_backend.chemistry import parse_smiles

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/agents", response_model=list[AgentMetadata])
def list_agents() -> list[AgentMetadata]:
    return [AgentMetadata(**metadata) for metadata in agent_registry.list_agents()]


@router.post("/molecules/validate", response_model=ValidateResponse)
def validate_molecules(request: ValidateRequest) -> ValidateResponse:
    results = []
    for smiles in request.smiles:
        mol, error = parse_smiles(smiles)
        results.append(ValidationResult(smiles=smiles, valid=mol is not None, error=error))
    return ValidateResponse(results=results)


def _compute_feature_output(agent: FeatureAgent, mol: Chem.Mol) -> FeatureOutput:
    """Dispatch generically on agent.output_structure -- never on agent_id."""
    if agent.output_structure == "vector":
        values = [float(value) for value in agent.compute(mol).tolist()]
        return VectorFeatureOutput(
            agent_id=agent.id,
            agent_version=agent.version,
            values=values,
            dim=len(values),
        )

    if agent.output_structure == "sequence":
        tokens = list(agent.compute(mol))
        return SequenceFeatureOutput(
            agent_id=agent.id,
            agent_version=agent.version,
            tokens=tokens,
            length=len(tokens),
        )

    raise AssertionError(f"Unknown output_structure {agent.output_structure!r} for agent {agent.id!r}")


@router.post("/features/compute", response_model=ComputeResponse)
def compute_features(request: ComputeRequest) -> ComputeResponse:
    agents = []
    for agent_id in request.agent_ids:
        try:
            agents.append(agent_registry.get(agent_id))
        except UnknownAgentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    results = []
    for smiles in request.smiles:
        mol, error = parse_smiles(smiles)
        if mol is None:
            results.append(
                MoleculeResult(smiles=smiles, valid=False, error=error, features=[])
            )
            continue

        # Per-agent isolation: each selected agent is attempted
        # independently, so one representation failing never discards
        # another's successful output.
        #
        # Only ValueError is caught, which is the documented way a
        # FeatureAgent signals "I could not produce a result for this
        # molecule" (SELFIES rejecting a hypervalent atom, the TF-IDF
        # artifact being unloadable, a molecule outside an agent's
        # applicability). Anything else -- a TypeError, an AttributeError,
        # a KeyError -- is a bug rather than a representation outcome, and
        # is deliberately left to propagate: silently reshaping programming
        # errors into routine per-agent failures would make them invisible
        # exactly where they most need to be seen.
        #
        # `valid` remains molecular/SMILES validity alone. A molecule RDKit
        # parsed successfully stays valid=True with error=None even when
        # every one of its representations fails; the failures are reported
        # per agent instead. Both lists follow the requested agent order,
        # so output ordering never depends on which agent happened to fail.
        features: list[FeatureOutput] = []
        feature_errors: list[FeatureComputationError] = []
        for agent in agents:
            try:
                features.append(_compute_feature_output(agent, mol))
            except ValueError as exc:
                feature_errors.append(
                    FeatureComputationError(
                        agent_id=agent.id, agent_version=agent.version, error=str(exc)
                    )
                )

        results.append(
            MoleculeResult(
                smiles=smiles,
                valid=True,
                error=None,
                features=features,
                feature_errors=feature_errors,
            )
        )

    return ComputeResponse(results=results)
