from fastapi import APIRouter, HTTPException
from rdkit import Chem

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.agents.base import FeatureAgent
from molfusion_backend.agents.registry import UnknownAgentError
from molfusion_backend.api.schemas import (
    AgentMetadata,
    ComputeRequest,
    ComputeResponse,
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

        # Generic (not agent-specific) computation-failure handling: any
        # FeatureAgent.compute() may raise ValueError to signal it could
        # not produce a result for this specific molecule (e.g. SELFIES
        # encoding failing under the pinned semantic constraints), even
        # though RDKit itself parsed the SMILES successfully.
        #
        # `valid` means molecular/SMILES validity, not "every requested
        # representation succeeded" -- a molecule RDKit parsed fine stays
        # valid=True even if one of its representations could not be
        # computed. The failure is instead reported via `error` (populated)
        # with features=[], distinguishing it from an RDKit parse failure
        # (valid=False) by the value of `valid` alone.
        #
        # This is intentionally all-or-nothing at the molecule level for
        # this phase: if any requested agent fails, no features are
        # returned for that molecule (not even from agents that would have
        # succeeded), because MoleculeResult has no schema for per-agent
        # partial success yet. Modeling partial success (succeeded
        # features alongside a separate failure list) would need a future
        # API revision to MoleculeResult -- not introduced here.
        try:
            features = [_compute_feature_output(agent, mol) for agent in agents]
        except ValueError as exc:
            results.append(
                MoleculeResult(smiles=smiles, valid=True, error=str(exc), features=[])
            )
            continue

        results.append(
            MoleculeResult(smiles=smiles, valid=True, error=None, features=features)
        )

    return ComputeResponse(results=results)
