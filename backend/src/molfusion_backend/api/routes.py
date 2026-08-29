from fastapi import APIRouter, HTTPException

from molfusion_backend.agents import registry as agent_registry
from molfusion_backend.agents.registry import UnknownAgentError
from molfusion_backend.api.schemas import (
    AgentMetadata,
    ComputeRequest,
    ComputeResponse,
    FeatureVector,
    HealthResponse,
    MoleculeResult,
    ValidateRequest,
    ValidateResponse,
    ValidationResult,
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

        features = []
        for agent in agents:
            values = [float(value) for value in agent.compute(mol).tolist()]
            features.append(
                FeatureVector(
                    agent_id=agent.id,
                    agent_version=agent.version,
                    values=values,
                    dim=len(values),
                )
            )
        results.append(
            MoleculeResult(smiles=smiles, valid=True, error=None, features=features)
        )

    return ComputeResponse(results=results)
