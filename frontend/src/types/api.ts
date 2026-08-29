export interface HealthResponse {
  status: string;
}

export interface AgentMetadata {
  id: string;
  name: string;
  version: string;
  output_dim: number;
  requires_3d: boolean;
  /** Per-feature names (e.g. RDKit descriptor names), when the agent exposes them. */
  feature_names: string[] | null;
}

export interface ValidateRequest {
  smiles: string[];
}

export interface ValidationResult {
  smiles: string;
  valid: boolean;
  error: string | null;
}

export interface ValidateResponse {
  results: ValidationResult[];
}

export interface ComputeRequest {
  smiles: string[];
  agent_ids: string[];
}

export interface FeatureVector {
  agent_id: string;
  agent_version: string;
  values: number[];
  dim: number;
}

export interface MoleculeResult {
  smiles: string;
  valid: boolean;
  error: string | null;
  features: FeatureVector[];
}

export interface ComputeResponse {
  results: MoleculeResult[];
}
