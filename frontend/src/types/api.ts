export interface HealthResponse {
  status: string;
}

/** "binary": every value is 0 or 1 (bit-vector fingerprints).
 * "count": non-negative integer counts (RDKit fragment descriptors).
 * "continuous": real-valued (physicochemical descriptors, ErG, ...).
 * "categorical": discrete symbols from a vocabulary (SELFIES tokens). */
export type AgentValueType = "binary" | "count" | "continuous" | "categorical";

/** "vector": a fixed-size, ordered, numeric array. "sequence": a
 * variable-length, ordered sequence of string tokens (e.g. SELFIES) -- has
 * no fixed output_dim. */
export type AgentOutputStructure = "vector" | "sequence";

export interface AgentMetadata {
  id: string;
  name: string;
  version: string;
  /** Fixed dimension for "vector" agents; null for "sequence" agents,
   * whose length varies per molecule (see SequenceFeatureOutput.length). */
  output_dim: number | null;
  requires_3d: boolean;
  value_type: AgentValueType;
  output_structure: AgentOutputStructure;
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

/** Output of a "vector" agent: a fixed-size numeric array. */
export interface VectorFeatureOutput {
  output_structure: "vector";
  agent_id: string;
  agent_version: string;
  values: number[];
  dim: number;
}

/** Output of a "sequence" agent: a variable-length ordered token list.
 * Deliberately has no values/dim -- a token sequence is not a numeric
 * vector. */
export interface SequenceFeatureOutput {
  output_structure: "sequence";
  agent_id: string;
  agent_version: string;
  tokens: string[];
  length: number;
}

/** Discriminated union: consumers must branch on `output_structure`, never
 * on agent_id, to stay generic across future sequence/vector agents. */
export type FeatureOutput = VectorFeatureOutput | SequenceFeatureOutput;

/** One selected agent that could not produce a representation for a
 * molecule the backend parsed successfully. Generic across vector and
 * sequence agents -- consumers report "this agent failed" without needing
 * to know which kind it is. */
export interface FeatureComputationError {
  agent_id: string;
  agent_version: string;
  error: string;
}

/** Three distinct outcomes, which must not be conflated:
 *  - valid=false with `error` set: the SMILES itself could not be parsed.
 *  - valid=true with entries in `feature_errors`: the molecule is fine but
 *    some representations failed. Agents that succeeded are still in
 *    `features`.
 *  - valid=true with a feature present: success, including legitimately
 *    empty results such as an all-zero TF-IDF vector.
 * `error` is molecule-level only; a per-agent failure never populates it. */
export interface MoleculeResult {
  smiles: string;
  valid: boolean;
  error: string | null;
  features: FeatureOutput[];
  feature_errors: FeatureComputationError[];
}

export interface ComputeResponse {
  results: MoleculeResult[];
}
