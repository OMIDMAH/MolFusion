import { API_BASE_URL } from "../config";
import type {
  AgentMetadata,
  AgentValueType,
  ComputeResponse,
  FeatureVector,
  HealthResponse,
  MoleculeResult,
  ValidateResponse,
  ValidationResult,
} from "../types/api";

/** Raised for any failure talking to the backend: network, HTTP, or shape errors. */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function extractErrorMessage(response: Response): Promise<string> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return `${response.status} ${response.statusText}`;
  }

  if (isRecord(body) && typeof body.detail === "string") {
    return body.detail;
  }

  // FastAPI/Pydantic 422 validation errors: { detail: [{ msg, loc, ... }, ...] }
  if (isRecord(body) && Array.isArray(body.detail)) {
    const messages = body.detail
      .map((entry) => (isRecord(entry) && typeof entry.msg === "string" ? entry.msg : null))
      .filter((msg): msg is string => msg !== null);
    if (messages.length > 0) {
      return messages.join("; ");
    }
  }

  return `${response.status} ${response.statusText}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(
      `Could not reach the MolFusion backend at ${API_BASE_URL}. Is it running?`,
    );
  }

  if (!response.ok) {
    throw new ApiError(await extractErrorMessage(response), response.status);
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError(
      "Backend returned a response that was not valid JSON.",
      response.status,
    );
  }
}

function expectString(value: unknown, field: string): string {
  if (typeof value !== "string") {
    throw new ApiError(`Backend response is malformed: expected string for "${field}".`);
  }
  return value;
}

function expectNumber(value: unknown, field: string): number {
  if (typeof value !== "number") {
    throw new ApiError(`Backend response is malformed: expected number for "${field}".`);
  }
  return value;
}

function expectBoolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") {
    throw new ApiError(`Backend response is malformed: expected boolean for "${field}".`);
  }
  return value;
}

function expectNullableString(value: unknown, field: string): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return expectString(value, field);
}

function expectArray(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ApiError(`Backend response is malformed: expected an array for "${field}".`);
  }
  return value;
}

function expectValueType(value: unknown, field: string): AgentValueType {
  if (value !== "binary" && value !== "continuous") {
    throw new ApiError(
      `Backend response is malformed: expected "binary" or "continuous" for "${field}", got ${JSON.stringify(value)}.`,
    );
  }
  return value;
}

function parseAgentMetadata(value: unknown): AgentMetadata {
  if (!isRecord(value)) {
    throw new ApiError("Backend returned malformed agent metadata.");
  }
  const featureNames =
    value.feature_names === null || value.feature_names === undefined
      ? null
      : expectArray(value.feature_names, "feature_names").map((entry, index) =>
          expectString(entry, `feature_names[${index}]`),
        );
  return {
    id: expectString(value.id, "id"),
    name: expectString(value.name, "name"),
    version: expectString(value.version, "version"),
    output_dim: expectNumber(value.output_dim, "output_dim"),
    requires_3d: expectBoolean(value.requires_3d, "requires_3d"),
    value_type: expectValueType(value.value_type, "value_type"),
    feature_names: featureNames,
  };
}

function parseValidationResult(value: unknown): ValidationResult {
  if (!isRecord(value)) {
    throw new ApiError("Backend returned malformed validation result.");
  }
  return {
    smiles: expectString(value.smiles, "smiles"),
    valid: expectBoolean(value.valid, "valid"),
    error: expectNullableString(value.error, "error"),
  };
}

function parseFeatureVector(value: unknown): FeatureVector {
  if (!isRecord(value)) {
    throw new ApiError("Backend returned malformed feature vector.");
  }
  const values = expectArray(value.values, "values").map((entry, index) =>
    expectNumber(entry, `values[${index}]`),
  );
  return {
    agent_id: expectString(value.agent_id, "agent_id"),
    agent_version: expectString(value.agent_version, "agent_version"),
    values,
    dim: expectNumber(value.dim, "dim"),
  };
}

function parseMoleculeResult(value: unknown): MoleculeResult {
  if (!isRecord(value)) {
    throw new ApiError("Backend returned malformed molecule result.");
  }
  return {
    smiles: expectString(value.smiles, "smiles"),
    valid: expectBoolean(value.valid, "valid"),
    error: expectNullableString(value.error, "error"),
    features: expectArray(value.features, "features").map(parseFeatureVector),
  };
}

export async function getHealth(): Promise<HealthResponse> {
  const data = await request<unknown>("/health");
  if (!isRecord(data) || typeof data.status !== "string") {
    throw new ApiError("Backend returned a malformed /health response.");
  }
  return { status: data.status };
}

export async function getAgents(): Promise<AgentMetadata[]> {
  const data = await request<unknown>("/agents");
  return expectArray(data, "agents").map(parseAgentMetadata);
}

export async function validateMolecules(smiles: string[]): Promise<ValidateResponse> {
  const data = await request<unknown>("/molecules/validate", {
    method: "POST",
    body: JSON.stringify({ smiles }),
  });
  if (!isRecord(data)) {
    throw new ApiError("Backend returned a malformed /molecules/validate response.");
  }
  return { results: expectArray(data.results, "results").map(parseValidationResult) };
}

export async function computeFeatures(
  smiles: string[],
  agentIds: string[],
): Promise<ComputeResponse> {
  const data = await request<unknown>("/features/compute", {
    method: "POST",
    body: JSON.stringify({ smiles, agent_ids: agentIds }),
  });
  if (!isRecord(data)) {
    throw new ApiError("Backend returned a malformed /features/compute response.");
  }
  return { results: expectArray(data.results, "results").map(parseMoleculeResult) };
}
