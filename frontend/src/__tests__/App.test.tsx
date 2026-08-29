import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { ApiError } from "../api/client";
import type { AgentMetadata, ComputeResponse, ValidateResponse } from "../types/api";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    getAgents: vi.fn(),
    validateMolecules: vi.fn(),
    computeFeatures: vi.fn(),
  };
});

const { getAgents, validateMolecules, computeFeatures } = await import("../api/client");

const MORGAN_AGENT: AgentMetadata = {
  id: "morgan_ecfp4_1024",
  name: "Morgan ECFP (radius=2, 1024 bits)",
  version: "1.0.0",
  output_dim: 1024,
  requires_3d: false,
  value_type: "binary",
  output_structure: "vector",
  feature_names: null,
};

const ERG_AGENT: AgentMetadata = {
  id: "erg_reduced_graph_315",
  name: "ErG Reduced-Graph Fingerprint",
  version: "1.0.0",
  output_dim: 315,
  requires_3d: false,
  value_type: "continuous",
  output_structure: "vector",
  feature_names: null,
};

const FRAGMENT_AGENT: AgentMetadata = {
  id: "rdkit_fragment_descriptors",
  name: "RDKit Fragment Descriptors",
  version: "1.0.0",
  output_dim: 4,
  requires_3d: false,
  value_type: "count",
  output_structure: "vector",
  feature_names: ["fr_Al_COO", "fr_Ar_COO", "fr_benzene", "fr_ester"],
};

const SELFIES_AGENT: AgentMetadata = {
  id: "selfies_sequence",
  name: "SELFIES Sequence",
  version: "1.0.0",
  output_dim: null,
  requires_3d: false,
  value_type: "categorical",
  output_structure: "sequence",
  feature_names: null,
};

const AGENTS: AgentMetadata[] = [MORGAN_AGENT];

beforeEach(() => {
  vi.mocked(getAgents).mockReset();
  vi.mocked(validateMolecules).mockReset();
  vi.mocked(computeFeatures).mockReset();
});

async function addSmilesViaTextarea(user: ReturnType<typeof userEvent.setup>, smiles: string) {
  const textarea = await screen.findByLabelText(/manual smiles entry/i);
  await user.type(textarea, smiles);
  await user.click(screen.getByRole("button", { name: /add to queue/i }));
}

describe("App", () => {
  it("renders agent metadata dynamically after startup, not hardcoded", async () => {
    vi.mocked(getAgents).mockResolvedValue(AGENTS);
    render(<App />);

    expect(await screen.findByText("morgan_ecfp4_1024")).toBeInTheDocument();
    expect(screen.getByText("Morgan ECFP (radius=2, 1024 bits)")).toBeInTheDocument();
  });

  it("displays a friendly error when the backend is unreachable", async () => {
    vi.mocked(getAgents).mockRejectedValue(
      new ApiError(
        "Could not reach the MolFusion backend at http://127.0.0.1:8000. Is it running?",
      ),
    );
    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /could not reach the molfusion backend/i,
    );
  });

  it("renders invalid molecule state after validation", async () => {
    vi.mocked(getAgents).mockResolvedValue(AGENTS);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [
        { smiles: "NOTASMILES", valid: false, error: "Invalid SMILES: could not be parsed." },
      ],
    } satisfies ValidateResponse);

    const user = userEvent.setup();
    render(<App />);
    await addSmilesViaTextarea(user, "NOTASMILES");

    expect(await screen.findByText("Invalid")).toBeInTheDocument();
    expect(screen.getByText("Invalid SMILES: could not be parsed.")).toBeInTheDocument();
  });

  it("sends the selected SMILES and agent IDs when Compute is clicked", async () => {
    vi.mocked(getAgents).mockResolvedValue(AGENTS);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "CCO", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockResolvedValue({
      results: [
        {
          smiles: "CCO",
          valid: true,
          error: null,
          features: [
            {
              output_structure: "vector",
              agent_id: "morgan_ecfp4_1024",
              agent_version: "1.0.0",
              values: new Array(1024).fill(0),
              dim: 1024,
            },
          ],
        },
      ],
    } satisfies ComputeResponse);

    const user = userEvent.setup();
    render(<App />);
    await addSmilesViaTextarea(user, "CCO");
    await screen.findByText("Valid");

    const agentCheckbox = screen.getByRole("checkbox", { name: /morgan ecfp/i });
    await user.click(agentCheckbox);
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    await waitFor(() =>
      expect(computeFeatures).toHaveBeenCalledWith(["CCO"], ["morgan_ecfp4_1024"]),
    );

    const resultsPanel = await screen.findByTestId("results-panel");
    expect(resultsPanel).toHaveTextContent("1024");
  });

  it("discovers a continuous-valued agent (e.g. ErG) automatically and renders fractional results", async () => {
    vi.mocked(getAgents).mockResolvedValue([MORGAN_AGENT, ERG_AGENT]);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "CC(=O)Oc1ccccc1C(=O)O", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockResolvedValue({
      results: [
        {
          smiles: "CC(=O)Oc1ccccc1C(=O)O",
          valid: true,
          error: null,
          features: [
            {
              output_structure: "vector",
              agent_id: "erg_reduced_graph_315",
              agent_version: "1.0.0",
              values: [0, 0.3, 1.6, 0.9],
              dim: 315,
            },
          ],
        },
      ],
    } satisfies ComputeResponse);

    const user = userEvent.setup();
    render(<App />);

    // Appears automatically, with no hardcoded ErG name/id/dim anywhere in App.
    expect(await screen.findByText("erg_reduced_graph_315")).toBeInTheDocument();

    await addSmilesViaTextarea(user, "CC(=O)Oc1ccccc1C(=O)O");
    await screen.findByText("Valid");

    // Select ErG, then deselect Morgan-only default to isolate the request.
    const ergCheckbox = screen.getByRole("checkbox", { name: /erg reduced-graph/i });
    await user.click(ergCheckbox);
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    await waitFor(() =>
      expect(computeFeatures).toHaveBeenCalledWith(
        ["CC(=O)Oc1ccccc1C(=O)O"],
        ["erg_reduced_graph_315"],
      ),
    );

    const resultsPanel = await screen.findByTestId("results-panel");
    expect(resultsPanel).toHaveTextContent("erg_reduced_graph_315");
    expect(resultsPanel).toHaveTextContent("315");
  });

  it("discovers a count-valued agent (e.g. fragment descriptors) automatically and preserves integer counts, not booleans", async () => {
    vi.mocked(getAgents).mockResolvedValue([MORGAN_AGENT, FRAGMENT_AGENT]);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "CC(=O)Oc1ccccc1C(=O)O", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockResolvedValue({
      results: [
        {
          smiles: "CC(=O)Oc1ccccc1C(=O)O",
          valid: true,
          error: null,
          features: [
            {
              output_structure: "vector",
              agent_id: "rdkit_fragment_descriptors",
              agent_version: "1.0.0",
              values: [0, 1, 2, 3],
              dim: 4,
            },
          ],
        },
      ],
    } satisfies ComputeResponse);

    const user = userEvent.setup();
    render(<App />);

    // Appears automatically, with no hardcoded fragment name/id/dim anywhere in App.
    expect(await screen.findByText("rdkit_fragment_descriptors")).toBeInTheDocument();

    await addSmilesViaTextarea(user, "CC(=O)Oc1ccccc1C(=O)O");
    await screen.findByText("Valid");

    const fragmentCheckbox = screen.getByRole("checkbox", {
      name: /rdkit fragment descriptors/i,
    });
    await user.click(fragmentCheckbox);
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    await waitFor(() =>
      expect(computeFeatures).toHaveBeenCalledWith(
        ["CC(=O)Oc1ccccc1C(=O)O"],
        ["rdkit_fragment_descriptors"],
      ),
    );

    const resultsPanel = await screen.findByTestId("results-panel");
    expect(resultsPanel).toHaveTextContent("rdkit_fragment_descriptors");
    expect(resultsPanel).toHaveTextContent("4");
    // Count values must render as numbers, never coerced to booleans.
    expect(resultsPanel).not.toHaveTextContent("true");
    expect(resultsPanel).not.toHaveTextContent("false");
  });

  it("discovers a sequence-valued agent (e.g. SELFIES) automatically and renders its token sequence, not as numeric features", async () => {
    vi.mocked(getAgents).mockResolvedValue([MORGAN_AGENT, SELFIES_AGENT]);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "CCO", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockResolvedValue({
      results: [
        {
          smiles: "CCO",
          valid: true,
          error: null,
          features: [
            {
              output_structure: "sequence",
              agent_id: "selfies_sequence",
              agent_version: "1.0.0",
              tokens: ["[C]", "[C]", "[O]"],
              length: 3,
            },
          ],
        },
      ],
    } satisfies ComputeResponse);

    const user = userEvent.setup();
    render(<App />);

    // Appears automatically, with no hardcoded SELFIES name/id/dim anywhere in App.
    expect(await screen.findByText("selfies_sequence")).toBeInTheDocument();

    await addSmilesViaTextarea(user, "CCO");
    await screen.findByText("Valid");

    const selfiesCheckbox = screen.getByRole("checkbox", { name: /selfies sequence/i });
    await user.click(selfiesCheckbox);
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    await waitFor(() =>
      expect(computeFeatures).toHaveBeenCalledWith(["CCO"], ["selfies_sequence"]),
    );

    const resultsPanel = await screen.findByTestId("results-panel");
    expect(resultsPanel).toHaveTextContent("selfies_sequence");
    expect(resultsPanel).toHaveTextContent("3 tokens");
    expect(resultsPanel).toHaveTextContent("[C][C][O]");
  });

  it("shows a valid molecule's badge AND its representation-computation error together (valid=true, error populated, features=[])", async () => {
    // Generic regression, not SELFIES-specific: the backend distinguishes
    // "RDKit rejected this SMILES" (valid=false) from "the molecule is
    // fine but a requested representation could not be computed for it"
    // (valid=true, error populated, features=[]). The UI must not hide
    // that error just because valid=true.
    vi.mocked(getAgents).mockResolvedValue([MORGAN_AGENT, SELFIES_AGENT]);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "Cl[I](Cl)Cl", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockResolvedValue({
      results: [
        {
          smiles: "Cl[I](Cl)Cl",
          valid: true,
          error: "selfies_sequence: failed to encode molecule as SELFIES: 'Cl[I](Cl)Cl'",
          features: [],
        },
      ],
    } satisfies ComputeResponse);

    const user = userEvent.setup();
    render(<App />);
    await addSmilesViaTextarea(user, "Cl[I](Cl)Cl");
    await screen.findByText("Valid");

    const selfiesCheckbox = screen.getByRole("checkbox", { name: /selfies sequence/i });
    await user.click(selfiesCheckbox);
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    await waitFor(() => expect(computeFeatures).toHaveBeenCalled());

    const resultsPanel = await screen.findByTestId("results-panel");
    // The "Valid" badge is still shown for this molecule...
    expect(resultsPanel).toHaveTextContent("Valid");
    // ...but the computation-failure error must still be visible, not
    // silently dropped because valid=true.
    expect(resultsPanel).toHaveTextContent("failed to encode molecule as SELFIES");
    // No fabricated feature content is rendered for the failed agent.
    expect(resultsPanel).not.toHaveTextContent("tokens");
  });

  it("omits a deselected agent from the compute request", async () => {
    vi.mocked(getAgents).mockResolvedValue([MORGAN_AGENT, ERG_AGENT]);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "CCO", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockResolvedValue({
      results: [{ smiles: "CCO", valid: true, error: null, features: [] }],
    } satisfies ComputeResponse);

    const user = userEvent.setup();
    render(<App />);
    await screen.findByText("erg_reduced_graph_315");

    await addSmilesViaTextarea(user, "CCO");
    await screen.findByText("Valid");

    // Select both, then deselect ErG: only Morgan should remain in the request.
    await user.click(screen.getByRole("checkbox", { name: /morgan ecfp/i }));
    await user.click(screen.getByRole("checkbox", { name: /erg reduced-graph/i }));
    await user.click(screen.getByRole("checkbox", { name: /erg reduced-graph/i }));
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    await waitFor(() =>
      expect(computeFeatures).toHaveBeenCalledWith(["CCO"], ["morgan_ecfp4_1024"]),
    );
  });

  it("shows a clear error when the compute request fails", async () => {
    vi.mocked(getAgents).mockResolvedValue(AGENTS);
    vi.mocked(validateMolecules).mockResolvedValue({
      results: [{ smiles: "CCO", valid: true, error: null }],
    } satisfies ValidateResponse);
    vi.mocked(computeFeatures).mockRejectedValue(new ApiError("Unknown agent id", 400));

    const user = userEvent.setup();
    render(<App />);
    await addSmilesViaTextarea(user, "CCO");
    await screen.findByText("Valid");

    await user.click(screen.getByRole("checkbox", { name: /morgan ecfp/i }));
    await user.click(screen.getByRole("button", { name: /^compute$/i }));

    const resultsPanel = await screen.findByTestId("results-panel");
    expect(resultsPanel).toHaveTextContent("Unknown agent id");
  });
});
