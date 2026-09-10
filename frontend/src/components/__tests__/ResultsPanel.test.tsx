import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ResultsPanel } from "../ResultsPanel";
import type { AgentMetadata, MoleculeResult } from "../../types/api";

const AGENTS: AgentMetadata[] = [
  {
    id: "morgan_ecfp4_1024",
    name: "Morgan ECFP",
    version: "1.0.0",
    output_dim: 1024,
    requires_3d: false,
    value_type: "binary",
    output_structure: "vector",
    feature_names: null,
    availability: { available: true, code: null, message: null },
  },
  {
    id: "selfies_sequence",
    name: "SELFIES",
    version: "1.0.0",
    output_dim: null,
    requires_3d: false,
    value_type: "categorical",
    output_structure: "sequence",
    feature_names: null,
    availability: { available: true, code: null, message: null },
  },
];

function partialSuccess(): MoleculeResult {
  return {
    smiles: "Cl[I](Cl)Cl",
    valid: true,
    error: null,
    features: [
      {
        output_structure: "vector",
        agent_id: "morgan_ecfp4_1024",
        agent_version: "1.0.0",
        values: [1, 0, 1],
        dim: 3,
      },
    ],
    feature_errors: [
      {
        agent_id: "selfies_sequence",
        agent_version: "1.0.0",
        error: "selfies_sequence: failed to encode molecule as SELFIES",
      },
    ],
  };
}

function renderPanel(results: MoleculeResult[]) {
  render(
    <ResultsPanel results={results} agents={AGENTS} computing={false} computeError={null} />,
  );
}

describe("ResultsPanel — per-agent failures", () => {
  it("still renders the successful representation when another agent failed", () => {
    renderPanel([partialSuccess()]);
    expect(screen.getByText("morgan_ecfp4_1024")).toBeInTheDocument();
  });

  it("renders the failed agent's error against that agent", () => {
    renderPanel([partialSuccess()]);
    const errors = screen.getByLabelText("Representation errors");
    expect(errors).toHaveTextContent("selfies_sequence");
    expect(errors).toHaveTextContent("failed to encode molecule as SELFIES");
  });

  it("keeps the molecule marked valid despite the failure", () => {
    renderPanel([partialSuccess()]);
    expect(screen.getByText("Valid")).toBeInTheDocument();
    expect(screen.queryByText("Invalid")).not.toBeInTheDocument();
  });

  it("shows no representation-error list when every agent succeeded", () => {
    const clean = partialSuccess();
    clean.feature_errors = [];
    renderPanel([clean]);
    expect(screen.queryByLabelText("Representation errors")).not.toBeInTheDocument();
  });

  it("renders every failure when all agents failed", () => {
    const allFailed: MoleculeResult = {
      smiles: "Cl[I](Cl)Cl",
      valid: true,
      error: null,
      features: [],
      feature_errors: [
        { agent_id: "morgan_ecfp4_1024", agent_version: "1.0.0", error: "boom one" },
        { agent_id: "selfies_sequence", agent_version: "1.0.0", error: "boom two" },
      ],
    };
    renderPanel([allFailed]);
    const errors = screen.getByLabelText("Representation errors");
    expect(errors).toHaveTextContent("boom one");
    expect(errors).toHaveTextContent("boom two");
    expect(screen.getByText("Valid")).toBeInTheDocument();
  });

  it("still shows a molecule-level error for an unparseable SMILES", () => {
    const invalid: MoleculeResult = {
      smiles: "INVALID",
      valid: false,
      error: "SMILES could not be parsed",
      features: [],
      feature_errors: [],
    };
    renderPanel([invalid]);
    expect(screen.getByText("SMILES could not be parsed")).toBeInTheDocument();
    expect(screen.getByText("Invalid")).toBeInTheDocument();
    expect(screen.queryByLabelText("Representation errors")).not.toBeInTheDocument();
  });
});
