import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AgentSelector } from "../AgentSelector";
import type { AgentMetadata } from "../../types/api";

const AGENTS: AgentMetadata[] = [
  {
    id: "morgan_ecfp4_1024",
    name: "Morgan ECFP (radius=2, 1024 bits)",
    version: "1.0.0",
    output_dim: 1024,
    requires_3d: false,
    value_type: "binary",
    output_structure: "vector",
    feature_names: null,
  },
  {
    id: "rdkit_physchem_descriptors",
    name: "RDKit Physicochemical Descriptors",
    version: "1.0.0",
    output_dim: 217,
    requires_3d: false,
    value_type: "continuous",
    output_structure: "vector",
    feature_names: ["MolWt", "MolLogP", "TPSA"],
  },
  {
    id: "erg_reduced_graph_315",
    name: "ErG Reduced-Graph Fingerprint",
    version: "1.0.0",
    output_dim: 315,
    requires_3d: false,
    value_type: "continuous",
    output_structure: "vector",
    feature_names: null,
  },
  {
    id: "rdkit_fragment_descriptors",
    name: "RDKit Fragment Descriptors",
    version: "1.0.0",
    output_dim: 85,
    requires_3d: false,
    value_type: "count",
    output_structure: "vector",
    feature_names: ["fr_Al_COO", "fr_benzene"],
  },
  {
    id: "selfies_sequence",
    name: "SELFIES Sequence",
    version: "1.0.0",
    output_dim: null,
    requires_3d: false,
    value_type: "categorical",
    output_structure: "sequence",
    feature_names: null,
  },
];

describe("AgentSelector", () => {
  it("renders agent metadata dynamically from props, not hardcoded", () => {
    render(
      <AgentSelector
        agents={AGENTS}
        selectedIds={new Set()}
        onToggle={vi.fn()}
        loading={false}
        error={null}
      />,
    );

    expect(screen.getByText("Morgan ECFP (radius=2, 1024 bits)")).toBeInTheDocument();
    expect(screen.getByText("morgan_ecfp4_1024")).toBeInTheDocument();
    expect(screen.getByText("rdkit_physchem_descriptors")).toBeInTheDocument();
    expect(screen.getByText("ErG Reduced-Graph Fingerprint")).toBeInTheDocument();
    expect(screen.getByText("erg_reduced_graph_315")).toBeInTheDocument();
    expect(screen.getByText("RDKit Fragment Descriptors")).toBeInTheDocument();
    expect(screen.getByText("rdkit_fragment_descriptors")).toBeInTheDocument();
    expect(screen.getByText("SELFIES Sequence")).toBeInTheDocument();
    expect(screen.getByText("selfies_sequence")).toBeInTheDocument();
    expect(screen.getAllByText("1.0.0")).toHaveLength(5);
    expect(screen.getByText("1024")).toBeInTheDocument();
    expect(screen.getByText("217")).toBeInTheDocument();
    expect(screen.getByText("315")).toBeInTheDocument();
    expect(screen.getByText("85")).toBeInTheDocument();
    expect(screen.getAllByText("No")).toHaveLength(5);
  });

  it("surfaces value_type generically for binary, count, continuous, and categorical agents", () => {
    render(
      <AgentSelector
        agents={AGENTS}
        selectedIds={new Set()}
        onToggle={vi.fn()}
        loading={false}
        error={null}
      />,
    );

    expect(screen.getByText("binary")).toBeInTheDocument();
    expect(screen.getAllByText("continuous")).toHaveLength(2);
    expect(screen.getByText("count")).toBeInTheDocument();
    expect(screen.getByText("categorical")).toBeInTheDocument();
  });

  it("surfaces output_structure generically, distinguishing vector from sequence agents", () => {
    render(
      <AgentSelector
        agents={AGENTS}
        selectedIds={new Set()}
        onToggle={vi.fn()}
        loading={false}
        error={null}
      />,
    );

    expect(screen.getAllByText("vector")).toHaveLength(4);
    expect(screen.getByText("sequence")).toBeInTheDocument();
  });

  it('shows "Variable" (never "0") for a sequence agent\'s dimension', () => {
    render(
      <AgentSelector
        agents={AGENTS}
        selectedIds={new Set()}
        onToggle={vi.fn()}
        loading={false}
        error={null}
      />,
    );

    expect(screen.getByText("Variable")).toBeInTheDocument();
    expect(screen.queryByText("0")).not.toBeInTheDocument();
  });

  it("lets an agent (e.g. ErG) be selected and deselected via its checkbox", () => {
    const onToggle = vi.fn();
    render(
      <AgentSelector
        agents={AGENTS}
        selectedIds={new Set()}
        onToggle={onToggle}
        loading={false}
        error={null}
      />,
    );

    const ergCheckbox = screen.getByRole("checkbox", { name: /erg reduced-graph/i });
    ergCheckbox.click();
    expect(onToggle).toHaveBeenCalledWith("erg_reduced_graph_315");
  });

  it("lets the SELFIES sequence agent be selected and deselected via its checkbox", () => {
    const onToggle = vi.fn();
    render(
      <AgentSelector
        agents={AGENTS}
        selectedIds={new Set()}
        onToggle={onToggle}
        loading={false}
        error={null}
      />,
    );

    const selfiesCheckbox = screen.getByRole("checkbox", { name: /selfies sequence/i });
    selfiesCheckbox.click();
    expect(onToggle).toHaveBeenCalledWith("selfies_sequence");
  });

  it("shows a friendly message when no agents are available", () => {
    render(
      <AgentSelector
        agents={[]}
        selectedIds={new Set()}
        onToggle={vi.fn()}
        loading={false}
        error={null}
      />,
    );
    expect(screen.getByText(/no feature agents are currently available/i)).toBeInTheDocument();
  });

  it("shows the backend error message when loading fails", () => {
    render(
      <AgentSelector
        agents={[]}
        selectedIds={new Set()}
        onToggle={vi.fn()}
        loading={false}
        error="Could not reach the MolFusion backend at http://127.0.0.1:8000. Is it running?"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/could not reach the molfusion backend/i);
  });
});
