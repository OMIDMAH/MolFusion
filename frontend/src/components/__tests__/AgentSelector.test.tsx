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
    feature_names: null,
  },
  {
    id: "rdkit_physchem_descriptors",
    name: "RDKit Physicochemical Descriptors",
    version: "1.0.0",
    output_dim: 217,
    requires_3d: false,
    value_type: "continuous",
    feature_names: ["MolWt", "MolLogP", "TPSA"],
  },
  {
    id: "erg_reduced_graph_315",
    name: "ErG Reduced-Graph Fingerprint",
    version: "1.0.0",
    output_dim: 315,
    requires_3d: false,
    value_type: "continuous",
    feature_names: null,
  },
  {
    id: "rdkit_fragment_descriptors",
    name: "RDKit Fragment Descriptors",
    version: "1.0.0",
    output_dim: 85,
    requires_3d: false,
    value_type: "count",
    feature_names: ["fr_Al_COO", "fr_benzene"],
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
    expect(screen.getAllByText("1.0.0")).toHaveLength(4);
    expect(screen.getByText("1024")).toBeInTheDocument();
    expect(screen.getByText("217")).toBeInTheDocument();
    expect(screen.getByText("315")).toBeInTheDocument();
    expect(screen.getByText("85")).toBeInTheDocument();
    expect(screen.getAllByText("No")).toHaveLength(4);
  });

  it("surfaces value_type generically for binary, count, and continuous agents", () => {
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
