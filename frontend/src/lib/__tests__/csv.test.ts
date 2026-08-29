import { describe, expect, it } from "vitest";
import { buildResultsCsv, parseSmilesCsv } from "../csv";
import type { AgentMetadata, MoleculeResult } from "../../types/api";

const MORGAN_AGENT: AgentMetadata = {
  id: "morgan_ecfp4_1024",
  name: "Morgan ECFP (radius=2, 1024 bits)",
  version: "1.0.0",
  output_dim: 1024,
  requires_3d: false,
  value_type: "binary",
  feature_names: null,
};

const MACCS_AGENT: AgentMetadata = {
  id: "maccs_keys_167",
  name: "MACCS Keys (167-bit)",
  version: "1.0.0",
  output_dim: 167,
  requires_3d: false,
  value_type: "binary",
  feature_names: null,
};

const DESCRIPTOR_AGENT: AgentMetadata = {
  id: "rdkit_physchem_descriptors",
  name: "RDKit Physicochemical Descriptors",
  version: "1.0.0",
  output_dim: 3,
  requires_3d: false,
  value_type: "continuous",
  feature_names: ["MolWt", "MolLogP", "TPSA"],
};

const ERG_AGENT: AgentMetadata = {
  id: "erg_reduced_graph_315",
  name: "ErG Reduced-Graph Fingerprint",
  version: "1.0.0",
  output_dim: 4,
  requires_3d: false,
  value_type: "continuous",
  feature_names: null,
};

describe("parseSmilesCsv", () => {
  it("extracts the smiles column case-insensitively and preserves row order", () => {
    const csv = "name,SMILES\nethanol,CCO\nbenzene,c1ccccc1\n";
    const result = parseSmilesCsv(csv);
    expect(result.error).toBeNull();
    expect(result.smiles).toEqual(["CCO", "c1ccccc1"]);
  });

  it("rejects a CSV with no smiles column", () => {
    const csv = "name,formula\nethanol,C2H6O\n";
    const result = parseSmilesCsv(csv);
    expect(result.smiles).toEqual([]);
    expect(result.error).toMatch(/smiles/i);
  });

  it("handles quoted fields containing embedded commas", () => {
    const csv = 'smiles,note\n"C1=CC=CC=C1","aromatic, six-membered"\n';
    const result = parseSmilesCsv(csv);
    expect(result.smiles).toEqual(["C1=CC=CC=C1"]);
  });

  it("rejects an empty CSV", () => {
    const result = parseSmilesCsv("");
    expect(result.smiles).toEqual([]);
    expect(result.error).toMatch(/empty/i);
  });
});

describe("buildResultsCsv", () => {
  it("quotes fields containing commas", () => {
    const results: MoleculeResult[] = [
      { smiles: "CCO, ethanol", valid: true, error: null, features: [] },
    ];
    const csv = buildResultsCsv(results, []);
    const [, row] = csv.split("\r\n");
    expect(row).toBe('"CCO, ethanol",true,,,,,,');
  });

  it("quotes fields containing double quotes and doubles them", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: false,
        error: 'parse error: unexpected "X", stop',
        features: [],
      },
    ];
    const csv = buildResultsCsv(results, []);
    const lines = csv.split("\r\n");
    expect(lines[0]).toBe(
      "smiles,valid,error,agent_id,agent_version,dim,feature_names,values",
    );
    expect(lines[1]).toContain('"parse error: unexpected ""X"", stop"');
  });

  it("quotes fields containing embedded newlines", () => {
    const results: MoleculeResult[] = [
      { smiles: "CCO", valid: false, error: "line one\nline two", features: [] },
    ];
    const csv = buildResultsCsv(results, []);
    // The embedded newline must stay inside a quoted field, not create a spurious row.
    expect(csv.split("\r\n")).toHaveLength(2);
    expect(csv).toContain('"line one\nline two"');
  });

  it("leaves empty strings (no error, no agent) unquoted and empty", () => {
    const results: MoleculeResult[] = [{ smiles: "CCO", valid: true, error: null, features: [] }];
    const csv = buildResultsCsv(results, []);
    const [, row] = csv.split("\r\n");
    expect(row).toBe("CCO,true,,,,,,");
  });

  it("preserves smiles, agent_id, agent_version, and dim for a valid molecule", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          { agent_id: "maccs_keys_167", agent_version: "1.0.0", values: [0, 1, 0], dim: 3 },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [MACCS_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe("CCO,true,,maccs_keys_167,1.0.0,3,,0;1;0");
  });

  it("produces one row per feature vector across multiple agents", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          { agent_id: "morgan_ecfp4_1024", agent_version: "1.0.0", values: [0, 0, 1], dim: 3 },
          { agent_id: "maccs_keys_167", agent_version: "1.0.0", values: [1, 0], dim: 2 },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [MORGAN_AGENT, MACCS_AGENT]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(3); // header + 2 feature rows
    expect(lines[1]).toBe("CCO,true,,morgan_ecfp4_1024,1.0.0,3,,0;0;1");
    expect(lines[2]).toBe("CCO,true,,maccs_keys_167,1.0.0,2,,1;0");
  });

  it("produces rows for multiple molecules, invalid ones included, in input order", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          { agent_id: "maccs_keys_167", agent_version: "1.0.0", values: [1, 0], dim: 2 },
        ],
      },
      { smiles: "INVALID", valid: false, error: "could not be parsed", features: [] },
    ];
    const csv = buildResultsCsv(results, [MACCS_AGENT]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain("CCO,true");
    expect(lines[2]).toBe("INVALID,false,could not be parsed,,,,,");
  });

  it("preserves descriptor feature names, aligned with values, when the agent exposes them", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            agent_id: "rdkit_physchem_descriptors",
            agent_version: "1.0.0",
            values: [46.069, -0.0014, 20.23],
            dim: 3,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [DESCRIPTOR_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      "CCO,true,,rdkit_physchem_descriptors,1.0.0,3,MolWt;MolLogP;TPSA,46.069;-0.0014;20.23",
    );
  });

  it("leaves feature_names empty for agents that expose none (e.g. fingerprints)", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          { agent_id: "morgan_ecfp4_1024", agent_version: "1.0.0", values: [0, 1], dim: 2 },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [MORGAN_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe("CCO,true,,morgan_ecfp4_1024,1.0.0,2,,0;1");
  });

  it("preserves continuous fractional values (e.g. ErG) exactly, without rounding or binarizing", () => {
    const values = [0, 0.3, 1.6, 0.9];
    const results: MoleculeResult[] = [
      {
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        error: null,
        features: [
          { agent_id: "erg_reduced_graph_315", agent_version: "1.0.0", values, dim: 4 },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [ERG_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      "CC(=O)Oc1ccccc1C(=O)O,true,,erg_reduced_graph_315,1.0.0,4,,0;0.3;1.6;0.9",
    );

    // Round-trip: splitting the values field back apart must reproduce the
    // original numbers exactly (no lossy formatting rule was applied).
    const valuesField = row.split(",").pop() ?? "";
    const parsedBack = valuesField.split(";").map(Number);
    expect(parsedBack).toEqual(values);
  });

  it("never recomputes or reorders values — echoes them verbatim from the API response", () => {
    const values = [3.14159, -1, 0, 999.999];
    const results: MoleculeResult[] = [
      {
        smiles: "c1ccccc1",
        valid: true,
        error: null,
        features: [{ agent_id: "x", agent_version: "9.9.9", values, dim: values.length }],
      },
    ];
    const csv = buildResultsCsv(results, []);
    const [, row] = csv.split("\r\n");
    expect(row.endsWith(values.join(";"))).toBe(true);
  });
});
