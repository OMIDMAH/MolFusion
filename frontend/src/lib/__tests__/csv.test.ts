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
  output_structure: "vector",
  feature_names: null,
};

const MACCS_AGENT: AgentMetadata = {
  id: "maccs_keys_167",
  name: "MACCS Keys (167-bit)",
  version: "1.0.0",
  output_dim: 167,
  requires_3d: false,
  value_type: "binary",
  output_structure: "vector",
  feature_names: null,
};

const DESCRIPTOR_AGENT: AgentMetadata = {
  id: "rdkit_physchem_descriptors",
  name: "RDKit Physicochemical Descriptors",
  version: "1.0.0",
  output_dim: 3,
  requires_3d: false,
  value_type: "continuous",
  output_structure: "vector",
  feature_names: ["MolWt", "MolLogP", "TPSA"],
};

const ERG_AGENT: AgentMetadata = {
  id: "erg_reduced_graph_315",
  name: "ErG Reduced-Graph Fingerprint",
  version: "1.0.0",
  output_dim: 4,
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

const HEADER_ROW =
  "smiles,valid,error,agent_id,agent_version,output_structure,value_type,dim,feature_names,values,sequence_length,sequence_tokens,sequence_string";

/** Mirrors csv.ts's own escaping rule, so expected rows built here go
 * through the same quoting logic as the code under test. */
function escapeForTest(value: string): string {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

function vectorRow(fields: {
  smiles: string;
  valid: boolean;
  error?: string;
  agentId: string;
  agentVersion: string;
  valueType: string;
  dim: number;
  featureNames?: string;
  values: string;
}): string {
  return [
    fields.smiles,
    String(fields.valid),
    fields.error ?? "",
    fields.agentId,
    fields.agentVersion,
    "vector",
    fields.valueType,
    String(fields.dim),
    fields.featureNames ?? "",
    fields.values,
    "",
    "",
    "",
  ]
    .map(escapeForTest)
    .join(",");
}

function sequenceRow(fields: {
  smiles: string;
  valid: boolean;
  error?: string;
  agentId: string;
  agentVersion: string;
  valueType: string;
  length: number;
  tokensJson: string;
  sequenceString: string;
}): string {
  return [
    fields.smiles,
    String(fields.valid),
    fields.error ?? "",
    fields.agentId,
    fields.agentVersion,
    "sequence",
    fields.valueType,
    "",
    "",
    "",
    String(fields.length),
    fields.tokensJson,
    fields.sequenceString,
  ]
    .map(escapeForTest)
    .join(",");
}

function noFeatureRow(smiles: string, valid: boolean, error = ""): string {
  return [smiles, String(valid), error, "", "", "", "", "", "", "", "", "", ""]
    .map(escapeForTest)
    .join(",");
}

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

describe("buildResultsCsv header", () => {
  it("has the generic header covering both vector and sequence columns", () => {
    const csv = buildResultsCsv([], []);
    expect(csv.split("\r\n")[0]).toBe(HEADER_ROW);
  });
});

describe("buildResultsCsv — vector outputs", () => {
  it("quotes fields containing commas", () => {
    const results: MoleculeResult[] = [
      { smiles: "CCO, ethanol", valid: true, error: null, features: [] },
    ];
    const csv = buildResultsCsv(results, []);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(noFeatureRow("CCO, ethanol", true));
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
    expect(lines[0]).toBe(HEADER_ROW);
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
    expect(row).toBe(noFeatureRow("CCO", true));
  });

  it("preserves smiles, agent_id, agent_version, and dim for a valid molecule", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "maccs_keys_167",
            agent_version: "1.0.0",
            values: [0, 1, 0],
            dim: 3,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [MACCS_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      vectorRow({
        smiles: "CCO",
        valid: true,
        agentId: "maccs_keys_167",
        agentVersion: "1.0.0",
        valueType: "binary",
        dim: 3,
        values: "0;1;0",
      }),
    );
  });

  it("produces one row per feature vector across multiple agents", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "morgan_ecfp4_1024",
            agent_version: "1.0.0",
            values: [0, 0, 1],
            dim: 3,
          },
          {
            output_structure: "vector",
            agent_id: "maccs_keys_167",
            agent_version: "1.0.0",
            values: [1, 0],
            dim: 2,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [MORGAN_AGENT, MACCS_AGENT]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(3); // header + 2 feature rows
    expect(lines[1]).toBe(
      vectorRow({
        smiles: "CCO",
        valid: true,
        agentId: "morgan_ecfp4_1024",
        agentVersion: "1.0.0",
        valueType: "binary",
        dim: 3,
        values: "0;0;1",
      }),
    );
    expect(lines[2]).toBe(
      vectorRow({
        smiles: "CCO",
        valid: true,
        agentId: "maccs_keys_167",
        agentVersion: "1.0.0",
        valueType: "binary",
        dim: 2,
        values: "1;0",
      }),
    );
  });

  it("produces rows for multiple molecules, invalid ones included, in input order", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "maccs_keys_167",
            agent_version: "1.0.0",
            values: [1, 0],
            dim: 2,
          },
        ],
      },
      { smiles: "INVALID", valid: false, error: "could not be parsed", features: [] },
    ];
    const csv = buildResultsCsv(results, [MACCS_AGENT]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain("CCO,true");
    expect(lines[2]).toBe(noFeatureRow("INVALID", false, "could not be parsed"));
  });

  it("preserves descriptor feature names and value_type, aligned with values", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
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
      vectorRow({
        smiles: "CCO",
        valid: true,
        agentId: "rdkit_physchem_descriptors",
        agentVersion: "1.0.0",
        valueType: "continuous",
        dim: 3,
        featureNames: "MolWt;MolLogP;TPSA",
        values: "46.069;-0.0014;20.23",
      }),
    );
  });

  it("leaves feature_names empty for agents that expose none (e.g. fingerprints)", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "morgan_ecfp4_1024",
            agent_version: "1.0.0",
            values: [0, 1],
            dim: 2,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [MORGAN_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      vectorRow({
        smiles: "CCO",
        valid: true,
        agentId: "morgan_ecfp4_1024",
        agentVersion: "1.0.0",
        valueType: "binary",
        dim: 2,
        values: "0;1",
      }),
    );
  });

  it("preserves continuous fractional values (e.g. ErG) exactly, without rounding or binarizing", () => {
    const values = [0, 0.3, 1.6, 0.9];
    const results: MoleculeResult[] = [
      {
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "erg_reduced_graph_315",
            agent_version: "1.0.0",
            values,
            dim: 4,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [ERG_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      vectorRow({
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        agentId: "erg_reduced_graph_315",
        agentVersion: "1.0.0",
        valueType: "continuous",
        dim: 4,
        values: "0;0.3;1.6;0.9",
      }),
    );

    // Round-trip: the values field must reproduce the original numbers exactly.
    const valuesField = row.split(",")[9];
    const parsedBack = valuesField.split(";").map(Number);
    expect(parsedBack).toEqual(values);
  });

  it("preserves count values (e.g. fragment descriptors) exactly, not as booleans", () => {
    const values = [0, 1, 2, 3];
    const results: MoleculeResult[] = [
      {
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "rdkit_fragment_descriptors",
            agent_version: "1.0.0",
            values,
            dim: 4,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [FRAGMENT_AGENT]);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      vectorRow({
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        agentId: "rdkit_fragment_descriptors",
        agentVersion: "1.0.0",
        valueType: "count",
        dim: 4,
        featureNames: "fr_Al_COO;fr_Ar_COO;fr_benzene;fr_ester",
        values: "0;1;2;3",
      }),
    );
    expect(csv).not.toContain("true;false");
  });

  it("never recomputes or reorders values — echoes them verbatim from the API response", () => {
    const values = [3.14159, -1, 0, 999.999];
    const results: MoleculeResult[] = [
      {
        smiles: "c1ccccc1",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "x",
            agent_version: "9.9.9",
            values,
            dim: values.length,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, []);
    const [, row] = csv.split("\r\n");
    expect(row).toBe(
      vectorRow({
        smiles: "c1ccccc1",
        valid: true,
        agentId: "x",
        agentVersion: "9.9.9",
        valueType: "",
        dim: values.length,
        values: values.join(";"),
      }),
    );
  });
});

describe("buildResultsCsv — sequence outputs (SELFIES)", () => {
  it("preserves a token sequence as a lossless JSON array, plus a convenience concatenated string", () => {
    const tokens = ["[C]", "[C]", "[O]"];
    const results: MoleculeResult[] = [
      {
        smiles: "CCO",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "sequence",
            agent_id: "selfies_sequence",
            agent_version: "1.0.0",
            tokens,
            length: 3,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [SELFIES_AGENT]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(2);
    expect(lines[1]).toBe(
      sequenceRow({
        smiles: "CCO",
        valid: true,
        agentId: "selfies_sequence",
        agentVersion: "1.0.0",
        valueType: "categorical",
        length: 3,
        tokensJson: JSON.stringify(tokens),
        sequenceString: "[C][C][O]",
      }),
    );
  });

  it("does not create per-token feature_0..feature_N columns", () => {
    const results: MoleculeResult[] = [
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
    ];
    const csv = buildResultsCsv(results, [SELFIES_AGENT]);
    expect(csv).not.toMatch(/feature_\d/);
  });

  it("correctly escapes tokens containing brackets and the JSON array's own quotes/commas", () => {
    const tokens = ["[C@H1]", "[Branch1]", "[=Branch2]"];
    const results: MoleculeResult[] = [
      {
        smiles: "C[C@H](O)Cl",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "sequence",
            agent_id: "selfies_sequence",
            agent_version: "1.0.0",
            tokens,
            length: tokens.length,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [SELFIES_AGENT]);
    const [, row] = csv.split("\r\n");

    // JSON.stringify(tokens) contains commas and double quotes, so the CSV
    // field must be wrapped in quotes with the inner quotes doubled.
    const expectedJson = JSON.stringify(tokens);
    const expectedEscaped = `"${expectedJson.replace(/"/g, '""')}"`;
    expect(row).toContain(expectedEscaped);
  });

  it("round-trips the token list from the exported sequence_tokens field via JSON.parse", () => {
    const tokens = ["[C]", "[=C]", "[Ring1]", "[Branch1]", "[C@H1]"];
    const results: MoleculeResult[] = [
      {
        smiles: "c1ccccc1",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "sequence",
            agent_id: "selfies_sequence",
            agent_version: "1.0.0",
            tokens,
            length: tokens.length,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [SELFIES_AGENT]);

    // The quoted sequence_tokens field is the only quoted field on this
    // line; extract it and reverse the CSV quoting to recover raw JSON.
    const [, row] = csv.split("\r\n");
    const match = row.match(/"((?:[^"]|"")*)"/);
    expect(match).not.toBeNull();
    const unescaped = (match as RegExpMatchArray)[1].replace(/""/g, '"');
    const recovered = JSON.parse(unescaped);

    expect(recovered).toEqual(tokens);
  });

  it("produces rows for multiple molecules' sequence outputs, in input order", () => {
    const results: MoleculeResult[] = [
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
      {
        smiles: "c1ccccc1",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "sequence",
            agent_id: "selfies_sequence",
            agent_version: "1.0.0",
            tokens: ["[C]", "[=C]", "[C]", "[=C]", "[C]", "[=C]", "[Ring1]", "[=Branch1]"],
            length: 8,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [SELFIES_AGENT]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(3);
    expect(lines[1]).toContain("CCO,true");
    expect(lines[1]).toContain(",3,");
    expect(lines[2]).toContain("c1ccccc1,true");
    expect(lines[2]).toContain(",8,");
  });
});

describe("buildResultsCsv — mixed vector and sequence outputs", () => {
  it("lets binary, count, continuous, and categorical/sequence agents coexist in the same export", () => {
    const results: MoleculeResult[] = [
      {
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        error: null,
        features: [
          {
            output_structure: "vector",
            agent_id: "morgan_ecfp4_1024",
            agent_version: "1.0.0",
            values: [0, 1],
            dim: 2,
          },
          {
            output_structure: "vector",
            agent_id: "rdkit_fragment_descriptors",
            agent_version: "1.0.0",
            values: [0, 1, 2, 3],
            dim: 4,
          },
          {
            output_structure: "vector",
            agent_id: "rdkit_physchem_descriptors",
            agent_version: "1.0.0",
            values: [180.159, 1.31, 63.6],
            dim: 3,
          },
          {
            output_structure: "sequence",
            agent_id: "selfies_sequence",
            agent_version: "1.0.0",
            tokens: ["[C]", "[C]"],
            length: 2,
          },
        ],
      },
    ];
    const csv = buildResultsCsv(results, [
      MORGAN_AGENT,
      FRAGMENT_AGENT,
      DESCRIPTOR_AGENT,
      SELFIES_AGENT,
    ]);
    const lines = csv.split("\r\n");
    expect(lines).toHaveLength(5); // header + 4 feature rows, no agent-specific special-casing

    expect(lines[1]).toBe(
      vectorRow({
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        agentId: "morgan_ecfp4_1024",
        agentVersion: "1.0.0",
        valueType: "binary",
        dim: 2,
        values: "0;1",
      }),
    );
    expect(lines[2]).toBe(
      vectorRow({
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        agentId: "rdkit_fragment_descriptors",
        agentVersion: "1.0.0",
        valueType: "count",
        dim: 4,
        featureNames: "fr_Al_COO;fr_Ar_COO;fr_benzene;fr_ester",
        values: "0;1;2;3",
      }),
    );
    expect(lines[3]).toBe(
      vectorRow({
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        agentId: "rdkit_physchem_descriptors",
        agentVersion: "1.0.0",
        valueType: "continuous",
        dim: 3,
        featureNames: "MolWt;MolLogP;TPSA",
        values: "180.159;1.31;63.6",
      }),
    );
    expect(lines[4]).toBe(
      sequenceRow({
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
        valid: true,
        agentId: "selfies_sequence",
        agentVersion: "1.0.0",
        valueType: "categorical",
        length: 2,
        tokensJson: JSON.stringify(["[C]", "[C]"]),
        sequenceString: "[C][C]",
      }),
    );
  });
});
