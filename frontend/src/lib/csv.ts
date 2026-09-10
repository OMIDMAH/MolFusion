import type { AgentMetadata, MoleculeResult } from "../types/api";

/** Minimal RFC-4180-ish CSV row parser: handles quoted fields, embedded
 * commas/newlines, and doubled-quote escaping. */
function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let i = 0;
  const len = text.length;

  while (i < len) {
    const char = text[i];

    if (inQuotes) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      field += char;
      i += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      i += 1;
      continue;
    }
    if (char === ",") {
      row.push(field);
      field = "";
      i += 1;
      continue;
    }
    if (char === "\r") {
      i += 1;
      continue;
    }
    if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      i += 1;
      continue;
    }
    field += char;
    i += 1;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

export interface CsvSmilesParseResult {
  smiles: string[];
  error: string | null;
}

/** Extract SMILES strings from an uploaded CSV, using a case-insensitive
 * "smiles" column header. Input order is preserved. No chemistry
 * validation happens here — that is the backend's job. */
export function parseSmilesCsv(text: string): CsvSmilesParseResult {
  const rows = parseCsvRows(text).filter(
    (row) => !(row.length === 1 && row[0].trim() === ""),
  );

  if (rows.length === 0) {
    return { smiles: [], error: "CSV file is empty." };
  }

  const [header, ...dataRows] = rows;
  const columnIndex = header.findIndex((cell) => cell.trim().toLowerCase() === "smiles");
  if (columnIndex === -1) {
    return {
      smiles: [],
      error: 'CSV must contain a column named "smiles" (case-insensitive).',
    };
  }

  const smiles = dataRows
    .map((row) => (row[columnIndex] ?? "").trim())
    .filter((value) => value.length > 0);

  if (smiles.length === 0) {
    return { smiles: [], error: 'The "smiles" column contains no values.' };
  }

  return { smiles, error: null };
}

function escapeCsvField(value: string): string {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

// A single generic row shape covers both "vector" and "sequence" outputs:
// vector-only columns (dim/feature_names/values) are left empty on a
// sequence row, and sequence-only columns (sequence_length/sequence_tokens/
// sequence_string) are left empty on a vector row -- the same pattern
// already used for feature_names being empty on agents that don't expose
// it. This avoids ever creating per-token feature_0..feature_N columns.
const EXPORT_HEADER = [
  "smiles",
  "valid",
  "error",
  "agent_id",
  "agent_version",
  "output_structure",
  "value_type",
  "dim",
  "feature_names",
  "values",
  "sequence_length",
  "sequence_tokens",
  "sequence_string",
  // Phase 5H. Populated only on a row describing an agent that failed for
  // this molecule; kept separate from `error` so a molecule-level failure
  // (unparseable SMILES) stays distinguishable from a representation-level
  // one. Appended rather than inserted so existing positional readers of
  // the earlier columns keep working.
  "feature_error",
];

const EMPTY_FEATURE_COLUMNS = ["", "", "", "", "", "", "", "", "", "", ""];

/** Build a CSV export directly from a /features/compute API response.
 * One row per (molecule, feature output); invalid molecules get a single
 * row with empty agent columns. `agents` supplies each agent's
 * `feature_names`/`value_type` (from GET /agents) so the export can label
 * values by name and type where the backend exposes them — nothing here is
 * recomputed, reordered, or fabricated.
 *
 * Sequence outputs (e.g. SELFIES) are preserved losslessly as a JSON array
 * in `sequence_tokens` (round-trippable via JSON.parse), never split into
 * per-token columns, never padded/truncated, and never converted to
 * numbers. `sequence_string` is a convenience concatenation
 * (tokens.join("")) included alongside the tokens, not instead of them. */
export function buildResultsCsv(results: MoleculeResult[], agents: AgentMetadata[]): string {
  const agentsById = new Map(agents.map((agent) => [agent.id, agent]));
  const lines = [EXPORT_HEADER.map(escapeCsvField).join(",")];

  for (const result of results) {
    if (result.features.length === 0 && result.feature_errors.length === 0) {
      lines.push(
        [result.smiles, String(result.valid), result.error ?? "", ...EMPTY_FEATURE_COLUMNS]
          .map(escapeCsvField)
          .join(","),
      );
      continue;
    }

    for (const feature of result.features) {
      const valueType = agentsById.get(feature.agent_id)?.value_type ?? "";

      if (feature.output_structure === "vector") {
        const featureNames = agentsById.get(feature.agent_id)?.feature_names;
        lines.push(
          [
            result.smiles,
            String(result.valid),
            result.error ?? "",
            feature.agent_id,
            feature.agent_version,
            feature.output_structure,
            valueType,
            String(feature.dim),
            featureNames ? featureNames.join(";") : "",
            feature.values.join(";"),
            "",
            "",
            "",
            "",
          ]
            .map(escapeCsvField)
            .join(","),
        );
        continue;
      }

      lines.push(
        [
          result.smiles,
          String(result.valid),
          result.error ?? "",
          feature.agent_id,
          feature.agent_version,
          feature.output_structure,
          valueType,
          "",
          "",
          "",
          String(feature.length),
          JSON.stringify(feature.tokens),
          feature.tokens.join(""),
          "",
        ]
          .map(escapeCsvField)
          .join(","),
      );
    }

    // One row per failed agent, after that molecule's successful rows.
    // Every value column is left empty rather than filled with a
    // placeholder: no number is fabricated for a representation that was
    // never computed.
    for (const failure of result.feature_errors) {
      lines.push(
        [
          result.smiles,
          String(result.valid),
          result.error ?? "",
          failure.agent_id,
          failure.agent_version,
          "",
          agentsById.get(failure.agent_id)?.value_type ?? "",
          "",
          "",
          "",
          "",
          "",
          "",
          failure.error,
        ]
          .map(escapeCsvField)
          .join(","),
      );
    }
  }

  return lines.join("\r\n");
}
