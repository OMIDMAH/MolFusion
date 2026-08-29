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

const EXPORT_HEADER = [
  "smiles",
  "valid",
  "error",
  "agent_id",
  "agent_version",
  "dim",
  "feature_names",
  "values",
];

/** Build a CSV export directly from a /features/compute API response.
 * One row per (molecule, feature vector); invalid molecules get a single
 * row with empty agent columns. `agents` supplies each agent's
 * `feature_names` (from GET /agents) so the export can label values by
 * name where the backend exposes them — nothing here is recomputed,
 * reordered, or fabricated. */
export function buildResultsCsv(results: MoleculeResult[], agents: AgentMetadata[]): string {
  const agentsById = new Map(agents.map((agent) => [agent.id, agent]));
  const lines = [EXPORT_HEADER.map(escapeCsvField).join(",")];

  for (const result of results) {
    if (result.features.length === 0) {
      lines.push(
        [result.smiles, String(result.valid), result.error ?? "", "", "", "", "", ""]
          .map(escapeCsvField)
          .join(","),
      );
      continue;
    }

    for (const feature of result.features) {
      const featureNames = agentsById.get(feature.agent_id)?.feature_names;
      lines.push(
        [
          result.smiles,
          String(result.valid),
          result.error ?? "",
          feature.agent_id,
          feature.agent_version,
          String(feature.dim),
          featureNames ? featureNames.join(";") : "",
          feature.values.join(";"),
        ]
          .map(escapeCsvField)
          .join(","),
      );
    }
  }

  return lines.join("\r\n");
}
