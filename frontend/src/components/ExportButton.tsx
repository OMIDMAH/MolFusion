import type { AgentMetadata, MoleculeResult } from "../types/api";
import { buildResultsCsv } from "../lib/csv";

interface ExportButtonProps {
  results: MoleculeResult[] | null;
  agents: AgentMetadata[];
}

export function ExportButton({ results, agents }: ExportButtonProps) {
  const disabled = !results || results.length === 0;

  function handleExport() {
    if (!results || results.length === 0) {
      return;
    }
    const csv = buildResultsCsv(results, agents);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "molfusion_results.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <section className="panel" aria-labelledby="export-heading">
      <h2 id="export-heading">Export</h2>
      <button type="button" onClick={handleExport} disabled={disabled}>
        Export results as CSV
      </button>
    </section>
  );
}
