import type { AgentMetadata, FeatureVector, MoleculeResult } from "../types/api";

interface ResultsPanelProps {
  results: MoleculeResult[] | null;
  agents: AgentMetadata[];
  computing: boolean;
  computeError: string | null;
}

/** Descriptor names worth highlighting when the backend tells us they exist
 * on this agent (via AgentMetadata.feature_names). Never assumes a fixed
 * array index — the index is looked up by name for each render. */
const HIGHLIGHT_DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "TPSA"];

function namedHighlights(
  feature: FeatureVector,
  agentsById: Map<string, AgentMetadata>,
): Array<{ name: string; value: number }> {
  const agent = agentsById.get(feature.agent_id);
  if (!agent || !agent.feature_names) {
    return [];
  }
  const highlights: Array<{ name: string; value: number }> = [];
  for (const name of HIGHLIGHT_DESCRIPTOR_NAMES) {
    const index = agent.feature_names.indexOf(name);
    if (index !== -1 && index < feature.values.length) {
      highlights.push({ name, value: feature.values[index] });
    }
  }
  return highlights;
}

export function ResultsPanel({ results, agents, computing, computeError }: ResultsPanelProps) {
  const agentsById = new Map(agents.map((agent) => [agent.id, agent]));

  return (
    <section className="panel" aria-labelledby="results-heading" data-testid="results-panel">
      <h2 id="results-heading">Results</h2>

      {computing && <p className="empty-state">Computing feature vectors, please wait…</p>}

      {!computing && computeError && (
        <p role="alert" className="error-text">
          {computeError}
        </p>
      )}

      {!computing && !computeError && (!results || results.length === 0) && (
        <p className="empty-state">No results yet. Add molecules, choose agents, and click Compute.</p>
      )}

      {!computing && !computeError && results && results.length > 0 && (
        <div className="results-list">
          {results.map((result, index) => (
            <article key={`${result.smiles}-${index}`} className="result-card" data-valid={result.valid}>
              <h3>
                <span className="smiles-cell">{result.smiles}</span>{" "}
                <span className={result.valid ? "badge badge-valid" : "badge badge-invalid"}>
                  {result.valid ? "Valid" : "Invalid"}
                </span>
              </h3>

              {!result.valid && <p className="error-text">{result.error}</p>}

              {result.valid && result.features.length > 0 && (
                <table>
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Version</th>
                      <th>Output dim</th>
                      <th>Highlights</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.features.map((feature) => (
                      <tr key={feature.agent_id}>
                        <td>{feature.agent_id}</td>
                        <td>{feature.agent_version}</td>
                        <td>{feature.dim}</td>
                        <td>
                          {namedHighlights(feature, agentsById)
                            .map((h) => `${h.name}: ${h.value.toFixed(3)}`)
                            .join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
