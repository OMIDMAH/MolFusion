import type {
  AgentMetadata,
  FeatureOutput,
  MoleculeResult,
  SequenceFeatureOutput,
  VectorFeatureOutput,
} from "../types/api";

interface ResultsPanelProps {
  results: MoleculeResult[] | null;
  agents: AgentMetadata[];
  computing: boolean;
  computeError: string | null;
}

/** Descriptor names worth highlighting when the backend tells us they exist
 * on this agent (via AgentMetadata.feature_names). Never assumes a fixed
 * array index — the index is looked up by name for each render. Only
 * applies to "vector" outputs; a token sequence has no numeric values to
 * highlight this way. */
const HIGHLIGHT_DESCRIPTOR_NAMES = ["MolWt", "MolLogP", "TPSA"];

function namedHighlights(
  feature: VectorFeatureOutput,
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

function isVectorFeature(feature: FeatureOutput): feature is VectorFeatureOutput {
  return feature.output_structure === "vector";
}

function isSequenceFeature(feature: FeatureOutput): feature is SequenceFeatureOutput {
  return feature.output_structure === "sequence";
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
          {results.map((result, index) => {
            const vectorFeatures = result.features.filter(isVectorFeature);
            const sequenceFeatures = result.features.filter(isSequenceFeature);

            return (
              <article
                key={`${result.smiles}-${index}`}
                className="result-card"
                data-valid={result.valid}
              >
                <h3>
                  <span className="smiles-cell">{result.smiles}</span>{" "}
                  <span className={result.valid ? "badge badge-valid" : "badge badge-invalid"}>
                    {result.valid ? "Valid" : "Invalid"}
                  </span>
                </h3>

                {/* `error` is shown whenever populated, independent of `valid`:
                    valid=false means RDKit rejected the SMILES; valid=true with
                    a populated error means the molecule itself is fine but a
                    requested representation could not be computed for it
                    (features=[] in that case). Deriving this from `!result.valid`
                    would silently hide the latter case. */}
                {result.error && <p className="error-text">{result.error}</p>}

                {result.valid && vectorFeatures.length > 0 && (
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
                      {vectorFeatures.map((feature) => (
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

                {result.valid &&
                  sequenceFeatures.map((feature) => (
                    <div className="sequence-result" key={feature.agent_id}>
                      <p>
                        <strong>{feature.agent_id}</strong> ({feature.agent_version}) —{" "}
                        {feature.length} token{feature.length === 1 ? "" : "s"}
                      </p>
                      <div className="sequence-tokens" aria-label={`${feature.agent_id} token sequence`}>
                        {feature.tokens.join("")}
                      </div>
                    </div>
                  ))}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
