import type { AgentMetadata } from "../types/api";

interface AgentSelectorProps {
  agents: AgentMetadata[];
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
  loading: boolean;
  error: string | null;
}

export function AgentSelector({ agents, selectedIds, onToggle, loading, error }: AgentSelectorProps) {
  return (
    <section className="panel" aria-labelledby="agents-heading">
      <h2 id="agents-heading">Available Feature Agents</h2>

      {loading && <p className="empty-state">Loading available feature agents…</p>}

      {!loading && error && (
        <p role="alert" className="error-text">
          {error}
        </p>
      )}

      {!loading && !error && agents.length === 0 && (
        <p className="empty-state">No feature agents are currently available from the backend.</p>
      )}

      {!loading && !error && agents.length > 0 && (
        <ul className="agent-list">
          {agents.map((agent) => (
            <li key={agent.id}>
              <label>
                <input
                  type="checkbox"
                  checked={selectedIds.has(agent.id)}
                  onChange={() => onToggle(agent.id)}
                />
                <span className="agent-name">{agent.name}</span>
              </label>
              <dl className="agent-meta">
                <div>
                  <dt>ID</dt>
                  <dd>{agent.id}</dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>{agent.version}</dd>
                </div>
                <div>
                  <dt>Output dim</dt>
                  <dd>{agent.output_dim === null ? "Variable" : agent.output_dim}</dd>
                </div>
                <div>
                  <dt>Output structure</dt>
                  <dd>{agent.output_structure}</dd>
                </div>
                <div>
                  <dt>Value type</dt>
                  <dd>{agent.value_type}</dd>
                </div>
                <div>
                  <dt>Requires 3D</dt>
                  <dd>{agent.requires_3d ? "Yes" : "No"}</dd>
                </div>
                <div>
                  <dt>Feature names</dt>
                  <dd>
                    {agent.feature_names
                      ? `${agent.feature_names.length} named`
                      : "Not provided"}
                  </dd>
                </div>
              </dl>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
