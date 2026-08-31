import { useEffect, useRef, useState } from "react";
import "./App.css";
import { ApiError, computeFeatures, getAgents, validateMolecules } from "./api/client";
import { AgentSelector } from "./components/AgentSelector";
import { ExportButton } from "./components/ExportButton";
import { MoleculeInputPanel } from "./components/MoleculeInputPanel";
import { MoleculeQueue } from "./components/MoleculeQueue";
import { ResultsPanel } from "./components/ResultsPanel";
import type { AgentMetadata, MoleculeResult } from "./types/api";
import type { QueueMolecule } from "./types/queue";

function errorMessage(err: unknown, fallback: string): string {
  if (err instanceof ApiError) {
    return err.message;
  }
  return fallback;
}

function App() {
  const [agents, setAgents] = useState<AgentMetadata[]>([]);
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [agentsError, setAgentsError] = useState<string | null>(null);
  const [selectedAgentIds, setSelectedAgentIds] = useState<Set<string>>(new Set());

  const [queue, setQueue] = useState<QueueMolecule[]>([]);
  const nextIdRef = useRef(0);

  const [computing, setComputing] = useState(false);
  const [computeError, setComputeError] = useState<string | null>(null);
  const [results, setResults] = useState<MoleculeResult[] | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAgents() {
      setAgentsLoading(true);
      setAgentsError(null);
      try {
        const list = await getAgents();
        if (!cancelled) {
          setAgents(list);
        }
      } catch (err) {
        if (!cancelled) {
          setAgentsError(errorMessage(err, "Failed to load feature agents."));
        }
      } finally {
        if (!cancelled) {
          setAgentsLoading(false);
        }
      }
    }

    void loadAgents();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleAddSmiles(smilesList: string[]) {
    const entries: QueueMolecule[] = smilesList.map((smiles) => ({
      id: nextIdRef.current++,
      smiles,
      status: "validating",
      error: null,
      included: true,
    }));

    setQueue((prev) => [...prev, ...entries]);
    setResults(null);

    try {
      const response = await validateMolecules(smilesList);
      const byId = new Map(entries.map((entry, index) => [entry.id, response.results[index]]));
      setQueue((prev) =>
        prev.map((molecule) => {
          const match = byId.get(molecule.id);
          if (!match) {
            return molecule;
          }
          return {
            ...molecule,
            status: match.valid ? "valid" : "invalid",
            error: match.error,
          };
        }),
      );
    } catch (err) {
      const message = errorMessage(err, "Could not validate molecules.");
      const idsToMark = new Set(entries.map((entry) => entry.id));
      setQueue((prev) =>
        prev.map((molecule) =>
          idsToMark.has(molecule.id)
            ? { ...molecule, status: "invalid", error: message }
            : molecule,
        ),
      );
    }
  }

  function handleToggleAgent(id: string) {
    setSelectedAgentIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function handleToggleIncluded(id: number) {
    setQueue((prev) =>
      prev.map((molecule) =>
        molecule.id === id ? { ...molecule, included: !molecule.included } : molecule,
      ),
    );
  }

  function handleRemove(id: number) {
    setQueue((prev) => prev.filter((molecule) => molecule.id !== id));
  }

  function handleClearQueue() {
    setQueue([]);
    setResults(null);
    setComputeError(null);
  }

  const includedSmiles = queue.filter((m) => m.included).map((m) => m.smiles);

  // An agent can become unavailable between one /agents load and the next
  // (an artifact removed under a running deployment), so a selection made
  // while it was healthy can go stale. Derived from the current metadata on
  // every render rather than tracked in state, so it cannot drift out of
  // date, and the backend rejects such a request anyway -- this just avoids
  // sending one we already know will be refused.
  const unavailableSelected = agents.filter(
    (agent) => selectedAgentIds.has(agent.id) && !agent.availability.available,
  );
  const canCompute =
    includedSmiles.length > 0 &&
    selectedAgentIds.size > 0 &&
    unavailableSelected.length === 0 &&
    !computing;

  async function handleCompute() {
    if (includedSmiles.length === 0) {
      setComputeError("Select at least one molecule in the queue to compute features for.");
      return;
    }
    if (selectedAgentIds.size === 0) {
      setComputeError("Select at least one feature agent.");
      return;
    }
    if (unavailableSelected.length > 0) {
      setComputeError(
        `Cannot compute: ${unavailableSelected
          .map((agent) => agent.id)
          .join(", ")} ${unavailableSelected.length === 1 ? "is" : "are"} currently unavailable.`,
      );
      return;
    }

    setComputing(true);
    setComputeError(null);
    try {
      const response = await computeFeatures(includedSmiles, Array.from(selectedAgentIds));
      setResults(response.results);
    } catch (err) {
      setComputeError(errorMessage(err, "Failed to compute features."));
      setResults(null);
    } finally {
      setComputing(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>MolFusion</h1>
        <p>RDKit-backed molecular feature extraction</p>
      </header>

      <main>
        <MoleculeInputPanel onAddSmiles={(smiles) => void handleAddSmiles(smiles)} />

        <MoleculeQueue
          molecules={queue}
          onToggleIncluded={handleToggleIncluded}
          onRemove={handleRemove}
          onClear={handleClearQueue}
        />

        <AgentSelector
          agents={agents}
          selectedIds={selectedAgentIds}
          onToggle={handleToggleAgent}
          loading={agentsLoading}
          error={agentsError}
        />

        <section className="panel">
          <button type="button" onClick={() => void handleCompute()} disabled={!canCompute}>
            {computing
              ? `Computing… (${includedSmiles.length} molecule(s), ${selectedAgentIds.size} agent(s))`
              : "Compute"}
          </button>
        </section>

        <ResultsPanel
          results={results}
          agents={agents}
          computing={computing}
          computeError={computeError}
        />

        <ExportButton results={results} agents={agents} />
      </main>
    </div>
  );
}

export default App;
