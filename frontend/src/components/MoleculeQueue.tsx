import type { QueueMolecule } from "../types/queue";

interface MoleculeQueueProps {
  molecules: QueueMolecule[];
  onToggleIncluded: (id: number) => void;
  onRemove: (id: number) => void;
  onClear: () => void;
}

function statusLabel(molecule: QueueMolecule): string {
  switch (molecule.status) {
    case "validating":
      return "Validating…";
    case "valid":
      return "Valid";
    case "invalid":
      return "Invalid";
  }
}

export function MoleculeQueue({
  molecules,
  onToggleIncluded,
  onRemove,
  onClear,
}: MoleculeQueueProps) {
  return (
    <section className="panel" aria-labelledby="molecule-queue-heading">
      <div className="panel-header">
        <h2 id="molecule-queue-heading">Molecule Queue ({molecules.length})</h2>
        <button type="button" onClick={onClear} disabled={molecules.length === 0}>
          Clear queue
        </button>
      </div>

      {molecules.length === 0 ? (
        <p className="empty-state">No molecules queued yet. Add SMILES above.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Include</th>
              <th>SMILES</th>
              <th>Status</th>
              <th>Error</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {molecules.map((molecule) => (
              <tr key={molecule.id} data-status={molecule.status}>
                <td>
                  <input
                    type="checkbox"
                    checked={molecule.included}
                    onChange={() => onToggleIncluded(molecule.id)}
                    aria-label={`Include ${molecule.smiles} in compute`}
                  />
                </td>
                <td className="smiles-cell">{molecule.smiles}</td>
                <td>{statusLabel(molecule)}</td>
                <td className="error-text">{molecule.error ?? ""}</td>
                <td>
                  <button type="button" onClick={() => onRemove(molecule.id)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
