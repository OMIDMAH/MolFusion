export type MoleculeStatus = "validating" | "valid" | "invalid";

export interface QueueMolecule {
  /** Stable identity, independent of array position, for async updates. */
  id: number;
  smiles: string;
  status: MoleculeStatus;
  error: string | null;
  /** Whether this molecule is included when Compute is clicked. */
  included: boolean;
}
