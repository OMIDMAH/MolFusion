import { useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { parseSmilesCsv } from "../lib/csv";

interface MoleculeInputPanelProps {
  onAddSmiles: (smiles: string[]) => void;
  disabled?: boolean;
}

function splitManualInput(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export function MoleculeInputPanel({ onAddSmiles, disabled = false }: MoleculeInputPanelProps) {
  const [manualText, setManualText] = useState("");
  const [csvError, setCsvError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function handleAddManual() {
    const smiles = splitManualInput(manualText);
    if (smiles.length === 0) {
      return;
    }
    onAddSmiles(smiles);
    setManualText("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : "";
      const parsed = parseSmilesCsv(text);
      if (parsed.error) {
        setCsvError(parsed.error);
        return;
      }
      setCsvError(null);
      onAddSmiles(parsed.smiles);
    };
    reader.onerror = () => {
      setCsvError("Could not read the selected file.");
    };
    reader.readAsText(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  return (
    <section className="panel" aria-labelledby="molecule-input-heading">
      <h2 id="molecule-input-heading">Molecule Input</h2>

      <div className="input-row">
        <label htmlFor="manual-smiles">Manual SMILES entry (one per line)</label>
        <textarea
          id="manual-smiles"
          rows={4}
          value={manualText}
          onChange={(event) => setManualText(event.target.value)}
          placeholder={"CCO\nc1ccccc1"}
          disabled={disabled}
        />
        <button type="button" onClick={handleAddManual} disabled={disabled || manualText.trim() === ""}>
          Add to queue
        </button>
      </div>

      <div className="input-row">
        <label htmlFor="csv-upload">Upload CSV (must contain a "smiles" column)</label>
        <input
          id="csv-upload"
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleFileChange}
          disabled={disabled}
        />
        {csvError && (
          <p role="alert" className="error-text">
            {csvError}
          </p>
        )}
      </div>
    </section>
  );
}
