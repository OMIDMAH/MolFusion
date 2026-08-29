# Reproducibility: artifact and representation contracts

This document covers the pieces that must be frozen *before* any fitted
representation exists: the generic artifact infrastructure (Phase 5E) and
the SMILES normalization/tokenization contracts (Phase 5F-A). No fitted or
pretrained representation (TF-IDF, Group SELFIES, ChemBERTa, GNN/VAE
checkpoints, etc.) is implemented yet -- these are the layers those will
build on.

## What "artifact" means here

An **artifact** is an immutable, versioned, checksummed file (or set of
files) on disk that a future `FeatureAgent` needs to produce its output
without re-fitting or re-downloading it at request time -- e.g. a fitted
TF-IDF vectorizer, a pretrained model checkpoint, or a vocabulary file.

This is distinct from **agent code**, which ships in the repo and is
versioned by `agent.version` on the `FeatureAgent` itself.

## `agent_version` vs `artifact_version`

These are independently versioned and must not be conflated:

- `agent_version` describes the *code* that computes a representation
  (e.g. `smiles_tfidf` agent, code version `1.1.0`).
- `artifact_version` describes one specific *fitted/pretrained payload*
  that agent code loads (e.g. `chem_reference_tfidf` artifact, version
  `1.0.0`).

The same agent code version may load different artifact versions over
time (e.g. re-fit on a larger corpus), and the same artifact version may
be loaded by successive agent code versions (e.g. a bugfix release that
doesn't change the fitted vectorizer). Bumping one must never silently
imply a change to the other.

## Directory convention

```
artifacts/
  <artifact_type>/
    <artifact_id>/
      <artifact_version>/
        metadata.json
        <payload files...>
```

Hypothetical example (TF-IDF is **not implemented** -- illustration only):

```
artifacts/
  tfidf/
    pubchem_smiles_tfidf/
      1.0.0/
        metadata.json
        vectorizer.joblib
```

The default root is `<backend>/artifacts`, resolved relative to the
installed backend package location -- never the current working
directory. Override it with the `MOLFUSION_ARTIFACT_ROOT` environment
variable when artifacts should live elsewhere (e.g. a shared model store).
No environment variable is required for normal development.

## Metadata requirements

Every artifact directory must contain a `metadata.json` validated against
`molfusion_backend.artifacts.metadata.ArtifactMetadata`:

- `artifact_id`, `artifact_version`, `artifact_type` -- must match the
  directory path exactly; the loader rejects a mismatch.
- `created_at` -- when the artifact was produced.
- `library_versions` -- versions of the libraries used to produce it
  (e.g. `{"scikit-learn": "1.5.0"}`).
- `configuration` -- the parameters used to produce it (e.g. TF-IDF
  `ngram_range`, `max_features`).
- `payload_files` -- every payload filename paired with its own SHA-256,
  so a file and its checksum can never drift apart.
- `fit_corpus` -- nullable; provenance (name, version, checksum, record
  count, source) of the corpus used, for artifacts that involved fitting.
- `random_seed` -- nullable; the seed used, or `null` if nothing in the
  fitting process was stochastic.
- `description` -- optional free text.

## Checksum verification

Every payload file's SHA-256 (stdlib `hashlib`, streaming/chunked
reads -- no extra dependency) is verified against `metadata.json` on every
load. A mismatch raises `ArtifactChecksumError` and the artifact is not
returned.

## Load-or-fail

`molfusion_backend.artifacts.load_artifact(artifact_type, artifact_id,
artifact_version)` either returns a fully validated, checksum-verified
`ArtifactDescriptor`, or raises. There is no partial success and no
silent fallback:

- an artifact is never refit at request time if its files are missing or
  invalid -- that is a deployment error, not a runtime recovery path;
- an artifact is never silently replaced by a different version just
  because the requested one failed to load;
- a requested `artifact_type`/`artifact_id`/`artifact_version` that
  doesn't match what `metadata.json` declares is always rejected, never
  coerced.

The loader is deliberately representation-agnostic: it validates and
resolves paths, but does not know how to deserialize a `.joblib`,
`.pt`, or any other payload format. Turning `payload_paths` into a
usable object (e.g. `joblib.load(...)`) is each consumer's job.

## SMILES normalization and tokenization contracts

Introduced in Phase 5F-A. Any future representation fitted on a corpus of
SMILES text (TF-IDF first) depends on two deterministic string-level
contracts *before* any fitting happens, because they decide corpus
deduplication, the corpus checksum, the vocabulary, document frequencies,
and runtime transformation alike. Like `agent_version` and
`artifact_version`, they are versioned independently of each other and of
the agent code, and each carries its own identifier for an artifact's
`metadata.json` to record:

| Contract | Identifier | Implementation |
| --- | --- | --- |
| Normalization | `rdkit_canonical_isomeric_smiles_v1` | `molfusion_backend.chemistry` |
| Tokenization | `rdkit_smiles_lexer_v1` | `molfusion_backend.smiles_tokenizer` |

They are two separate steps and are never fused; the tokenizer does not
canonicalize its input:

```python
canonical = canonicalize_smiles(raw_smiles)
tokens = tokenize_smiles(canonical)
```

### Normalization: `rdkit_canonical_isomeric_smiles_v1`

`chemistry.canonicalize_smiles(smiles)` parses with RDKit and writes
canonical isomeric SMILES (`canonical=True`, `isomericSmiles=True`,
`kekuleSmiles=False`); `chemistry.canonical_smiles_from_mol(mol)` is the
same serialization for an already-parsed `Mol`, and is what the SELFIES
agent uses, so one contract governs every consumer.

This is **serialization, not molecular standardization**. Deliberately
absent, and not to be added under this identifier:

- no salt stripping or largest-fragment selection
- no neutralization or charge normalization
- no tautomer canonicalization
- no stereochemistry removal
- no benchmark-specific preprocessing

Stereochemistry, isotopes, formal charges, disconnected components and
explicit bracket expressions all survive verbatim; aromatic rings are
written in aromatic (not Kekule) form, so each molecule has exactly one
serialization. Equivalent spellings (`CCO`, `OCC`) collapse to one string;
opposite stereoisomers do not. Unparseable input raises `ValueError`.
`""` is valid input and canonicalizes to `""` -- a successful empty
result, distinct from a parse failure.

### Tokenization: `rdkit_smiles_lexer_v1`

`smiles_tokenizer.tokenize_smiles(smiles)` returns a `tuple[str, ...]`. It
is purely lexical -- no RDKit parse, no vocabulary, no integer IDs, no
padding, no embeddings -- and **lossless**:

```python
"".join(tokenize_smiles(s)) == s
```

Tokens are consecutive non-overlapping matches covering the whole string,
so nothing can be dropped or rewritten. Case is never folded (`C` and `c`
are different atoms). Bracket atoms stay indivisible (`[NH4+]`, `[C@@H]`,
`[13CH3]`, `[nH]`, `[Co@OH24]`), and `Cl`/`Br` are single tokens.
Recognized: organic-subset and aromatic atoms, `*`, branches, `.`, every
bond symbol the pinned RDKit emits (`- = # $ : / \ ~` plus the dative
`->` / `<-`), and ring closures in all three forms RDKit writes -- `1`,
`%12`, and the extended `%(100)` form it switches to past 99
simultaneously-open rings (verified against the installed RDKit, not
assumed from an older tokenizer regex).

Unrecognized input raises `ValueError` naming the offending offset; the
lexer never skips a character or returns a shortened sequence. `""`
tokenizes to `()` -- again a successful empty result, not a failure.
