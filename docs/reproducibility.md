# Reproducibility: artifact infrastructure

Introduced in Phase 5E. This document covers **generic artifact
infrastructure** only. No fitted or pretrained representation (TF-IDF,
Group SELFIES, ChemBERTa, GNN/VAE checkpoints, etc.) is implemented yet --
this is the loading/validation layer those will build on.

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
