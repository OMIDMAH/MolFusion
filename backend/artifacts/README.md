# Artifact root

This directory is the default root for fitted/pretrained MolFusion
artifacts (see `docs/reproducibility.md` for the full policy). It is
intentionally empty in version control except for this file -- no fitted
artifacts exist yet as of Phase 5E.

Layout convention:

```
artifacts/
  <artifact_type>/
    <artifact_id>/
      <artifact_version>/
        metadata.json
        <payload files...>
```

Hypothetical example (not implemented):

```
artifacts/
  tfidf/
    pubchem_smiles_tfidf/
      1.0.0/
        metadata.json
        vectorizer.joblib
```

Override the root with the `MOLFUSION_ARTIFACT_ROOT` environment variable
if artifacts should live somewhere other than `<backend>/artifacts`.
