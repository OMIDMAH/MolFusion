# Reproducibility: artifact and representation contracts

This document covers the pieces that must be frozen *before* any fitted
representation exists: the generic artifact infrastructure (Phase 5E), the
SMILES normalization/tokenization contracts (Phase 5F-A), and the ChEMBL 37
reference corpus (Phase 5F-B). No fitted or pretrained representation
(TF-IDF, Group SELFIES, ChemBERTa, GNN/VAE checkpoints, etc.) is
implemented yet -- these are the layers those will build on.

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

## Reference corpus: ChEMBL 37

Introduced in Phase 5F-B. The reference corpus is the frozen, unsupervised
set of molecules that a future fitted text representation (TF-IDF first)
will be fitted on. Phase 5F-B builds it and nothing else: no vectorizer is
fitted, no vocabulary or n-gram range is chosen, and no fitting parameter
appears anywhere in the output.

### Source

The official EMBL-EBI bulk release, used as a concrete versioned release
rather than an ambiguous `latest` pointer:

```
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_sqlite.tar.gz
```

The SQLite distribution is preferred over the ChEMBL web API: millions of
paginated requests would be neither deterministic nor reproducible.
Third-party repackaged copies of ChEMBL are never used. EBI publishes a
`checksums.txt` next to the archive; pass its digest to
`--expected-archive-sha256` and the build refuses to run unless the bytes
match.

Source data is an external build dependency, not a repository artifact.
The archive, the decompressed database, and the built corpus are all
ignored by `backend/.gitignore` (`/corpus_data/`) and never committed.

### Building

```
python -m molfusion_backend.corpus \
    --source-db  E:\chembl_source\chembl_37.db \
    --output-dir E:\chembl_source\corpus \
    --source-archive E:\chembl_source\chembl_37_sqlite.tar.gz \
    --expected-archive-sha256 <digest from the official checksums.txt>
```

Downloading and building are separate concerns: the builder never touches
the network, so a rebuild against an existing official database is fully
offline. It streams the source table row by row and retains only the
canonical SMILES strings -- never a DataFrame of ChEMBL, and never a
population of RDKit `Mol` objects or token tuples.

Existing finalized output is not overwritten unless `--force` is given,
and both files are written to a staging directory and moved into place
only after the whole build succeeds, so a build that fails midway can
never leave a mixture of old and new payloads.

### Pipeline

```
ChEMBL SQLite -> compound_structures -> MolFusion canonicalization
-> lossless tokenizer validation -> drop unusable records
-> deduplicate on canonical SMILES -> lexicographic sort
-> logical corpus bytes -> SHA-256 -> build report
```

**Extraction.** One query against the structural table only:

```sql
SELECT s.molregno, d.chembl_id, s.canonical_smiles
FROM compound_structures AS s
LEFT JOIN molecule_dictionary AS d ON d.molregno = s.molregno
ORDER BY s.molregno
```

`molecule_dictionary` is joined solely so a diagnostic can name a ChEMBL
accession instead of an internal `molregno`; a source without that table
still produces the identical corpus. The `ORDER BY` does not make the
corpus deterministic -- the final sort does that -- it only makes
scan-order diagnostics reproducible.

**Normalization.** Every record goes through the Phase 5F-A production
helpers (`parse_smiles` + `canonical_smiles_from_mol`, which together are
exactly `canonicalize_smiles`). The builder contains no RDKit
normalization logic of its own. No MW, LogP, Rule-of-Five, or any other
property filter is applied.

**Exclusions**, each counted in its own category so no record can vanish
silently: `null_smiles`, `empty_smiles` (empty or whitespace-only),
`rdkit_parse_failures`, `zero_atom_molecules`. Phase 5F-A deliberately
allows `"" -> ""` at the helper level; the corpus does not inherit that,
and an empty structure is never emitted as a document.

**Tokenizer validation.** Every corpus document must satisfy
`"".join(tokenize_smiles(s)) == s`. The input here is our own normalizer's
output, so a violation is a contract breach between 5F-A's two halves, not
input noise: the build **aborts** and names the offending SMILES.
`--allow-tokenizer-failures` downgrades that to an excluded, counted
record for a deliberately documented exception, and the report always
records that it was used, so a lenient build can never be mistaken for a
clean one. This is validated with an explicit check, never a bare
`assert`, which `python -O` would strip.

Deduplication keys on the exact canonical string, so validating each
distinct canonical SMILES once is equivalent to validating every record
that produced it.

**Deduplication** happens only *after* canonicalization, keyed on the
canonical isomeric SMILES -- never on the raw ChEMBL string, `molregno`,
ChEMBL ID, InChIKey, or a non-isomeric form. Distinct stereoisomers stay
distinct because the canonical form distinguishes them.

**Ordering** is `sorted(unique_canonical_smiles)`: Python's Unicode
code-point ordering, which is locale-independent. The corpus bytes are
identical for any source row order, any output path, and any host locale.

### Logical serialization contract

Frozen as `utf8_lf_sorted_unique_final_newline_v1`:

| Property | Value |
| --- | --- |
| Encoding | UTF-8, no BOM |
| Newline | LF only |
| Layout | one canonical SMILES per line, no header |
| Contents | unique entries, lexicographically sorted |
| Final newline | yes |

For N molecules the logical bytes are exactly:

```python
("\n".join(sorted_unique_smiles) + "\n").encode("utf-8")
```

and the empty corpus is `b""` -- zero lines means zero terminators, not a
lone newline. Every write goes through raw binary I/O, so a Windows build
produces the same bytes as a Linux one; the code never opens the corpus in
text mode. A record containing a line break is rejected rather than
silently split into two documents.

### `fit_corpus_sha256`

> SHA-256 of the exact normalized logical corpus bytes that will later be
> supplied to fitting.

It is therefore independent of SQLite physical layout, gzip/tar metadata,
source row ordering, Windows vs Linux newline defaults, download time, and
file modification time. It is computed over the corpus in the same
streaming pass that writes it, so it is necessarily the digest of the
bytes that landed on disk.

**Three different checksums exist and must never be conflated:**

| Field | Identifies |
| --- | --- |
| `source.assets.archive.sha256` | the exact downloaded archive bytes |
| `source.assets.database.sha256` | the decompressed SQLite database |
| `fit_corpus.sha256` | the logical corpus -- the scientific identity |

The first two are provenance: they say which bytes a build consumed. Only
the third identifies the corpus itself. An archive repack or a SQLite
`VACUUM` changes the first two without changing the third, and that is
correct.

### Build report

`corpus_build_report.json` records the contract IDs (normalization,
tokenizer, serialization), full source provenance including checksum
verification status, every filtering and deduplication count, the fit
corpus digest/size/document count, corpus statistics, and the Python,
RDKit, SQLite and MolFusion git versions that produced it.

Exactly one field is volatile: `build.built_at`. Everything else is a
function of the source database and the frozen contracts, so two builds of
the same source agree everywhere else -- which is what makes comparing
`deterministic_report_view(report)` a meaningful determinism check. The
timestamp never enters the corpus hash.

Statistics are corpus *shape* only -- document count, SMILES length and
token count (min/max/mean/median), and how many entries carry disconnected
components or stereochemistry. Vocabulary, document-frequency, n-gram and
TF-IDF analysis are deliberately absent; they are Phase 5F-C.

### Leakage policy

The corpus is an external, unsupervised molecular reference corpus, built
independently of every downstream supervised task. No target, assay,
activity measurement, potency value, ADMET endpoint, publication, maximum
phase, or benchmark-membership flag is read -- the builder opens the
structural table and nothing else, and the report asserts
`source.uses_downstream_labels: false`. No train/validation/test
assignment, TDC label, or MolFusion model result is consulted.

Unlabeled molecules may of course overlap between ChEMBL and a future
benchmark. That is an unsupervised-pretraining exposure question to be
documented and audited separately; no downstream-data exclusion logic
belongs in this phase.

## Vocabulary study: token n-grams (Phase 5F-C)

Phase 5F-C measures what a SMILES TF-IDF vocabulary would look like if it
were fitted on the frozen corpus, so that the n-gram range, pruning
threshold, dimension and feature-ranking rule are chosen from evidence
rather than convention. It is a **study**: it fits nothing, freezes
nothing, and writes no production payload. Phase 5F-D does that, after the
policy this phase recommends has been reviewed.

```
python -m molfusion_backend.corpus.study \
    --corpus     backend/corpus_data/chembl37/canonical_smiles.smi \
    --output-dir backend/corpus_data/chembl37/studies/ngram_vocabulary
```

The corpus is immutable input. The study opens it read-only, twice, and
never re-canonicalizes, re-sorts, re-deduplicates, rewrites its build
report, or reopens the ChEMBL SQLite release.

### Corpus identity gate

The first operation is a SHA-256 of the corpus, compared against the
frozen `fit_corpus_sha256`. A mismatch raises `CorpusIdentityError` and
aborts before a single molecule is read. This is a hard stop rather than a
warning: a study run against different bytes is not a weaker result, it is
a result about something else wearing the same label. The expected
document count is checked the same way after the counting pass.

### N-grams are token tuples, never concatenated strings

An n-gram is a tuple of Phase 5F-A tokens. Concatenation is genuinely
lossy for this tokenizer, because a multi-character token can be re-split
at a different boundary:

```
("Cl", "C")  and  ("C", "lC")   both concatenate to "ClC"
```

A string-keyed counter merges those into one feature and corrupts every
count derived from it. Tuple keys make the collision impossible, and the
human-readable study outputs serialize each n-gram as a JSON array
(`["Cl", "C"]`) so the property survives the round trip to CSV.

### DF and TF are tracked separately

| | Meaning |
| --- | --- |
| `DF` | molecules containing the n-gram **at least once** |
| `TF` | **total occurrences** across all molecules |

A molecule whose SMILES contains `C` twenty times contributes `DF += 1`
and `TF += 20`. The two are never interchangeable, and which of them
should drive feature selection is one of the questions this phase exists
to answer, so both are carried through every table.

Counts are additionally split by study subset and by molecule token-count
band (`(0,32]`, `(32,64]`, `(64,128]`, `(128,256]`, `(256,512]`,
`(512,inf)`). The bands are what make the long-molecule question
answerable after the fact: TF restricted to short molecules is a sum of
band slots, so "would this ranking change if unusually long molecules were
dropped?" needs no second pass over the corpus. No molecule is ever
excluded or length-filtered; the bands only record where its counts came
from.

### The study split

An analysis-only holdout, defined so that membership is a pure function of
the molecule:

```python
digest = hashlib.sha256(smiles.encode("utf-8")).digest()
bucket = int.from_bytes(digest, byteorder="big", signed=False) % 20
holdout = (bucket == 0)          # ~5%; buckets 1-19 are the fit subset
```

Every step is pinned -- UTF-8 encoding, SHA-256, the whole 32-byte digest,
big-endian, unsigned, modulo 20 -- under

```
STUDY_SPLIT_ID = "sha256_utf8_digest_bigendian_unsigned_mod20_bucket0_holdout_v1"
```

because a byte-order or signedness change would silently reshuffle the
split while still looking deterministic.

Deliberately **not** a positional split: the corpus is lexicographically
sorted, so a contiguous slice is a slice of chemical space rather than a
sample of it, and bucket membership would change if the corpus were ever
re-sorted. Deliberately **not** `random`: that would make the split depend
on a seed and a Python version rather than on the data. Under the hash
rule, adding, removing or reordering other molecules cannot move a given
molecule.

This is an analysis split and nothing else. It is not a downstream
train/test split, it never touches a TDC dataset or an ADMET label, and
the Phase 5F-D production artifact will still be fitted on all 2,897,639
reference molecules.

### Deterministic feature ranking

MolFusion ranks its own features rather than delegating to a vectorizer's
`max_features`:

```
sort key:  (-frequency, ngram_token_tuple)
```

The tie-break is the token tuple itself, compared element by element, so
the order is **total** -- no two distinct n-grams can tie, and no ranking
position is ever resolved by insertion order, hash seed, sort stability or
locale. A rule stated this way can be re-applied years later from the
recorded counts alone; a rule that lives inside a third-party library's
sort is reproducible only for as long as that library's internals are, and
its tie behaviour is not part of its public contract.

One structural consequence is used throughout. Under descending-DF
ranking, `{DF >= min_df}` is exactly a **prefix** of the ranking, so every
`min_df` threshold is also a dimension cap and vice versa. Candidate
vocabularies for a given ranking are therefore nested, which is both a
scientific simplification and what keeps the holdout sweep affordable: one
dictionary lookup per n-gram per molecule answers every candidate
dimension at once.

Thresholds are absolute molecule counts, never percentages -- an absolute
threshold can be audited against this fixed corpus by inspection, whereas
a percentage silently changes meaning if the corpus size does.

### What is measured

Per n-gram order (1, 2, 3): distinct vocabulary size, total occurrences,
DF and TF summaries, and a cumulative rarity histogram
(`DF <= 1, 2, 5, 10, 25, 50`) with fractions. Per `min_df` threshold
(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000): unigram, bigram and trigram
vocabulary sizes and the combined `(1,1)`, `(1,2)`, `(1,3)`, `(2,3)`
sizes. Per candidate policy, ranking and dimension: holdout occurrence
coverage, holdout unique-n-gram coverage, the per-molecule OOV
distribution, all-zero molecule count, retained-feature counts and the
resulting sparsity, and whether every unigram survived the cut.

Percentiles are **nearest-rank** (`p` is the `ceil(p/100 * n)`-th smallest
observed value, never interpolated), stated explicitly because "the 95th
percentile" is not a single definition.

Two coverage notions are reported and must not be conflated: aggregate
occurrence coverage pools all holdout occurrences, while the mean
per-molecule OOV fraction weights every molecule equally. Pooling flatters
long molecules; averaging does not.

### Exact counts, and why they fit in memory

The counters are exact Python dictionaries, not sketches. That is a
measured choice, not an assumption: canonical SMILES have a token alphabet
of a few hundred symbols, so the distinct n-gram vocabulary is thousands
of entries rather than millions, and the counting pass over the full
corpus peaks well under 100 MB. The study report records its own
`run.peak_memory_bytes` and pass timings so the claim is re-checkable on
any machine. Nothing is retained per molecule: the corpus is streamed, and
each document's token tuple and n-gram counts are consumed and dropped.

### Study outputs

Written to an ignored location alongside the corpus:

```
backend/corpus_data/chembl37/studies/ngram_vocabulary/
  study_report.json        every table, plus full provenance
  df_thresholds.csv        vocabulary size per min_df, per order
  vocabulary_coverage.csv  policy x min_df vocabulary sizes
  holdout_coverage.csv     the candidate-configuration table
  ranking_comparison.csv   DF vs TF top-K agreement
  top_ngrams.csv           bounded diagnostic sample of each ranking
```

Generated study output is ignored by git for the same reason the corpus is
(`/corpus_data/`); only the study code, its tests, and the measured
conclusions in documentation are committed. The top-n-gram dump is bounded
to a few hundred rows per order on purpose -- the point is to see what a
ranking promotes, which a short table shows and a multi-million-row dump
hides.

`study_report.json` records the study schema version, the verified corpus
digest, document count, the normalization/tokenizer/serialization contract
IDs, the split definition and its counts, the n-gram and ranking
definitions, every threshold swept, and the Python, RDKit and MolFusion
git versions. Exactly five fields are volatile -- the start timestamp,
three timings and peak memory -- and `deterministic_study_view()` strips
them, so two runs over the same corpus are otherwise identical, including
byte-for-byte identical CSV tables. No result depends on dict iteration
order, set order, OS locale, filesystem ordering, or process completion
order.

### Leakage policy

The study reads the frozen corpus and nothing else. No TDC dataset, ADMET
endpoint, benchmark split, label, or MolFusion prediction output is
opened; `corpus.uses_downstream_labels` is asserted `false` in the study
report as it is in the build report. The unseen-molecule estimate comes
entirely from the internal hash holdout described above.
