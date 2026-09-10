# Reproducibility release plan — prepared, not executed

**No Git tag, GitHub release or Zenodo archive was created.** You chose
"prepare the plan only", and this document is the plan. Everything below is
a proposal; nothing here has been applied to the repository.

---

## 1. Why a tag rather than a merge

PR #1 (`develop` → `main`) is open and carries 24 commits, 180 files and
the entire benchmark stack behind a title about the v2 frontend. Merging it
days before submission would change the branch architecture and the default
branch at the worst possible moment, and it would do so for a reason
unrelated to the science.

An immutable tag on the exact manuscript commit achieves what the
availability statement actually needs — a permanent, citable pointer to the
code that produced the results — without touching branch structure. The
merge can then happen on its own schedule, on its own merits.

## 2. Tag name candidates

| Candidate | Reads as | Assessment |
| --- | --- | --- |
| `molfusion-paper-v1.0.0` | the paper, semantically versioned | Clear, but "v1.0.0" implies a software release line that does not exist; a second paper would need `v2.0.0`, which is misleading |
| **`molfusion-jcheminform-2026-09`** | the artifact submitted to this journal, at this date | **Recommended.** Names the venue and the moment, so a later submission elsewhere gets its own unambiguous tag and neither collides |
| `molfusion-benchmark-2026-09` | the benchmark, at this date | Accurate but under-specified: the tag exists to support *this manuscript*, not the benchmark in general |

**Recommendation: `molfusion-jcheminform-2026-09`.**

## 3. Exact commit to tag

The tag should point at the commit that contains the final submission
package, i.e. the commit produced by this phase.

```
commit:  <THIS PHASE'S COMMIT — see git log after the 6C.7 commit lands>
branch:  develop
```

Rationale for tagging this commit rather than an earlier one: it is the
first commit at which the manuscript, the verified bibliography, the
supplementary data exports and the submission package all exist together.
Earlier commits reproduce the *science* but not the submitted artifact.

**The science itself is older and is separately identified** — the tag does
not imply the results were produced by it:

| Component | Commit |
| --- | --- |
| Track A1 execution | `459653b`, `ddabb42`, `2bcb467` |
| Track A2 execution | `e6ae297` |
| Track A1 analysis | `fe4bc60` |
| Track A2 analysis | `15b78a2` |
| Provenance hardening (after execution) | `89335dc` |
| Publication evidence package | `0da5bf9` |

## 4. Commands, for when this is authorised

```bash
git tag -a molfusion-jcheminform-2026-09 <commit> \
  -m "MolFusion manuscript submitted to Journal of Cheminformatics"
git push origin molfusion-jcheminform-2026-09

gh release create molfusion-jcheminform-2026-09 \
  --title "MolFusion — Journal of Cheminformatics submission" \
  --notes-file docs/manuscript/submission/jcheminformatics/metadata/RELEASE_NOTES.md
```

Note that a tag on `develop` is reachable and permanent regardless of what
later happens to the branch, so this remains valid even if PR #1 is
eventually merged, rebased or closed.

## 5. Release notes draft

> **MolFusion — Journal of Cheminformatics submission**
>
> Source state for the manuscript *MolFusion: Probe-Dependent Performance of
> Molecular Representations Across 22 ADMET Endpoints*.
>
> This release contains the MolFusion framework, the frozen benchmark
> protocol, the Track A1 and Track A2 execution and analysis code, the
> publication evidence package and the manuscript source. It is a source
> reference for the article, not a new software version.
>
> **Frozen scientific identities**
>
> | Item | Identity |
> | --- | --- |
> | Benchmark release | `TDC-ADMET-2026-09` |
> | Track A1 scientific identity | `d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868` |
> | Track A2 scientific identity | `9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14` |
> | Publication evidence identity | `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18` |
> | TF-IDF artifact | `chembl37_token_ngrams_1_3` v1.0.0 |
>
> **Environment:** Python 3.11.15, RDKit 2026.03.5, NumPy 2.4.6,
> scikit-learn 1.9.0, SciPy 1.17.1.
>
> **Execution history.** Track A1 ran at `459653b`, `ddabb42` and `2bcb467`;
> Track A2 at `e6ae297`. The provenance-hardening work at `89335dc` came
> afterwards and did not produce these results. 338 of 616 result shards
> lack an embedded commit identifier owing to a logging defect in the
> runner as it existed at execution time; the values and their
> content-derived identities were unaffected, attribution was verified by
> audit, and no shard was modified. See the Supplementary Information.
>
> **Not included:** downloaded benchmark datasets (available from the
> Therapeutics Data Commons), the ChEMBL 37 corpus (available from the
> EBI), feature caches and generated result artifacts. All are reproducible
> from this source plus the public inputs, and are verifiable against the
> identities above.

## 6. Release manifest

| Included | Why |
| --- | --- |
| `backend/src/molfusion_backend/` | framework, benchmark, analysis, publication and submission code |
| `backend/tests/` | the guard suite, including every manuscript and submission guard |
| `backend/benchmark_manifests/tdc_admet_group.json` | provenance manifest with per-endpoint checksums — makes the ignored dataset bytes verifiable |
| `docs/` | protocol, data, execution, analysis and publication-evidence documentation |
| `docs/manuscript/` | manuscript sources, evidence maps, references, submission package |
| `pyproject.toml`, `uv.lock` | pinned environment |

| Excluded | Why |
| --- | --- |
| `backend/benchmark_data/` | external TDC download; redistributable only under TDC's terms |
| `backend/corpus_data/` | ChEMBL-derived; multi-GB and externally available |
| `backend/benchmark_cache/` | feature caches, reproducible and large |
| `backend/benchmark_runs/` | generated shards, results, analysis and figures — reproducible, and identified by the scientific identities above |
| Generated `.docx` / `.pdf` | build products of the tracked source |

The exclusions are why the identities matter: they are what lets a reader
confirm a regenerated result set matches the published one without the
repository carrying hundreds of megabytes.

## 7. Zenodo archive checklist

Zenodo archiving is **optional** and was not performed. If wanted:

1. Link the GitHub repository in Zenodo (Settings → GitHub → enable
   `OMIDMAH/MolFusion`). This must be done **before** the release is
   published — Zenodo only archives releases created after linking.
2. Create the tag and GitHub release above.
3. Confirm Zenodo minted a DOI for the release.
4. Add the DOI to the manuscript's *Availability of code* statement,
   replacing the branch-and-commit wording.
5. Add the DOI to `references.json` if the framework is to be cited as
   software in its own right.
6. Regenerate the submission package so the statement propagates:
   `python -m molfusion_backend.benchmark.submission_cli`.

Step 1 is the one that is easy to get wrong; a release created before
linking is not archived and the DOI never appears.

## 8. Effect on the availability statement

**Until a tag exists**, the statement continues to name the `develop`
branch and the exact commits, and to say plainly that the default branch
does not contain this code. That wording is currently accurate and must not
be pre-emptively changed.

**Once a tag or archive DOI exists**, the statement should prefer the
permanent identifier:

> The code that produced the results reported in this article is archived at
> `<DOI>` and tagged `molfusion-jcheminform-2026-09` in the repository at
> <https://github.com/OMIDMAH/MolFusion>.

That edit belongs in `metadata/declarations.md`, after which the submission
package is regenerated. It has not been made.
