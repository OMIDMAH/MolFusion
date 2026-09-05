#### Availability of data and materials

The benchmark datasets analysed in this study are the ADMET benchmark group
of the Therapeutics Data Commons, which is publicly available at
<https://tdcommons.ai>. Data were obtained once from that source and
re-serialised into a frozen MolFusion dataset release, `TDC-ADMET-2026-09`,
under a deterministic serialisation contract; every endpoint file carries a
SHA-256 checksum and the release as a whole carries a content-derived
release identity. The provenance manifest containing those checksums is
distributed with the source code, so the frozen release can be verified
byte-for-byte after regeneration from the public source. The downloaded
datasets themselves are not redistributed, because they are available from
the Therapeutics Data Commons under its own terms.

The SMILES TF-IDF representation depends on a vocabulary and inverse
document frequency weights fitted on ChEMBL 37, which is publicly available
at <https://www.ebi.ac.uk/chembl/>.

Generated benchmark artefacts — per-cell result shards, collected result
tables, analysis outputs, figures and the publication evidence package —
are reproducible from the frozen release plus the source code and are not
tracked in the repository. The result sets are identified by
content-derived scientific identities, which are reported in the
Supplementary Information so that a regenerated result set can be checked
against the values underlying this article:

- Track A1 scientific identity:
  `d40ef09b398f47914aa51f99fd6a4f5893f7778b50c0cca04404b575632de868`
- Track A2 scientific identity:
  `9dd5dfa6067c8a760b0bb8fb39648f71f662f2fa1bbf4cc5d7cb0cd495a69f14`
- Publication evidence identity:
  `5790359bb24db13653750d9c43075b53b3e47edd7156910f9dab9d8853c49a18`

**[AUTHOR INPUT REQUIRED]** — if the journal or the authors wish to deposit
the generated result tables and the publication evidence package in a
persistent archive (for example Zenodo), that deposition and its DOI must
be created before submission. No archive has been created at the time of
writing.

#### Availability of code

MolFusion is open source and is available at
<https://github.com/OMIDMAH/MolFusion>.

The code that produced the results reported in this article is on the
`develop` branch. **At the time of writing the repository's default branch
(`main`) does not contain this code**; a pull request from `develop` to
`main` is open and unmerged. Readers should therefore use the `develop`
branch, or the specific commits below, rather than the default branch.

| Component | Commit |
| --- | --- |
| Track A1 execution | `459653b`, `ddabb42`, `2bcb467` |
| Track A2 execution | `e6ae297` |
| Track A1 analysis | `fe4bc60` |
| Track A2 analysis | `15b78a2` |
| Provenance hardening (after execution) | `89335dc` |
| Publication evidence package | `0da5bf9` |

**[AUTHOR INPUT REQUIRED]** — a Git tag and an archived release DOI would
give readers a permanent identifier and are recommended before submission.
Neither has been created, and creating them was outside the scope of the
work described here.

#### Competing interests

**[AUTHOR INPUT REQUIRED]**

#### Funding

**[AUTHOR INPUT REQUIRED]**

#### Authors' contributions

**[AUTHOR INPUT REQUIRED]**

#### Acknowledgements

**[AUTHOR INPUT REQUIRED]**

#### Ethics approval and consent to participate

Not applicable. This study used only publicly available molecular property
datasets and involved no human participants, human data or animals.

#### Consent for publication

Not applicable.
