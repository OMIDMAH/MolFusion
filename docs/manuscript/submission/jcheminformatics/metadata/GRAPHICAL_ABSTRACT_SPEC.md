# Graphical abstract — specification only

## Journal requirement

Optional. From the official *Journal of Cheminformatics* Research article
template:

> A graphical abstract can be supplied which, together with the article
> title, should provide the reader with a visual description of the type of
> chemistry covered in the article. The graphical abstract should be
> 920 x 300 pixels and a maximum of 150KB jpeg, png or svg file.

**Status: not created.** The journal permits rather than requires one, and
the brief for this phase directs that artwork be specified rather than
generated. Producing it is a small, well-defined task if the authors want
it.

## Constraints

| Property | Requirement |
| --- | --- |
| Dimensions | 920 × 300 pixels |
| Maximum size | 150 KB |
| Formats | JPEG, PNG or SVG |
| Content | visual description of the chemistry covered, read together with the title |

SVG is the natural choice here: the existing figures are already
deterministic hand-emitted SVG, so a graphical abstract in the same style
would match the article visually and stay well under the size limit.

## Proposed content

A left-to-right flow, matching the manuscript's argument rather than
decorating it:

```
  molecule
     │
     ├─ seven fixed-vector representations
     │    (fingerprints · descriptors · reduced graph · SMILES TF-IDF)
     │
     ├─────────────┬──────────────┐
     │             │              │
  linear probe   nonlinear probe
     │             │
     ▼             ▼
  no clearly     physicochemical
  separated      descriptors rank
  leader         strongest
     │             │
     └──── Track A1 → Track A2 ────┘
            (robustness)
```

## Constraints on what it may show

The graphical abstract is a submission artifact and is bound by the same
guards as the manuscript:

- It must **not** present the nonlinear result without its probe label.
  "Physicochemical descriptors win" without the probe is exactly the
  overclaim the study is written to avoid.
- It must **not** show the linear panel as "no difference". If the linear
  outcome is depicted, it is *no clearly separated leader*.
- It must **not** label Track A2 as external validation.
- If any confidence interval appears, it must be `[1.45, 2.41]` (A1) or
  `[1.32, 2.27]` (A2), and must not be drawn so that interval separation
  reads as a significance test.
- If any compute share appears, A1 is 35.4 % and A2 is 30.7 % on the
  nonlinear-only denominator; 29.8 % must not appear beside 35.4 %.
- No endpoint-specific claim for the six low-stability endpoints.

If the authors decide to produce it, these constraints should be checked
against the same guard suite that covers the manuscript, and the artwork
added to `figures/` with its source.
