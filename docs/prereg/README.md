# Pre-registration — where each version lives

| Version | Content | Location |
|---|---|---|
| v1.1 | Protocol | preparation repo `tarudesu/viLens-concepts-preparation` (pinned here read-only at `data/external/viLens-concepts-preparation/`) |
| v1.2 | `PREREG.md` (tag `prereg-v1.2`) and amendments | preparation repo, `PREREG.md`, `PROTOCOL_AMENDMENTS.md` |
| v1.3 | Cloze rebuild (data amendment) | preparation repo, `PROTOCOL_AMENDMENTS.md`, tag `data-v1.3.1` |
| **v1.4 (governing)** | Analysis amendment, effective analysis specification, pre-written abstracts | **this directory, at git tag `prereg-v1.4` of this repository** |

- `PREREG.md` here reproduces the frozen v1.1 text and the v1.2 amendments unchanged, adds v1.3 by reference, adds the v1.4 amendment in full, and ends with the **effective analysis specification**. That section governs wherever the texts differ.
- `v1.4_amendment.md` is the amendment text on its own.
- `abstracts_prewritten.md` holds the three abstracts (H1, H2, inconclusive/mixed) and the rule for choosing one at Checkpoint H.
- These files are frozen at tag `prereg-v1.4`. Any later change is a new amendment with a new tag, recorded in `docs/decisions.md`.
- No study-model run preceded the tag. The only earlier model outputs are the three exploratory notebooks described in `docs/experiments.md` §2.9.
