# MechaDroid anonymous review artifact

This package supports audit of the USENIX Security 2027 submission. It contains
no APK, DEX, signing key, checkpoint tensor, serialized pickle, author identity,
or absolute host path. Raw CICMalDroid/AndroZoo payloads are not redistributed.

## Quick validation

Run `python3 validate_artifact.py` from this directory. The validator checks the
content manifest, three GIN and three GCN seeds, the matched six-condition
cohorts, independent node-local restoration reports, latent-to-APK alignment,
all 1,105 matched-projection arms, and all 2,924 APK gate evidence rows. It
does not deserialize untrusted model or attack-state files.

## Contents

- `results/gin` and `results/gcn`: training summaries, six-condition reports,
  and independent node-local restoration evidence for seeds 42, 43, and 44.
- `results/apk`: the 1,462-source-APK, two-operator fail-closed gate summary and
  2,924 non-payload evidence rows. Timeouts remain in the denominator.
- `results/projection/alignment.json`: 442 de-identified row-level alignment
  measurements and per-seed aggregates supporting the projection-alignment
  table. Method names and attack-state identifiers are omitted.
- `results/projection/matched`: the 1,105-arm five-condition summary plus
  de-identified JSON/CSV evidence supporting the matched-projection table.
- `manifests`: de-identified exact-identifier split assignments and hashes.
- `configs`: relative-path GIN and GCN configurations.
- `code`: relevant data, training, evaluation, restoration, gate aggregation,
  paper validation, and regression-test sources.
- `CHECKSUMS.sha256`: hashes of every exported file.

## Safety and data acquisition

Reviewers who hold the corresponding CICMalDroid/AndroZoo corpus may reconstruct
local datasets using the included scripts and the SHA-256 identifiers. This
review package deliberately omits APK mutation/materialization automation and
live third-party integration. Emulator execution must use a fresh isolated
image, reject physical devices, and disable guest radios.

## Reproduction levels

1. `python3 validate_artifact.py` validates the supplied evidence without GPUs,
   Android tooling, or malware payloads.
2. The relative configurations document the GIN/GCN training and factorial
   parameters; local corpus and checkpoint paths must be supplied by reviewers.
3. Projection and APK summaries can be audited from the supplied de-identified
   rows. Raw or modified APKs remain intentionally outside this package.

The five-minute limit is an operational per-checkpoint scoring timeout, not an
adversarial perturbation budget. Its value was frozen before the full cohort run;
timeouts remain failures in the fixed denominator and must be reported separately.
