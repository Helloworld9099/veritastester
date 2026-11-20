# M0 checkpoint inventory: BridgeCaption EN, ES and FR

Repository: [Helloworld9099/veritastester](https://github.com/Helloworld9099/veritastester). The machine-readable companion, `m0-checkpoint-inventory.json`, lists every checkpoint blob by its full Git object ID and actual content SHA-256, each pinned ref/path where it occurs, research filename role, and associated metadata evidence. This is a read-only audit, not a compatibility, license, quality, or release decision.

## Git checkpoint payloads

**102 unique checkpoint blobs at 121 distinct paths across 11 pinned trees. All 102 content hashes were verified.** Identical Git objects at multiple paths or refs were hashed once. There were 18 unique blobs with at least two distinct paths. Aggregate unique raw size: 121,155,600 bytes (about 115.5 MiB). Every payload measured 1,187,800 bytes and had 72 contiguous, shape-verified F32 LoRA matrices; none was a full neural-network base model. The filename and archive scan found no other conventional checkpoint binary or model archive, including optimizer-state files, sharded checkpoint index files, or ONNX exports in scope. Reports, model/dataset manifests, receipt checksums, registry entries and prior experiment findings are **metadata**, not payloads or proof of quality. The JSON maps 114 distinct metadata path/object combinations without including the contents of reports.

| Pinned ref | Commit | Checkpoint paths | Unique blobs | New distinct blobs vs pinned main |
|---|---|---:|---:|---:|
| main | `7ad5f9e0b198020cf11ce35aa269206bb34de87b` | 94 | 79 | 0 |
| PR #3 head | `3041690d8db624d94e2b25aea9ae0d78f8a11d2c` | 0 | 0 | 0 |
| PR #36 head | `2dc37a88f87ebc4cce4cff38caf2598a382b134d` | 89 | 75 | 0 |
| PR #39 head | `2a565eacb9a5a20ef9b90abc9e52123c053ef86e` | 94 | 79 | 0 |
| PR #40 head | `48461460acffd8f05dd3fd2e97f20df03cfc38f5` | 94 | 79 | 0 |
| PR #41 head | `8f93fbcbf9d58ea9c1fcdef60ab5baaa1ca6a693` | 110 | 94 | 15 |
| PR #42 head | `a9c53aff1456e4abe3e33965cc03a2a045b22e61` | 119 | 100 | 21 |
| PR #43 base (human-feedback branch) | `e832e0d4ff8371a81d3e064e76b1d92b40d359b8` | 119 | 100 | 21 |
| PR #43 head (shared research registry) | `ca755aec8f7111f03145e18ef9948cd6127aaec0` | 119 | 100 | 21 |
| Spanish review-directed workflow execution commit | `bd99fa684d34e5beb750199ab23e9b8bcb7ccc6a` | 119 | 100 | 21 |
| Spanish review-directed branch tip at audit | `cb9ce29e64165485169602d3311a1827fc9e55c0` | 121 | 102 | 23 |

The totals are tree contents, not PR-introduced objects. PR #36 did not carry four unique objects subsequently in pinned main. PRs #39 and #40 had exactly the same checkpoint paths and objects as pinned main; their findings or off-tree output references are not new Git payloads. PR #43 has the same 100 weight objects as its pinned base and PR #42. The registry is a code/metadata layer over separate research checkpoints, not one new merged multilingual weight. PR #41's champion-named file is an exact alias of its round-10 adapter; PR #42's `candidate_after_three.safetensors` is the exact retained round-02 adapter, rather than evidence of a third-round adapter having been promoted.

## Exact research references relevant to consolidation

| PR #43 registry direction | Active research adapter SHA-256 | Rollback SHA-256 | External base revision |
|---|---|---|---|
| ES→EN | `3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71` | `14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b` | `Helsinki-NLP/opus-mt-es-en@c96e2c5399ebfae4fc43d9669556b9afa74bb69d` |
| FR→EN | `fba2bd78958bf0bf8a4c14dc55e601e08663633718c4c9c17f91ef9b52dcfa1c` | `072f7f5d2d90b26991c5c62bf8ddb40f686f65e66be52717e7b121d507008ad9` | `Helsinki-NLP/opus-mt-fr-en@c4aed37b318c763fd177aa449b44e3b783cc6c02` |
| EN→ES | None: upstream base only | None: upstream base only | `Helsinki-NLP/opus-mt-en-es@5bc4493d463cf000c1f0b50f8d56886a392ed4ab` |

All four active/rollback Git adapter payloads match their registry SHA-256 values. The upstream base model binaries are out of repo. The JSON records their *expected* manifest/report hashes rather than passing these off as M0 content checks; the binaries were not fetched during this structural sub-audit. The separate M0 fixture replay fetched and verified all three pinned active base snapshots. Structural similarities among LoRA matrices do not make different underlying models, tokenizers or vocabularies merge-compatible.

The separately persisted Spanish review-directed run added two content-verified **candidates** under `research/es_en_v2/review_directed_round_20261120/weights/`, on the branch tip only, neither substituted into the audited registry:

- Step 32: Git blob `b62cc7bbfd2689ec158e8414bdea42d99de17aae`; SHA-256 `16091608417301f6862bf29d66a6a8e25580678b04b7e22e05135ef5dd90d5e3`.
- Step 64: Git blob `654657d96d3b55f07bbb2603010ab2ba949ebdcc`; SHA-256 `1a6ac23bc5c17839988a4d2b2ae862089f71a35ca86f9a6a42f07ef9fe3605f8`.

## Releases, LFS and scope boundaries

- The repository's public [Releases](https://api.github.com/repos/Helloworld9099/veritastester/releases?per_page=100) and [Tags](https://api.github.com/repos/Helloworld9099/veritastester/tags?per_page=100) APIs each returned an empty list. That is the public response at audit time, not a claim about private draft assets or previously deleted releases.
- The [review-directed workflow run](https://github.com/Helloworld9099/veritastester/actions/runs/44823434888) [artifact listing](https://api.github.com/repos/Helloworld9099/veritastester/actions/runs/44823434888/artifacts?per_page=100) lists one unexpired artifact, `bridgecaption-one-final-spanish-training-round` (ID `14672851841`, 2,197,630 bytes, API-reported archive SHA-256 `49a4c08a986fdec6ce56492a9a414358900fbd61dae47a1f588e27f11eac8058`). The archive was not downloaded, so the listed digest is remote metadata. This is an ephemeral CI artifact, not a GitHub Release. Persisted checkpoint bytes were verified independently from the branch tip.
- None of the pinned trees contained `.gitattributes` or `.lfsconfig`; `git lfs ls-files HEAD` was empty in the pinned PR #43 checkout. All 102 conventional checkpoint candidates were verified as actual Safetensors, not LFS pointer text. This does not attest to every historical ref or invisible remote LFS storage.
- Method: queried each complete pinned Git tree, selected conventional checkpoint and archive suffixes plus checkpoint-named metadata, then verified bytes against Git's `SHA-1("blob " + size + NUL + data)` before hashing raw content with SHA-256. The pinned PR #43 clone already provided 100 objects; two new objects were fetched by immutable Git Blob IDs. Safetensors structure was checked without loading a model or executing weights. Nothing changed on a remote branch, no weights were published or copied into the audit deliverables, and no training or model evaluation was run.
