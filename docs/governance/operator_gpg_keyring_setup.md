# Operator handoff: GPG keyring setup for the methodology sign-off guard

**Issue reference:** #226 (sub-issue of #192 — the
`v4-N3-signature-infra` backlog item).
**Code-side landing PR:** see issue #226 thread for the merged PR ref.
**Audience:** repo operator / security team owner of the methodology
sign-off Gate N3 infrastructure.
**Time estimate:** 1–2 hours, mostly pubkey collection coordination.

After completing the three steps in this doc, items 1 (H1 — GPG keyring)
and 4 (H4 — CoI body signature verification) in
`docs/governance/n3_known_limitations_20260510.md` flip from
**PARTIALLY RESOLVED** → **RESOLVED** and issue #226 (and its parent
#192 if last open child) can close.

---

## Why this doc exists

The CI-side infrastructure for cryptographic signature verification of
methodology sign-offs lives in:

* `scripts/check_methodology_signoff.py` — implements
  `check_keyring_present` + `check_coi_body_signature_verifies`, the
  `--keyring-dir` flag, the `--strict-gpg` flag, and exit code `4` for
  STRICT_GPG keyring/CoI-sig gaps.
* `.github/workflows/methodology-signoff-validator.yml` — has a
  "Provision GPG keyring" step that imports an operator-supplied
  ASCII-armor key bundle from the
  `GPG_REVIEWER_KEYS_ARMOR_BASE64` repo secret into a per-job
  `$KEYRING_DIR`, then passes `--keyring-dir` to the validator.

**Important — production CI fails closed by default.** The caller
workflow `.github/workflows/methodology_signoff_guard.yml` passes
`strict_gpg: '1'` so as soon as this PR merges, the next CI run of
the methodology sign-off guard against a touched sign-off artifact
will **exit code `4`** if the operator has NOT yet:

1. Provisioned the `GPG_REVIEWER_KEYS_ARMOR_BASE64` repo secret, OR
2. Populated the `fingerprint` column for the active reviewer rows, OR
3. Ensured the CoI declaration carries either an inline armor block or
   a sibling `<coi>.asc` detached signature.

If you need a temporary advisory-mode rollout window (e.g. while
collecting reviewer pubkeys), edit the caller line `strict_gpg: '1'`
→ `strict_gpg: '0'`. With the opt-out, signature/keyring/CoI-sig
checks pass with a WARN but the workflow exits `0`. **Revert to `'1'`
once operator setup is complete.**

The CI workflow defaults `STRICT_GPG: '1'` so once the operator action
is complete, missing or invalid sigs hard-block PRs without further
caller-side change.

This doc walks through the three operator steps end-to-end.

---

## Architecture decision: secret-encoded ASCII-armor bundle

The chosen distribution mechanism is a single **base64-encoded
ASCII-armor multi-key bundle** stored in one GitHub repo secret named
`GPG_REVIEWER_KEYS_ARMOR_BASE64`. The CI workflow base64-decodes the
secret, pipes the result through `gpg --import`, and sets `$KEYRING_DIR`
for the validator step.

**Why this strategy** (alternatives considered):

* **Pre-baked runner image** — would require a parallel image-build
  pipeline + image-tag pinning + per-runner trust state. Adds a
  significant operator surface for a small number of reviewer keys.
* **Cross-account KMS / Vault** — overkill for a repo-scoped reviewer
  set; introduces an external trust dependency (the KMS / Vault tenant
  becomes part of the supply chain).
* **Actions cache** — security-critical material in a cache with
  eviction risk + cross-PR cache-poisoning attack surface. Wrong
  primitive.

**Trade-off (acknowledged in `n3_known_limitations_20260510.md`):**
secrets-store dependency for key distribution. Mitigation: GitHub repo
secrets are encrypted at rest, access-logged, and rotation reduces to
re-encoding + `gh secret set`. The `STRICT_GPG=0` opt-out preserves
back-compat for dev workflows that haven't provisioned the secret.

---

## Step 1 — Generate / collect reviewer GPG pubkeys

For each row whose `status` is `active` in
`docs/governance/methodology_reviewer_registry.md`, obtain that
reviewer's GPG public key.

**If the reviewer already has a GPG identity:**

```bash
# Reviewer runs locally:
gpg --armor --export <their-keyid-or-email> > <handle>.asc
# Then sends <handle>.asc to operator via secure channel.
```

**If the reviewer needs to generate a key from scratch:**

```bash
# Reviewer runs locally — interactive walks them through:
gpg --full-generate-key
# Recommended: RSA 4096 bits, no expiration (or 2-year expiration with
# planned rotation), passphrase-protected.

# Then export:
gpg --armor --export <email-they-used> > <handle>.asc
```

Each `<handle>.asc` is a single-key ASCII-armor bundle starting with
`-----BEGIN PGP PUBLIC KEY BLOCK-----` and ending with
`-----END PGP PUBLIC KEY BLOCK-----`. **Verify the file is plaintext
ASCII** (not base64 or binary) before proceeding to step 2.

For the fingerprint pinning step (step 3), also extract the canonical
40-char hex fingerprint:

```bash
gpg --list-keys --with-fingerprint --keyid-format=long <reviewer-email>
# Example output:
#   pub   rsa4096/AAAAAAAAAAAAAAAA 2026-05-14 [SC]
#         Key fingerprint = ABCD EF01 2345 6789 ABCD  EF01 2345 6789 ABCD EF01
# Strip whitespace: ABCDEF0123456789ABCDEF0123456789ABCDEF01
```

---

## Step 2 — Concatenate, base64-encode, upload as secret

Once all `<handle>.asc` files are collected:

```bash
# Concatenate ASCII-armor exports into one bundle. gpg --import handles
# multi-key armor blobs natively (each PUBLIC KEY BLOCK is parsed in
# sequence).
cat alice.asc bob.asc carol.asc > reviewer-bundle.asc

# Sanity check: count the BEGIN markers (should equal the # of keys).
grep -c 'BEGIN PGP PUBLIC KEY BLOCK' reviewer-bundle.asc

# Local roundtrip test — verify the bundle imports cleanly:
TMPHOME=$(mktemp -d)
chmod 700 "$TMPHOME"
gpg --homedir "$TMPHOME" --batch --import < reviewer-bundle.asc
gpg --homedir "$TMPHOME" --list-keys --with-fingerprint
rm -rf "$TMPHOME"
```

If the roundtrip test prints all reviewer fingerprints, base64-encode
and upload as a single repo secret:

```bash
# `-w0` is REQUIRED — multi-line base64 will not survive transit
# through `gh secret set`'s shell-arg path on some shells.
base64 -w0 reviewer-bundle.asc > reviewer-bundle.b64

# Upload to the repo. (Use `gh repo set-default` first to confirm the
# right repo is targeted.)
gh secret set GPG_REVIEWER_KEYS_ARMOR_BASE64 < reviewer-bundle.b64

# Verify the secret was set (gh redacts the value):
gh secret list | grep GPG_REVIEWER_KEYS_ARMOR_BASE64
```

After this step, the next CI run of the methodology sign-off guard
will see the secret, decode it, import all keys into `$KEYRING_DIR`,
and pass `--keyring-dir "$KEYRING_DIR"` to the validator.

**Cleanup local artifacts:**

```bash
shred -u reviewer-bundle.asc reviewer-bundle.b64 *.asc
```

(The pubkeys are not secret per se — they're public-key material — but
the practice of `shred`-ing keeps the operator workflow uniform with
private-key handling.)

---

## Step 3 — Populate the `fingerprint` column

Edit `docs/governance/methodology_reviewer_registry.md`. For each row
whose `status` is `active`, replace the
`<TBD — populated by operator>` placeholder in the `fingerprint`
column with the 40-char hex fingerprint extracted in step 1:

```diff
-| E. Nunez | etn3724@gmail.com | enunezvn | engineering_owner | 2026-05-11 | methodology, ... | active | `<TBD — populated by operator>` |
+| E. Nunez | etn3724@gmail.com | enunezvn | engineering_owner | 2026-05-11 | methodology, ... | active | ABCDEF0123456789ABCDEF0123456789ABCDEF01 |
```

The validator's `_normalize_fingerprint` accepts these formats
interchangeably:

* `ABCDEF0123456789ABCDEF0123456789ABCDEF01`
* `abcdef0123456789abcdef0123456789abcdef01` (lowercased)
* `ABCD EF01 2345 6789 ABCD  EF01 2345 6789 ABCD EF01` (gpg's spaced format)
* `0xABCDEF0123456789ABCDEF0123456789ABCDEF01` (with `0x` prefix)
* Backticked variants (e.g. `` `ABCDEF...` ``)

Open the registry-update PR and let the methodology sign-off guard
self-validate (the workflow runs on changes to
`docs/governance/methodology_reviewer_registry.md`).

---

## Verification — confirm the end-to-end loop works

After the secret is set AND the registry is populated:

1. Open a test PR that adds or modifies a file matching
   `docs/results/optum_methodology_signoff_*.md`. The PR's sign-off
   doc must carry a real ASCII-armor signature (use `gpg --detach-sign
   --armor` against the doc body up to but not including the
   `## Cryptographic signature` heading).
2. The methodology sign-off guard workflow will run with
   `STRICT_GPG=1` (the production default). The
   "Provision GPG keyring" step should log:
   * `Imported keys` group with the reviewer fingerprints listed.
3. The "Run validator" step should print
   `[PASS] signature_verifies: gpg --verify OK: ...` and exit `0`.
4. If the sign-off references a CoI document, the
   `coi_body_signature_verifies` check should similarly print
   `[PASS] ... gpg --verify OK on CoI body` (assuming the CoI carries
   either an inline armor block or a sibling `.asc`; if it does NOT,
   the check will print `[PASS] WARN: no CoI body signature found ...`
   and the validator exits `4` under STRICT_GPG=1 — which is the
   correct fail-closed behavior).

If step 3 reports `gpg --verify FAILED: no public key`, the registered
fingerprint and the imported pubkey don't match — re-export and re-
upload the secret.

---

## STRICT_GPG opt-out (for dev / early-stage runs)

The caller workflow `.github/workflows/methodology_signoff_guard.yml`
passes `strict_gpg: '1'` to the reusable workflow. To opt out of fail-
closed mode (e.g. while pubkeys are still being collected), edit that
caller line to `strict_gpg: '0'`. The validator will then run signature
checks in advisory mode — failures print as WARN but the workflow
exits `0`.

This is a deliberate design choice: STRICT_GH (issue #192 H2/M1) and
STRICT_GPG (issue #226 H1+H4) compose orthogonally so an operator can
roll out the gh-CLI provisioning and the keyring infra independently.

---

## Rotation

To rotate keys (reviewer leaves, key compromised, etc.):

1. Remove the rotated reviewer's pubkey from the registry (set their
   row's `status` to `inactive` AND replace `fingerprint` with `<TBD>`).
2. Rebuild `reviewer-bundle.asc` from the remaining active rows
   (step 2 above).
3. Re-encode and re-upload (`base64 -w0 ... | gh secret set ...`).
4. The next CI run uses the new bundle.

For an emergency revocation, also push a revocation certificate to a
keyserver per the reviewer's local `gpg --gen-revoke` workflow — but
this is out of scope for the repo-side infrastructure.

---

## See also

* `docs/governance/n3_known_limitations_20260510.md` items 1 + 4 (the
  formal risk-register entries this doc retires).
* `scripts/check_methodology_signoff.py` — `check_keyring_present`,
  `check_coi_body_signature_verifies`, `_resolve_strict_gpg`, exit code
  `4`.
* `.github/workflows/methodology-signoff-validator.yml` — the
  "Provision GPG keyring" step.
* Issue #226 — operator scope tracker.
* Issue #192 — parent (umbrella) issue.
