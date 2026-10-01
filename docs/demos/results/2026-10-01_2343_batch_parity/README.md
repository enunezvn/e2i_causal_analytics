# #2343 batch vs single predict parity, post-#2339 (2026-10-01)

Part of #2343.

## Verdict: PASS

On current prod, `POST /api/models/predict/{m}/batch` gives the same scores as N single
`POST /api/models/predict/{m}` calls for the same rows. This holds for all three models
tested: two HCP-grain models and one patient-grain model.

- The batch is routed to the named model, not to the sidecar default.
- The single-predict probabilities did not move across the #2339 sidecar rebuild
  (sklearn 1.9.1 to 1.6.1).

## Deployed state

| Item | Value |
|---|---|
| run_at | 2026-10-01T00:44:00Z |
| git HEAD (main) | 8845cd65d |
| api `image_sha` | 8845cd65d9041f936ccb8b594389c824d068e229 |
| sidecar image id | sha256:b32c2ff93c4daa5d7a754e59f663cef571af09828195a12360e100b9320b3533 (started 2026-10-01T00:40:18Z) |
| sidecar sklearn | 1.6.1 |

Each model's `bundle_sha256`, from `POST :3000/model_info {"input_data":{"model_name":...}}`:

| Model | bundle_sha256 |
|---|---|
| hcp_adoption_kisqali_goldstd_lr_v1 | 2878366567378bc6ee000001d202f0878de7aea1bfecba0ffaa5b0ab501ce786 |
| hcp_adoption_fabhalta_goldstd_lr_v1 | eb1fa337f35cd8eceaf450bbef814c135fa60c225acd56ad539ef5925b00fa07 |
| initiation_kisqali_goldstd_lr_v1 | 5e7a18962db5e4fd5c06a72414ceb206bd31ab22f485e3407502ab2e7eaf5125 |
| initiation_fabhalta_goldstd_lr_v1 (teeth comparator) | faed35c3fa450b22082f81ebe09e5a3f8ffaffd070433e53af7f469243e57426 |

## Method

The script is `live_batch_single_parity_2343.py`. It extends the pre-#2339 script in two
ways: it adds a patient-grain case and it compares each probability with the pre-#2339
value.

For each model:

1. Read 8 real `test`-split rows, read-only, ordered by id. The script asserts that the
   columns match the model's `/model_info` `keep_columns`.
   - HCP-grain rows come from `public.hcp_adoption_goldstd_v` for the brand.
   - Patient-grain rows come from `public.patient_journeys` with
     `brand=… and is_synthetic and data_split='test'`. This is the FeatureBuilder patient
     path.
2. Make 8 single predicts and 1 batch predict through the API, using a Supabase password
   grant as the admin user.
3. **Teeth (a):** send the same rows through a different goldstd model's batch route.
4. **Teeth (b):** send the same rows to the sidecar's unrouted `/predict_batch` with no
   `model_name`. This is what the pre-#2346 route effectively did.

A model passes when all of these hold:

- max|single−batch| ≤ 1e-9;
- the predicted classes are equal;
- every single-predict `model_version` equals the requested model;
- teeth (a) max diff > 1e-3.

The predict routes allow 10 requests per 60 s, so the script sleeps 65 s before each model.
Each model uses exactly 10 requests.

## Numbers

| Model | n | max\|single−batch\| | classes equal | teeth (a) max\|this−other\| | teeth (b) default n_pred |
|---|---|---|---|---|---|
| hcp_adoption_kisqali_goldstd_lr_v1 | 8 | 0.0 | yes | 0.2632 (vs fabhalta) | 0 |
| hcp_adoption_fabhalta_goldstd_lr_v1 | 8 | 0.0 | yes | 0.2836 (vs kisqali) | 0 |
| initiation_kisqali_goldstd_lr_v1 | 8 | 1.1e-16 (float rounding) | yes | 0.0435 (vs initiation_fabhalta) | 0 |

The teeth margins are at least 4e7 times the parity tolerance, so a batch that scored with
the wrong model would fail. The unrouted default produces no predictions for these rows.

## Pre-#2339 vs post-#2339

The 16 HCP rows from the pre-#2339 run (`baseline_pre2339_live_batch_single_parity_2343.out`,
2026-09-30T22:05Z, git 9f66d1015) were re-scored by single predict:

- max|post−pre| = **4.7e-9**.

The baseline printed 8 decimals, so any difference below about 5e-9 cannot be seen. The
result is therefore consistent with zero change.

- **What this shows:** models that are now unpickled on their native sklearn 1.6.1 score
  the same as they did when unpickled under 1.9.1.
- **What it does not show:** that unpickling under 1.9.1 was harmful. For these logistic
  regression bundles it did not change the scores.
- The direct unpickle-health evidence is the dispatcher's post-deploy check. Since the
  sidecar restarted it found 0 InconsistentVersionWarning and 0 duplicate-bundle errors.
- No pre-#2339 baseline exists for the initiation model.

## Side effects

The run made no writes to the DB, Redis or registry. Its only effects were routine request
telemetry:

- a Supabase auth token mint;
- rate-limit counters;
- user-activity logging for 24 single predicts and 6 batch predicts;
- 3 direct sidecar `/predict_batch` calls and 7 `/model_info` calls.

## Known open defect: #2351

Batch prediction rows report `model_version: null`, shown as `['None']` in the output. The
batch is routed correctly; the scores prove that. The metadata field is simply not filled
in. This is recorded here and not fixed in this PR.

## Files

- `live_batch_single_parity_2343.py`: post-#2339 script (this run).
- `live_batch_single_parity_2343_post2339.out`: output of this run.
- `baseline_pre2339_live_batch_single_parity_2343.py` and `.out`: the pre-#2339 script and
  its passing run.
- `baseline_pre2339_run1_rate_limited_partial.out`: the first pre-#2339 attempt.
  - It hit a 429 on its second model.
  - Its kisqali "FAIL" came from a check bug in that script version: it mixed the batch's
    null `model_version` into the single-predict version set. It was not a parity failure.
    The next run fixed the check.
