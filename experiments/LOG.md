# Experiment log

Every Kaggle submission of this project, oldest first. Scores are `private / public` (`adjusted_edge_jaccard + 0.1 * division_jaccard`). Notebooks 1-6 of the table are kept in full under `notebooks/`; the rest are documented here as the hypothesis they tested plus the exact change against their parent.

## E01 - Harmonic Fusion baseline

- Kaggle notebook: `biohub-v1`
- Parent: -
- Score (private / public): **0.91448 / 0.94762**
- Full notebook: `notebooks/01_harmonic_fusion_baseline_v1.ipynb`

> public 0.939 base + holdout-selected post-process configuration


## E02 - + motion relink preserves raw links

- Kaggle notebook: `biohub-v2`
- Parent: E01
- Score (private / public): **0.91274 / 0.94689**

> public 0.947 base + ILP-preserving residual motion relink

Change vs E01:

```text
BIOHUB_MOTION_RELINK_PRESERVE_RAW: - -> 1
```


## E03 - Hengck end-to-end linker only

- Kaggle notebook: `notebook78ece55754`
- Parent: E02
- Score (private / public): **0.90700 / 0.92437**

> public 0.947 detector baseline + single public end2end linker

Change vs E02:

```text
BIOHUB_END2END_ARTIFACT_SLUG: - -> hengck23-cell-point-detector-demo
BIOHUB_END2END_CHECKPOINT_NAME: - -> 00000008.pth
BIOHUB_LINKER_IMPLEMENTATION: - -> hengck23_end2end_only
BIOHUB_MOTION_RELINK_PRESERVE_RAW: 1 -> -
BIOHUB_OUTPUT_GAP_CLOSE: - -> 1
BIOHUB_OUTPUT_MOTION_RELINK: - -> 1
BIOHUB_OUTPUT_PRUNE_ISOLATED: - -> 1
BIOHUB_PPSWEEP_MAX_ADJ_LOSS: 0.0005 -> -
BIOHUB_PPSWEEP_SELECT_MARGIN: 0.001 -> -
BIOHUB_USE_ILP: - -> 0
BIOHUB_VALIDATOR_ENABLE: - -> 0
```


## E04 - post-processing stages switched off

- Kaggle notebook: `biohub-version4`
- Parent: E03
- Score (private / public): **0.90354 / 0.93177**

> public 0.947 detector baseline + single public end2end linker

Change vs E03:

```text
BIOHUB_OUTPUT_GAP2_RECOVERY: 1 -> 0
BIOHUB_OUTPUT_GAP_CLOSE: 1 -> 0
BIOHUB_OUTPUT_MOTION_RELINK: 1 -> 0
```


## E05 - baseline + Hengck confidence gate

- Kaggle notebook: `biohub-v5`
- Parent: E04
- Score (private / public): **0.90687 / 0.94611**

> public 0.947 protected base + conservative hengck residual gate

Change vs E04:

```text
BIOHUB_GATE_ADD_MIN_MARGIN: - -> 0.35
BIOHUB_GATE_ADD_MIN_PROB: - -> 0.92
BIOHUB_GATE_MAX_CHANGES_ABS: - -> 400
BIOHUB_GATE_MAX_CHANGES_FRAC: - -> 0.005
BIOHUB_GATE_MAX_DISTANCE_UM: - -> 8.5
BIOHUB_GATE_REPLACE_MIN_MARGIN: - -> 0.55
BIOHUB_GATE_REPLACE_MIN_PROB: - -> 0.97
BIOHUB_GATE_REPLACE_OLD_MAX_PROB: - -> 0.55
BIOHUB_LINKER_IMPLEMENTATION: hengck23_end2end_only -> baseline947_hengck_confidence_gate
BIOHUB_USE_ILP: 0 -> 1
```


## E06 - post-process, then Hengck gate

- Kaggle notebook: `biohub-v6`
- Parent: E05
- Score (private / public): **0.91342 / 0.94677**
- Full notebook: `notebooks/02_harmonic_fusion_hengck_gate_v6.ipynb`

> public 0.947 complete postprocess base + conservative hengck residual gate

Change vs E05:

```text
BIOHUB_LINKER_IMPLEMENTATION: baseline947_hengck_confidence_gate -> baseline947_postprocess_then_hengck_gate
BIOHUB_OUTPUT_GAP2_RECOVERY: 0 -> 1
BIOHUB_OUTPUT_GAP_CLOSE: 0 -> 1
BIOHUB_OUTPUT_MOTION_RELINK: 0 -> 1
```


## E07 - K3 + Hengck fork-transfer base

- Kaggle notebook: `biohub-k3-hengck-951`
- Parent: -
- Score (private / public): **0.91740 / 0.95133**
- Full notebook: `notebooks/03_k3_hengck_base_951.ipynb`

> # K3 + Hengck: timeout-only corrective candidate
> Unchanged prediction policy. Internal runtime budget follows the 12-hour competition limit.
> Ordinary private GPU commits remain capped at 7200 seconds through the Kaggle push timeout.
> V1 hidden failure cause is unconfirmed; this candidate only addresses reproducible premature deadlines.


## E08 - V1327 W3 as primary checkpoint

- Kaggle notebook: `biohub-k3-hengck-v1327-w3-test`
- Parent: E07
- Score (private / public): **0.92003 / 0.95118**

> # K3 + Hengck + V1327 W3 primary A/B test
>
> Control mechanics are preserved from the confirmed 0.951 K3 + Hengck fork-transfer notebook. Only the primary K3 checkpoint is replaced by the public V1327 W3 candidate.

Change vs E07:

```diff
-BIOHUB_PRESET = 'harmonic_v3_division_wide'
-BIOHUB_SCORE_AXIS = 'unscored paired diagnostic; do not inherit 0.947 from the source card'
+BIOHUB_PRESET = 'harmonic_v3_division_wide_v1327_w3'
+BIOHUB_SCORE_AXIS = 'K3+Hengck 0.951 control with V1327 W3 primary-model A/B test'
-EXPERIMENT_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"
+EXPERIMENT_TAG = "v1327_w3_primary_k3_hengck_fork_transfer"
+# A/B arm: replace only the primary K3 checkpoint with the public V1327 W3
+# model. The K3 detector/linker code, secondary seed, DeepCenter, all
+# post-processing, and the Hengck fork-transfer stage remain unchanged.
+import hashlib as _v1327_hashlib
+
+_V1327_EXPECTED_SHA256 = "c0562f356250032eee7982f29cec7655e4d4c26acf683cdf25da794044bac00b"
+_V1327_CONFIG_SHA256 = "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a"
+_V1327_ROOT_CANDIDATES = [
+    Path("/kaggle/input/datasets/josephadamski91/biohub-v1327-w3-real-model"),
+    Path("/kaggle/input/biohub-v1327-w3-real-model"),
+]
+_V1327_ROOT = next((p for p in _V1327_ROOT_CANDIDATES if p.is_dir()), None)
+if _V1327_ROOT is None:
+    raise FileNotFoundError({
+        "missing_v1327_input": [str(p) for p in _V1327_ROOT_CANDIDATES]
+    })
+_V1327_MODEL = _V1327_ROOT / "model.pth"
+_V1327_CONFIG = _V1327_ROOT / "config.json"
+_V1327_RECEIPT = _V1327_ROOT / "asset_receipt.json"
+for _v1327_required in (_V1327_MODEL, _V1327_CONFIG, _V1327_RECEIPT):
+    if not _v1327_required.is_file():
+        raise FileNotFoundError(f"Missing V1327 asset: {_v1327_required}")
+
+def _v1327_sha256(path: Path) -> str:
... (62 more lines)
```


## E09 - V1327 endpoint consensus

- Kaggle notebook: `biohub-k3-v1327-hengck-consensus`
- Parent: E07
- Score (private / public): **failed (notebook threw an exception)**

> # K3 + V1327 conservative endpoint consensus + Hengck
>
> The confirmed K3 + Hengck 0.951 graph is reproduced unchanged. V1327 may only append a small number of one-frame links between an existing free track end and a free track start when strict same-frame mutual mapping and two-sided motion checks pass. No nodes or existing edges are replaced. Internet must remain disabled.

Change vs E07:

```diff
-
+[[1, "# Only change this selector when intentionally choosing the output arm.\nHF_SUBMISSION_ARM = \"control\"  # \"control\" or \"fork_preserve\"\nH ...
-"""Sequential GPU subprocesses; only final submission and aggregate receipts escape tmp."""
+"""Sequential frozen inference and append-only consensus transfer."""
+from src.k3_v1327_endpoint_consensus import transfer_endpoint_consensus
-            # Also reap any grandchildren after a failed inference or timeout.
-    hg, k3 = runtime/'hengck', runtime/'k3'
-    hg.mkdir(); k3.mkdir()
+    hg, k3, v1327 = runtime/'hengck', runtime/'k3', runtime/'v1327'
+    hg.mkdir(); k3.mkdir(); v1327.mkdir()
+
+
-    print('STAGE Hengck complete; GPU process exited; K3 starting', flush=True)
+
+    print('STAGE frozen K3 control starting', flush=True)
-    print('STAGE K3 complete; integer CSV transfer starting', flush=True)
-    transfer_start = time.monotonic()
-    receipt = transfer_csv(k3/'submission.csv', runtime/'final.csv', hg, expected)
-    assert receipt['csv']['baseline_sha256'] == kreport['csv_sha256']
+
+    print('STAGE frozen V1327 starting', flush=True)
+    run_child([sys.executable, 'v1327_child.py', str(v1327)], runtime, remaining(), runtime/'v1327.log')
+    vreport = json.loads((v1327/'experiment_diagnostics.json').read_text())
+    assert sorted(vreport['combined_test_stems']) == expected
+
+    print('STAGE unchanged Hengck fork transfer starting', flush=True)
+    hengck_start = time.monotonic()
+    transfer = transfer_csv(k3/'submission.csv', runtime/'k3_hengck.csv', hg, expected)
+    assert transfer['csv']['baseline_sha256'] == kreport['csv_sha256']
+
... (372 more lines)
```


## E10 - V1327 fragment recovery

- Kaggle notebook: `biohub-k3-v1327-hengck-fragments`
- Parent: E07
- Score (private / public): **failed (notebook threw an exception)**

> # K3 + V1327 + Hengck confirmed fragment recovery
>
> The confirmed K3 + Hengck graph remains the immutable prefix. V1327 may add only complete two- or three-frame missing-cell fragments between free K3 endpoints when Hengck supports the majority of their edges and strict physical-motion checks pass. The earlier conservative endpoint consensus runs afterward. Internet must remain disabled.

Change vs E07:

```diff
-
+[[1, "# Only change this selector when intentionally choosing the output arm.\nHF_SUBMISSION_ARM = \"control\"  # \"control\" or \"fork_preserve\"\nH ...
-"""Sequential GPU subprocesses; only final submission and aggregate receipts escape tmp."""
+"""Sequential K3, V1327 and Hengck inference with append-only recovery."""
-import re
+from src.k3_v1327_fragment_recovery import transfer_fragment_recovery
+from src.k3_v1327_endpoint_consensus import transfer_endpoint_consensus
-        process = subprocess.Popen(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
-                                   start_new_session=True)
+        process = subprocess.Popen(command, cwd=directory, stdout=stream,
+                                   stderr=subprocess.STDOUT, start_new_session=True)
-                errors = [line[:400] for line in lines if re.match(r'^(?:[A-Za-z.]+Error|AssertionError|TimeoutError):', line)]
-                raise RuntimeError('Child process failed: '+Path(command[1]).name+' exit '+str(code)+'; '+'; '.join(errors[-3:]))
+                child_name = Path(command[1]).stem
+                failure_log = Path('/kaggle/working') / (child_name + '_failure.log')
+                shutil.copyfile(log, failure_log)
+                raise RuntimeError(
+                    'Child process failed: ' + Path(command[1]).name
+                    + ' exit ' + str(code) + '; full log: ' + str(failure_log)
+                    + '\n--- child log tail ---\n' + '\n'.join(lines[-120:])
+                )
-            # Also reap any grandchildren after a failed inference or timeout.
-    hg, k3 = runtime/'hengck', runtime/'k3'
-    hg.mkdir(); k3.mkdir()
+    hg, k3, v1327 = runtime/'hengck', runtime/'k3', runtime/'v1327'
+    hg.mkdir(); k3.mkdir(); v1327.mkdir()
+
+
-    run_child([sys.executable, 'hengck_child.py', str(hg), str(test)], runtime, remaining(), runtime/'hengck.log')
+    run_child([sys.executable, 'hengck_child.py', str(hg), str(test)],
... (883 more lines)
```


## E11 - broader 3-frame gap recovery

- Kaggle notebook: `biohub-k3-hengck-gap2-recall`
- Parent: E07
- Score (private / public): **0.91740 / 0.95128**

> # Biohub K3 Hengck Gap2 Recall
>
> Isolated experiment derived from the verified K3+Hengck 0.951 baseline.
>
> Hypothesis: Broaden only the strict three-frame recovery budget. This tests whether a small number of short detector dropouts limits the edge score.
>
> The detector, model artifacts and base linker are unchanged. This notebook does not submit to the competition automatically.

Change vs E07:

```diff
+# EXP-GAP2-01: conservative expansion of the existing three-frame repair.
+os.environ["BIOHUB_GAP2_MAX_TOTAL_UM"] = "10.8"
+os.environ["BIOHUB_GAP2_MAX_STEP_UM"] = "4.6"
+os.environ["BIOHUB_GAP2_MAX_LINKS_FRAC"] = "0.006"
+os.environ["BIOHUB_GAP2_MAX_LINKS_ABS"] = "220"
+
```


## E12 - motion-aware relink

- Kaggle notebook: `biohub-k3-hengck-motion-relink`
- Parent: E07
- Score (private / public): **0.92124 / 0.95228**
- Full notebook: `notebooks/04_k3_hengck_motion_relink_952.ipynb`

> # Biohub K3 Hengck Motion Relink
>
> Isolated experiment derived from the verified K3+Hengck 0.951 baseline.
>
> Hypothesis: Enable the existing motion-aware one-to-one reassignment. It may repair ambiguous edges where apparent movement is smoother than the raw linker choice.
>
> The detector, model artifacts and base linker are unchanged. This notebook does not submit to the competition automatically.

Change vs E07:

```diff
-os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "0"
+os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1"
-    "BIOHUB_OUTPUT_MOTION_RELINK": "0",
+    "BIOHUB_OUTPUT_MOTION_RELINK": "1",
-    if not expected or ns['OUTPUT_MOTION_RELINK'] is not False:
+    if not expected:
-assert OUTPUT_MOTION_RELINK is False and ILP_DIVISION_WEIGHT == 1.2 and HF_PRESERVE_FORKS is False
+assert OUTPUT_MOTION_RELINK is True and ILP_DIVISION_WEIGHT == 1.2 and HF_PRESERVE_FORKS is False
-        assert _s[_k] == 0, (_s["dataset"], _k, _s[_k])
+        assert _s[_k] >= 0, (_s["dataset"], _k, _s[_k])
+    # EXP-MOTION-01 intentionally permits motion replacements; the values are
+    # retained in diagnostics for a direct comparison with the frozen control.
-                errors = [line[:400] for line in lines if re.match(r'^(?:[A-Za-z.]+Error|AssertionError|TimeoutError):', line)]
-                raise RuntimeError('Child process failed: '+Path(command[1]).name+' exit '+str(code)+'; '+'; '.join(errors[-3:]))
+                tail = lines[-40:]
+                raise RuntimeError('Child process failed: '+Path(command[1]).name+' exit '+str(code)+'; log tail: '+'\n'.join(tail))
```


## E13 - motion relink + division recall

- Kaggle notebook: `biohub-k3-hengck-motion-division`
- Parent: E12
- Score (private / public): **0.92250 / 0.94902**

> # Biohub K3 Hengck Motion Division
>
> Isolated experiment derived from the verified K3+Hengck 0.951 baseline.
>
> Hypothesis: Combine the confirmed motion relink gain with the conservative division-recall envelope, while retaining mutual-NN, divergence and DeepCenter validation.
>
> The detector, model artifacts and base linker are unchanged. This notebook does not submit to the competition automatically.

Change vs E12:

```diff
+# EXP-MOTION-DIV-01: safe-division expansion still requires DeepCenter, mutual NN and divergence.
+os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.5"
+os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "15.0"
+os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.75"
+os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0095"
+os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.0045"
+
-assert SAFE_DIV_SISTER_SYMMETRY_TAU == 0.6
+assert SAFE_DIV_SISTER_SYMMETRY_TAU == 0.75
-    # EXP-MOTION-01 intentionally permits motion replacements; the values are
-    # retained in diagnostics for a direct comparison with the frozen control.
```


## E14 - gap-2 recovery off

- Kaggle notebook: `biohub-k3-hengck-motion-gap2-off`
- Parent: E12
- Score (private / public): **0.91983 / 0.95139**

> # Biohub K3 Hengck Motion Gap2 Off
>
> Isolated experiment derived from the verified K3+Hengck 0.951 baseline.
>
> Hypothesis: Ablate the base three-frame synthetic recovery after motion relinking. The previous expanded Gap2 variant declined, so this tests whether the base pass also contributes false tracks in the motion-corrected graph.
>
> The detector, model artifacts and base linker are unchanged. This notebook does not submit to the competition automatically.

Change vs E12:

```diff
-os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
+os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "0"
-    # EXP-MOTION-01 intentionally permits motion replacements; the values are
-    # retained in diagnostics for a direct comparison with the frozen control.
```


## E15 - long-edge gate (>8.4 um)

- Kaggle notebook: `biohub-k3-hengck-motion-long-edge-gate`
- Parent: E12
- Score (private / public): **0.92219 / 0.95286**

> # Biohub K3 Hengck Motion: Conservative Long-Edge Gate
>
> Single-change ablation from the 0.952 Motion Relink reference. Only weak relaxed-pass motion assignments beyond 8.4 µm may be removed; detector, ILP, Hengck, DeepCenter, divisions and all other PP remain frozen.

Change vs E12:

```diff
+# EXP-LONG-EDGE-01: conservative audit of only weak relaxed motion matches.
+os.environ["BIOHUB_CONSERVATIVE_LONG_EDGE_GATE"] = "1"
+os.environ["BIOHUB_CONSERVATIVE_LONG_EDGE_UM"] = "8.4"
+os.environ["BIOHUB_CONSERVATIVE_LONG_EDGE_MAX_PROB"] = "0.50"
+os.environ["BIOHUB_CONSERVATIVE_LONG_EDGE_MAX_FRAC"] = "0.002"
+CONSERVATIVE_LONG_EDGE_GATE = os.environ.get("BIOHUB_CONSERVATIVE_LONG_EDGE_GATE", "0") != "0"
+CONSERVATIVE_LONG_EDGE_UM = float(os.environ.get("BIOHUB_CONSERVATIVE_LONG_EDGE_UM", "8.4"))
+CONSERVATIVE_LONG_EDGE_MAX_PROB = float(os.environ.get("BIOHUB_CONSERVATIVE_LONG_EDGE_MAX_PROB", "0.50"))
+CONSERVATIVE_LONG_EDGE_MAX_FRAC = float(os.environ.get("BIOHUB_CONSERVATIVE_LONG_EDGE_MAX_FRAC", "0.002"))
+if not 0.0 < CONSERVATIVE_LONG_EDGE_UM <= MOTION_RELINK_RELAXED_UM:
+    raise ValueError("BIOHUB_CONSERVATIVE_LONG_EDGE_UM must be in (0, relaxed motion gate]")
+if not 0.0 <= CONSERVATIVE_LONG_EDGE_MAX_PROB <= 1.0:
+    raise ValueError("BIOHUB_CONSERVATIVE_LONG_EDGE_MAX_PROB must be in [0, 1]")
+if not 0.0 < CONSERVATIVE_LONG_EDGE_MAX_FRAC <= 0.01:
+    raise ValueError("BIOHUB_CONSERVATIVE_LONG_EDGE_MAX_FRAC must be in (0, 0.01]")
-    stats = {'raw_edges': len(raw_edges), 'dropped_nonconsecutive_edges': 0, 'dropped_long_edges': 0, 'dropped_multi_parent_edges': 0, 'dropped_multi ...
+    stats = {'raw_edges': len(raw_edges), 'dropped_nonconsecutive_edges': 0, 'dropped_long_edges': 0, 'dropped_multi_parent_edges': 0, 'dropped_multi ...
+
+    # Only weak assignments from the *relaxed* motion pass can be removed. This
+    # cannot alter tight motion matches, raw Hengck links, or any later division.
+    if CONSERVATIVE_LONG_EDGE_GATE and edges:
+        candidates = []
+        for edge in edges:
+            if not edge.get("motion_relinked") or edge.get("motion_pass") != "relaxed":
+                stats['long_edge_gate_protected'] += 1
+                continue
+            probability = float(edge.get("edge_prob", 0.0) or 0.0)
+            distance = float(edge.get("distance_um", 0.0) or 0.0)
+            if distance > CONSERVATIVE_LONG_EDGE_UM and probability <= CONSERVATIVE_LONG_EDGE_MAX_PROB:
+                candidates.append(edge)
... (22 more lines)
```


## E16 - HOCT endpoint repair

- Kaggle notebook: `biohub-k3-hengck-motion-hoct-endpoint-repair`
- Parent: E12
- Score (private / public): **0.92124 / 0.95228**

> # Biohub K3 Hengck Motion: HOCT Endpoint Repair
>
> HOCT is advisory only: it may add a tiny number of physically and motion-consistent edges between free internal endpoints. It cannot remove or replace any K3, Hengck, Motion, DeepCenter or division edge.

Change vs E12:

```diff
+"""Bounded HOCT agreement repair for an already valid K3+Hengck CSV."""
+from __future__ import annotations
+
+import csv
+from collections import Counter, defaultdict
+import glob
+import json
+import math
+from pathlib import Path
+import subprocess
+import sys
+import time
+
+import numpy as np
+
+SCALE_ZYX = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
+RADIUS_UM = 3.0
+MAX_DISTANCE_UM = 8.4
+MAX_MOTION_RESIDUAL_UM = 3.0
+MAX_ADDED_FRAC = 0.00010
+MAX_ADDED_ABS = 12
+
+
+def _find(pattern: str) -> str:
+    # Kaggle may mount an input either directly below /kaggle/input or below
+    # /kaggle/input/datasets/<owner>/<slug>.  Search every mounted depth.
+    hits = sorted(set(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))
+    if not hits:
+        raise FileNotFoundError(f"HOCT endpoint repair: missing input matching {pattern!r}")
+    return hits[0]
... (291 more lines)
```


## E17 - 1-frame gap close off

- Kaggle notebook: `biohub-k3-hengck-motion-gap-close-off`
- Parent: E12
- Score (private / public): **0.91607 / 0.95002**

> # Biohub K3 Hengck Motion Gap Close Off
>
> Single-variable ablation from the confirmed K3 + Hengck + Motion Relink + Gap2 baseline.
>
> Hypothesis: Ablate only one-frame synthetic gap closing; retain Motion Relink and Gap2.

Change vs E12:

```diff
+os.environ["BIOHUB_OUTPUT_GAP_CLOSE"] = "0"
+
+# Single-variable ablation guard.
+assert OUTPUT_GAP_CLOSE is False
+
```


## E18 - keep tracks of length 5

- Kaggle notebook: `biohub-k3-hengck-motion-short-track-len-5`
- Parent: E12
- Score (private / public): **0.92118 / 0.95274**

> # Biohub K3 Hengck Motion Short Track Len 5
>
> Single-variable ablation from the confirmed K3 + Hengck + Motion Relink + Gap2 baseline.
>
> Hypothesis: Retain components of length five while keeping all other post-processing unchanged.

Change vs E12:

```diff
-os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
+os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "5"
-    "BIOHUB_OUTPUT_MIN_TRACK_LEN": 6.0,
+    "BIOHUB_OUTPUT_MIN_TRACK_LEN": 5.0,
+# Single-variable ablation guard.
+assert OUTPUT_MIN_TRACK_LEN == 5
+
```


## E19 - relaxed relink radius 10 -> 9 um

- Kaggle notebook: `biohub-k3-hengck-motion-relaxed-9um`
- Parent: E12
- Score (private / public): **0.92228 / 0.95219**

> # Biohub K3 Hengck Motion Relaxed 9um
>
> Single-variable ablation from the confirmed K3 + Hengck + Motion Relink + Gap2 baseline.
>
> Hypothesis: Tighten only the relaxed Motion Relink radius from 10 to 9 microns.

Change vs E12:

```diff
+os.environ["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "9.0"
+
+# Single-variable ablation guard.
+assert abs(MOTION_RELINK_RELAXED_UM - 9.0) < 1e-12
+
```


## E20 - detector threshold 0.960

- Kaggle notebook: `biohub-k3-hengck-motion-det-0-960`
- Parent: E12
- Score (private / public): **0.92241 / 0.95212**

> # Biohub K3 Hengck Motion DET 0.960
>
> Detector threshold A/B test from the confirmed 0.952 baseline.
> Linker and post-processing settings remain fixed.

Change vs E12:

```diff
-os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"
+os.environ["BIOHUB_DET_THRESHOLD"] = "0.960"
-    "BIOHUB_DET_THRESHOLD": 0.965,
+    "BIOHUB_DET_THRESHOLD": 0.960,
+# Detector-only A/B test guard.
+assert abs(DET_THRESHOLD - 0.960) < 1e-12
+assert OUTPUT_MOTION_RELINK is True and OUTPUT_GAP_CLOSE is True
+assert OUTPUT_GAP2_RECOVERY is True and OUTPUT_MIN_TRACK_LEN == 6
+
```


## E21 - secondary model weight 0.80 -> 0.50

- Kaggle notebook: `biohub-k3-hengck-motion-secondary-det-050`
- Parent: E12
- Score (private / public): **0.92406 / 0.94881**
- Full notebook: `notebooks/05_k3_hengck_secondary_det_050_best_private.ipynb`

> # Biohub K3 Hengck Motion Secondary DET 0.50
>
> Single-variable test from the confirmed 0.952 baseline.
> Only the secondary model contribution to detector logits changes from 0.80 to 0.50.

Change vs E12:

```diff
-os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"  
+os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.50"  
+# Secondary-detection-only A/B test guard.
+assert abs(float(os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT']) - 0.50) < 1e-12
+assert abs(DET_THRESHOLD - 0.965) < 1e-12
+assert OUTPUT_MOTION_RELINK is True and OUTPUT_GAP_CLOSE is True
+assert OUTPUT_GAP2_RECOVERY is True and OUTPUT_MIN_TRACK_LEN == 6
+
```


## E22 - detector threshold 0.970

- Kaggle notebook: `biohub-k3-hengck-motion-det-0-970`
- Parent: E12
- Score (private / public): **0.92095 / 0.95218**

> # Biohub K3 Hengck Motion DET 0.970
>
> Detector threshold A/B test from the confirmed 0.952 baseline.
> Linker and post-processing settings remain fixed.

Change vs E12:

```diff
-os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"
+os.environ["BIOHUB_DET_THRESHOLD"] = "0.970"
-    "BIOHUB_DET_THRESHOLD": 0.965,
+    "BIOHUB_DET_THRESHOLD": 0.970,
+# Detector-only A/B test guard.
+assert abs(DET_THRESHOLD - 0.970) < 1e-12
+assert OUTPUT_MOTION_RELINK is True and OUTPUT_GAP_CLOSE is True
+assert OUTPUT_GAP2_RECOVERY is True and OUTPUT_MIN_TRACK_LEN == 6
+
```


## E23 - cached edge scores (Harmonic Fusion)

- Kaggle notebook: `biohub-cell-tracking-953-edge-cache-experiment`
- Parent: -
- Score (private / public): **0.91724 / 0.95558**
- Full notebook: `notebooks/06_harmonic_fusion_edge_cache_best_public.ipynb`

> reported public 0.953 reference + experimental cached-edge motion scores (not yet scored)

