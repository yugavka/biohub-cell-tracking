# Biohub: Cell Tracking During Development

![Kaggle](https://img.shields.io/badge/Kaggle-bronze%20medal-cd7f32) ![Rank](https://img.shields.io/badge/rank-252%20%2F%203947-blue)

My work on the Kaggle competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
detect zebrafish cells in 3D microscopy, link them across time and find cell divisions to reconstruct lineages.
Final result: **bronze medal, 252nd of 3,947 teams**.

This repository is a record of the 23 submissions I made: the key notebooks, a log of every experiment with the
hypothesis it tested, and the exact code change against its parent.

## Results

![Public and private score of every submission](assets/scores.png)

Score is `adjusted_edge_jaccard + 0.1 * division_jaccard` (higher is better; it can exceed 1.0).

| ID | Change | Public | Private | Comment |
|----|--------|-------:|--------:|---------|
| E01 | Harmonic Fusion baseline | 0.94762 | 0.91448 | first submission |
| E03 | Hengck end-to-end linker only | 0.92437 | 0.90700 | swapping in a single end-to-end linker hurt badly |
| E06 | post-process, then Hengck gate | 0.94677 | 0.91342 | Hengck gate after post-processing recovers the loss |
| E07 | K3 + Hengck fork-transfer base | 0.95133 | 0.91740 | new base: K3 + Hengck fork transfer |
| E12 | motion-aware relink | 0.95228 | 0.92124 | motion-aware relink: +0.004 private over E07 |
| E15 | long-edge gate (>8.4 um) | 0.95286 | 0.92219 | best public among the relink ablations |
| E21 | secondary model weight 0.80 -> 0.50 | 0.94881 | 0.92406 | **best private score** |
| E23 | cached edge scores (Harmonic Fusion) | 0.95558 | 0.91724 | **best public score**, but private drops |

All 23 submissions are in [`experiments/results.csv`](experiments/results.csv) and described in
[`experiments/LOG.md`](experiments/LOG.md).

### What the experiments showed

- **Linker choice matters more than any single knob.** Replacing the baseline linker with a single end-to-end
  linker dropped the public score from 0.947 to 0.924 (E03); gating its edges against the baseline recovered it (E05, E06).
- **A better base gave the largest step.** The K3 + Hengck fork-transfer pipeline (E07) improved both scores over the
  best Harmonic Fusion version (E06): public 0.94677 to 0.95133, private 0.91342 to 0.91740.
- **Motion-aware relinking helped on both leaderboards** (E12: public +0.001, private +0.004 over E07).
- **Public and private scores disagreed.** The secondary-model weight of 0.50 (E21) has the best private score
  (0.92406) and one of the lowest public scores (0.94881); cached edge scores (E23) have the best public score
  (0.95558) but a private score below E12. Selecting by public score alone would have picked the wrong runs.

## Approach

The pipeline is configured through `BIOHUB_*` environment variables at the top of each notebook. Main stages:

1. **Detection**: 3D U-Net detectors with a primary and a secondary model fused at the logit level; threshold 0.965 by default.
2. **Linking**: candidate edges scored by temporal models (with edge-feature test-time augmentation and
   bidirectional fusion), then solved as an ILP with SCIP (appearance, disappearance and division weights).
3. **Post-processing**: motion-aware relink with a neighbourhood flow prior, gap closing over one and two frames,
   short-track filtering with rescue, and geometry-validated divisions (parent, sister and divergence gates).
4. **Veto**: a DeepCenter 3D U-Net center prior that can veto gap fills and divisions.
5. **Validation**: a re-implementation of the competition metric (node matching within 7.0 um, voxel size
   1.625 x 0.40625 x 0.40625 um) used for held-out checks.

## Repository layout

```
notebooks/      key notebooks, outputs stripped (Kaggle notebooks, GPU, internet off)
experiments/
  LOG.md        every experiment: hypothesis, result, diff against its parent
  results.csv   scores of all 23 submissions
assets/         figures
```

| Notebook | Experiment |
|----------|------------|
| `01_harmonic_fusion_baseline_v1.ipynb` | E01, first baseline |
| `02_harmonic_fusion_hengck_gate_v6.ipynb` | E06, best of the first phase |
| `03_k3_hengck_base_951.ipynb` | E07, base of the second phase |
| `04_k3_hengck_motion_relink_952.ipynb` | E12, motion relink |
| `05_k3_hengck_secondary_det_050_best_private.ipynb` | E21, best private score |
| `06_harmonic_fusion_edge_cache_best_public.ipynb` | E23, best public score |

## Reproducing

The notebooks run only on Kaggle: they read the competition data and third-party model datasets from
`/kaggle/input` and need a GPU (T4) with internet disabled.

1. Create a Kaggle notebook and attach the competition data plus the datasets listed under Credits.
2. Upload one of the notebooks from `notebooks/` and run it on a GPU accelerator.
3. Submit the resulting `submission.csv` (nodes and edges in the competition format).

## Credits and notes

- The pipeline builds on public Kaggle resources that are **not** included here:
  datasets by [pilkwang](https://www.kaggle.com/pilkwang) (`biohub-deepcenter-unet3d-center-prior-v1`,
  `biohub-temporal-unet3d-seed314159-v1`, `biohub-tracking-support-pack-50ep-v1`) and by
  [hengck23](https://www.kaggle.com/hengck23) (`hengck23-cell-point-detector-demo`), plus the
  `tracksdata` library. Respect their licenses and terms if you reuse them.
- The K3 notebooks embed the code of those upstream pipelines in their own `FILES` dictionary.
- Competition and metric by Biohub; see the competition page for the full citation.
