# Effect of voxel spacing and field of view on cross-center breast DCE-MRI tumor segmentation

Code accompanying the paper

> M. Musaev et al., *Effect of voxel spacing and field of view on cross-center breast DCE-MRI tumor segmentation*, submitted to ANT 2027 (Procedia Computer Science).

The study measures what geometric preprocessing choices cost in Dice under a
leave-one-source-out (LOSO) protocol on the MAMA-MIA benchmark, by varying one
factor at a time:

* **Resolution sweep** – effective in-plane spacing 1.5, 2.0, 3.0 and 4.55 mm, with
  field of view, apparent lesion scale, pixel count and reference masks held fixed.
* **Magnification sweep** – magnification k = 0.5, 1, 2 about the image center,
  which changes apparent lesion scale and field of view together; lesion
  retention separates the two at analysis time.

Held-out sources: NACT (n = 64) and DUKE (n = 290). Two seeds per configuration,
averaged per case before statistics.

## Repository layout

```
notebooks/
  01_exploration.ipynb          dataset inventory, native geometry per source (Table 1)
  02_preprocessing.ipynb        raw MAMA-MIA -> per-source HDF5 (192 x 192, 1.5 mm)
  03_variants.ipynb             preprocessing variants used by the sweeps (V2_iso_scale)
  04_resolution_sweep.ipynb     LOSO training + resolution dose-response
  05_magnification_sweep.ipynb  LOSO training + magnification / truncation
  06_figure_slices.ipynb        exports example slices for the qualitative figure
analysis/
  stats_final.py                every number in the paper -> results/numbers.json
  figs.py                       Figure 2 (resolution, lesion size, truncation)
  fig_qual.py                   Figure 1 (needs results/figure_slices.npz from notebook 06)
results/
  ressweep_percase.csv          per-case, per-seed metrics of the resolution sweep
  magsweep_percase.csv          per-case, per-seed metrics of the magnification sweep
  numbers.json                  output of stats_final.py
```

Notebook comments are partly in Uzbek; code and variable names are in English.

## Reproducing

1. **Data.** Obtain MAMA-MIA from its authors (Garrucho et al., *Scientific Data*
   12:453, 2025, doi:10.1038/s41597-025-04707-4). Notebook 02 downloads the
   MAMA-MIA-Lite mirror from the Hugging Face Hub. The imaging data are licensed
   CC BY-NC 4.0 and are **not** redistributed here.
2. **Preprocessing.** Run notebooks 01 → 03. They were written for Kaggle
   (`/kaggle/input`, `/kaggle/working`) and fall back to local paths.
3. **Experiments.** Run notebooks 04 and 05 on a GPU (written for Kaggle). Each writes a `*_percase.csv` identical in format to the files
   in `results/`.
4. **Statistics and figures.**
   ```bash
   pip install -r requirements.txt
   python analysis/stats_final.py
   python analysis/figs.py
   ```
   Figures are written to `figures/`. With the CSVs shipped in `results/`,
   step 4 reproduces all reported numbers without retraining.

## Citation

If you use this code, please cite the paper above and the MAMA-MIA dataset.

## License

Code: MIT (see `LICENSE`). Data: not included; MAMA-MIA is distributed by its
authors under CC BY-NC 4.0.
