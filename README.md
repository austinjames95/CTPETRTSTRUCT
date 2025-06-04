# PET-Dose Analysis Toolkit

## 🔗 Overview

This project supports the visualization and analysis of PET (Positron Emission Tomography) and RTDOSE data from DICOM files. It includes features for PET-Dose overlay visualization, hotspot analysis, and activity profiling across slices.

---

## 📅 Update: June 4, 2025

### ✅ New Additions

#### 📸 `PET&DOSE.PNG`

Overlay visualization showing both PET activity and RTDOSE distribution. This enables better assessment of spatial alignment between functional imaging and treatment planning.

#### 🔥 `PET_Hotspot_Analysis.PNG`

Automatically identifies and visualizes the hottest (most active) PET slice. Includes a secondary image thresholded at `mean + 2*std` to reveal concentrated regions of high activity.

#### 📈 `PET_Activity_Profile.PNG`

Two-panel line plot:

* **Left**: Mean PET activity per slice
* **Right**: Maximum PET activity per slice
  Useful for evaluating activity trends throughout the volume.

---

## Next Steps

We're planning to extend this analysis with ROI-based sampling and reporting:

* [ ] Extract PET activity values **within each structure/ROI** (e.g., GTV, PTV).
* [ ] Export PET statistics **per structure** to CSV:

  * Mean, max, and standard deviation of activity
  * Volume and fraction of active voxels
* [ ] Enable structure-specific overlays and measurements during interactive review

Stay tuned for further updates!

---

## 🔄 How to Use

1. Place DICOM CT, PET, RTSTRUCT, RTDOSE, and REG files in a single folder.
2. Run `main.py` to trigger processing.
3. Results will be saved in the `dvh_data_combined` folder.
4. Interactive viewer will launch if PET and dose data are available.

---

## 🌐 Output Files

| File                       | Description                                |
| -------------------------- | ------------------------------------------ |
| `PET&DOSE.PNG`             | PET and dose overlay visualization         |
| `PET_Hotspot_Analysis.PNG` | Hottest PET slice with high activity zones |
| `PET_Activity_Profile.PNG` | Activity trends across slices              |
| `CumulativeDVH_*.csv`      | Absolute and relative DVH data exports     |

---
