# PET-Dose Analysis Toolkit

## Overview

This project supports the visualization and analysis of PET and RTDOSE data from DICOM files. It includes features for PET-Dose overlay visualization, hotspot analysis, and activity profiling across slices.

---

## Update: June 6, 2025

### New Additions

#### 'CumulativePVH_AllStructures_AbsoluteUnits.csv'

Exports BQML data 0-413914 and shows the dosage effect overtime for all ROI structures.

## Removals

Removed Dose Data from the pet overlay, and instead replaced it with the CT data overlay

## CT Overlay

Resized the Pet image so it correctly fits over the CT overlay so the structures can properly be analyized

## Known Problems

The Pet image overlay origin needed to be moved by 34mm higher as it does not have the same amount of slices as the CT imaging
Data inside the 'CumulativePVH_AllStructures_AbsoluteUnits.csv' is slightly off from what is expected

## Next Steps

I am planning to fix the issues with the errors in the PVH data. Additionally I will look into the issue regarding the origin deviation.
Additionally I will be removing the interactive windows initally implemented for debugging the overlays of image scans.
Possible removal of the generated graphs depending if they are necessary.

---

## Update: June 4, 2025

### New Additions

#### `PET&DOSE.PNG`

Overlay visualization showing both PET activity and RTDOSE distribution. This enables better assessment of spatial alignment between functional imaging and treatment planning.

#### `PET_Hotspot_Analysis.PNG`

Automatically identifies and visualizes the hottest (most active) PET slice. Includes a secondary image thresholded at `mean + 2*std` to reveal concentrated regions of high activity.

#### `PET_Activity_Profile.PNG`

Two-panel line plot:

* **Left**: Mean PET activity per slice
* **Right**: Maximum PET activity per slice
  Useful for evaluating activity trends throughout the volume.

---

## Next Steps

I am planning to extend this analysis with ROI-based sampling and reporting:

* Extract PET activity values **within each structure/ROI** (e.g., GTV, PTV).
* Export PET statistics **per structure** to CSV:

  * Mean, max, and standard deviation of activity
  * Volume and fraction of active voxels
* Enable structure-specific overlays and measurements during interactive review

---

## How to Use

1. Place DICOM CT, PET, RTSTRUCT, RTDOSE, and REG files in a single folder.
2. Run `main.py` to trigger processing.
3. Results will be saved in the `dvh_data_combined` folder.
4. Interactive viewer will launch if PET and dose data are available.

---
