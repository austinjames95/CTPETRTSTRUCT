from dicompylercore import dicomparser, dvhcalc
import pydicom
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import SimpleITK as sitk
import csv

def get_dose_resolution(rd_dataset):
    dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling
    unique_doses = np.unique(dose_array.flatten())
    dose_diffs = np.diff(np.sort(unique_doses))
    min_step = np.min(dose_diffs[dose_diffs > 0])
    return min_step, np.max(unique_doses)

# Set the folder with DICOM files
base_dir = r'C:\\Users\\austi\\OneDrive\\Desktop\\COH\\BGRT^P_uPTV5mm4cmSphere_20250110_160d82'

# Global Variables
ct_datasets = []
pet_datasets = []
rs_dataset = None
rd_dataset = None

print("\nscanning files\n")

dicom_files = glob.glob(os.path.join(base_dir, '*'))
for file_path in dicom_files:
    try:
        ds = pydicom.dcmread(file_path)
        modality = ds.Modality
        print(f"{os.path.basename(file_path)}: Modality = {modality}")

        if modality == 'CT':
            ct_datasets.append(pydicom.dcmread(file_path))
        elif modality == 'PT':
            pet_datasets.append(pydicom.dcmread(file_path))
        elif modality == 'RTSTRUCT':
            rs_dataset = pydicom.dcmread(file_path)
        elif modality == 'RTDOSE':
            rd_dataset = pydicom.dcmread(file_path)

    except Exception as e:
        print(f"Could not read {file_path}: {e}")

print("\n Load Summary \n")

print(f"Loaded {len(ct_datasets)} CT slices")
print(f"Loaded {len(pet_datasets)}  PET slices")

if rs_dataset:
    print("\nRTSTRUCT loaded - PASS - Ready")
else:
    print("\nRTSTRUCT loaded - FAIL")

if rd_dataset:
    print("\nRTDOSE loaded - PASS - Ready")
else:
    print("\nRTDOSE loaded - FAIL")

if ct_datasets:
    print("\nLaunching CT slice viewer...")
    # scroll_through_slices(ct_datasets, modality_name="CT", cmap='gray')

if pet_datasets:
    print("\nLaunching PET slice viewer...")
    # scroll_through_slices(pet_datasets, modality_name="PET", cmap='hot')


if rs_dataset and rd_dataset:
    print("\n🔍 Parsing DICOM datasets with dicompylercore:")

    dp_struct = None
    dp_dose = None

    try:
        dp_struct = dicomparser.DicomParser(rs_dataset)
        print("RTSTRUCT parsed successfully.")
    except Exception as e:
        print("Failed to parse RTSTRUCT:", e)

    try:
        dp_dose = dicomparser.DicomParser(rd_dataset)
        print("RTDOSE parsed successfully.")
    except Exception as e:
        print("Failed to parse RTDOSE:", e)

    if not dp_struct or not dp_dose:
        print("Cannot continue. One or both required datasets failed to parse.")
        exit(1)

    try:
        structures = dp_struct.GetStructures()
    except Exception as e:
        print("Failed to get structures from RTSTRUCT:", e)
        exit(1)

    print("\nAvailable Structures:")

    for ids, struct in structures.items():
        print(f"  [{ids}] {struct['name']}")

    for sid, struct in structures.items():
        try:
            dose_step, max_dose = get_dose_resolution(rd_dataset)
            custom_bins = np.arange(0, max_dose + dose_step, dose_step)

            dvh = dvhcalc.get_dvh(dp_struct.ds, dp_dose.ds, sid, calculate_full_volume=True, bin=custom_bins)


            if dvh:
                print(f"\nStructure: {struct['name']}")
                print(f"  Volume (cc): {dvh.volume:.2f}")
                print(f"  Mean Dose (Gy): {dvh.mean:.2f}")
                print(f"  Max Dose (Gy): {dvh.max:.2f}")
                print(f"  Min Dose (Gy): {dvh.min:.2f}")

                # Export DVH to CSV
                safe_name = struct['name'].replace(" ", "_").replace("/", "_")
                output_path = os.path.join("dvh_data", f"{safe_name}_DVH_dicompyler.csv")
                os.makedirs("dvh_data", exist_ok=True)

                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Dose (Gy)", "Volume (cm3)"])

                    dose_bins = dvh.absolute_dose().bins
                    volume_counts = dvh.counts

                    prev_vol = None
                    ignore_dec = False

                    for dose, vol in zip(dose_bins, volume_counts):
                        if prev_vol is None:
                            diff_vol = 0
                            ignore_dec = True
                        else:
                            diff_vol = prev_vol - vol
                            ignore_dec = False

                        if ignore_dec:
                            writer.writerow([f"{dose:.4f}", f"0"])
                        else:
                            writer.writerow([f"{dose: 4f}", f"{diff_vol: 7f}"])
                        prev_vol = vol

                bqml_dir = os.path.join("dvh_data_bqml")
                os.makedirs(bqml_dir, exist_ok=True)
                bqml_path = os.path.join(bqml_dir, f"{safe_name}_DVH_BQML.csv")

                with open(bqml_path, "w", newline="") as bqml_file:
                    writer = csv.writer(bqml_file)
                    writer.writerow(["Dose (BQ/ML)", "Volume (cm3)"])

                    dose_bins = dvh.absolute_dose().bins
                    volume_counts = dvh.counts

                    prev_vol = None

                    for dose, vol in zip(dose_bins, volume_counts):
                        if prev_vol is None:
                            diff_vol = 0
                        else:
                            diff_vol = prev_vol - vol
                        bqml_dose = dose
                        writer.writerow([bqml_dose, diff_vol])
                        prev_vol = vol

        except Exception as e:
            print(f"Error computing DVH for {struct['name']}: {e}")
