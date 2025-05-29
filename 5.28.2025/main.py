from dicompylercore import dicomparser, dvhcalc
import pydicom
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import SimpleITK as sitk
import csv
from submain import scroll_overlay, extract_contours, scroll_overlay, compute_dvh_from_mask

# load DICOM series
def load_dicom_series(path):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    return image, array

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


if rs_dataset and rd_dataset and ct_datasets:

    print("\nLaunching combined overlay viewer...")
    
    temp_dir = tempfile.mkdtemp()
    for ds in ct_datasets:
        ds.save_as(os.path.join(temp_dir, f"ct_{ds.InstanceNumber}.dcm"))
    ct_img, ct_array = load_dicom_series(temp_dir)
    pet_img, pet_array = load_dicom_series(temp_dir)
    shutil.rmtree(temp_dir)

    pet_resampled = sitk.Resample(pet_img, ct_img)
    pet_array = sitk.GetArrayFromImage(pet_resampled)

    dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling

    contours_by_slice = extract_contours(rs_dataset, ct_img)

    scroll_overlay(ct_array, pet_array, dose_array, contours_by_slice)

    dp_struct = dicomparser.DicomParser(rs_dataset)
    structures = dp_struct.GetStructures()

    print("\nAvailable Structures:")

    for ids, struct in structures.items():
        print(f"  [{ids}] " + struct['name'])

    allPrint = str(input("\nDo you want to print all on a single graph? (Y/N): ")).strip()

    if allPrint.lower() == 'y':
        output_dir = os.getcwd()
        summary_csv = os.path.join(output_dir, "dvh_summary_stats.csv")
        dvh_data_dir = os.path.join(output_dir, "dvh_data")
        os.makedirs(dvh_data_dir, exist_ok=True)
        
        try:
            with open(summary_csv, mode='w', newline='') as summary_file:
                summary_writer = csv.writer(summary_file)
                summary_writer.writerow(["Structure", "Max Dose (Gy)", "Mean Dose (Gy)", "Min Dose (Gy)", "Volume (cc)"])

                for sid, struct in structures.items():
                    try:
                        voxel_volume = float(rd_dataset.PixelSpacing[0]) * float(rd_dataset.PixelSpacing[1]) * float(rd_dataset.SliceThickness)
                        stats = compute_dvh_from_mask(struct['name'], contours_by_slice, dose_array, voxel_volume)

                        if stats:
                            prescription_dose = np.percentile(dose_array[dose_array > 0], 98)

                            abs_dose = stats['bins'][:-1]
                            abs_volume = stats['cumulative']

                            rel_dose = (stats['bins'][:-1] / prescription_dose) * 100
                            rel_volume = (stats['cumulative'] / stats['volume']) * 100

                            plt.plot(rel_dose, rel_volume, label=struct['name'])

                            summary_writer.writerow([
                                struct['name'],
                                f"{stats['max']:.2f}",
                                f"{stats['mean']:.2f}",
                                f"{stats['min']:.2f}",
                                f"{stats['volume']:.2f}"
                            ])

                            # DVH data CSV per structure
                            safe_name = struct['name'].replace(' ', '_').replace('/', '_')
                            dvh_path = os.path.join(dvh_data_dir, f"{safe_name}_DVH.csv")
                            with open(dvh_path, mode='w', newline='') as dvh_file:
                                dvh_writer = csv.writer(dvh_file)
                                dvh_writer.writerow(["Dose (Gy)", "Volume (cm3)"])
                                for d, v in zip(abs_dose, abs_volume):
                                    dvh_writer.writerow([f"{d:.2f}", f"{v:.2f}"])
                        else:
                            print(f"   Skipped {struct['name']} — No mask found.")
                    except Exception as e:
                        print(f"   Error processing {struct['name']}: {str(e)}")

            plt.title("Normalized DVH (Relative Dose vs Volume)")
            plt.xlabel("Relative Dose (%)")
            plt.ylabel("Ratio of total structure volume (%)")
            plt.xlim(0, 120)
            plt.ylim(0, 105)
            plt.grid(True)
            plt.legend(fontsize='small', loc='best')
            plt.tight_layout()
            plt.show()

            print(f"\nDVH summary written to {summary_csv}")
            print(f"Full DVH data written to folder: {dvh_data_dir}")

        except Exception as e:
            print("Error during DVH calculation and file writing:", str(e))
