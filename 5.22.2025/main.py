from dicompylercore import dicomparser, dvhcalc
import pydicom
import os
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon, Patch
from matplotlib.colors import to_rgba
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from skimage.draw import polygon
import tempfile
import shutil
from shapely.geometry import Polygon as ShapelyPolygon
import SimpleITK as sitk
import csv
from collections import defaultdict
from submain import scroll_overlay, extract_contours, scroll_through_slices, scroll_overlay, compute_dvh_from_mask, create_structure_mask_from_contours

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

    try:
        allPrint = str(input("\nDo you want to print all on a single graph? (Y/N): ")).strip()
        if allPrint.lower() == 'y':
            for sid, struct in structures.items():
                try:
                    voxel_volume = float(rd_dataset.PixelSpacing[0]) * float(rd_dataset.PixelSpacing[1]) * float(rd_dataset.SliceThickness)
                    stats = compute_dvh_from_mask(struct['name'], contours_by_slice, dose_array, voxel_volume)
                    if stats:
                        prescription_dose = np.percentile(dose_array[dose_array > 0], 98)  # or set manually


                        rel_dose = (stats['bins'][:-1] / prescription_dose) * 100
                        rel_volume = (stats['cumulative'] / stats['volume']) * 100

                        plt.plot(rel_dose, rel_volume, label=struct['name'])
                        print(f"{struct['name']}: Max {stats['max']:.2f} Gy, Mean {stats['mean']:.2f} Gy, Volume {stats['volume']:.2f} cc")
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

            print(f"\n\n")

        elif allPrint.lower() == 'n':
            print("If you wish to skip the graphing please enter 0")
            target_id = int(input("Enter the ID of the target structure: "))
            if target_id not in structures and target_id != 0:
                print(f"Not a valid ID")
                sys.exit(1)
            elif target_id == 0:
                sys.exit(1)
            else:
                print("Calculating Data for " + structures[target_id]["name"] + "\n")
                voxel_volume = float(rd_dataset.PixelSpacing[0]) * float(rd_dataset.PixelSpacing[1]) * float(rd_dataset.SliceThickness)
                target_name = structures[target_id]['name']
                stats = compute_dvh_from_mask(target_name, contours_by_slice, dose_array, voxel_volume)
                if stats:
                    print(f"DVH for {target_name}:")
                    print(f"  Max Dose:  {stats['max']:.2f} Gy")
                    print(f"  Mean Dose: {stats['mean']:.2f} Gy")
                    print(f"  Min Dose:  {stats['min']:.2f} Gy")
                    print(f"  Volume:    {stats['volume']:.2f} cc")

                    plt.figure(figsize=(8, 6))
                    plt.plot(stats['bins'][:-1], stats['cumulative'], label=target_name)
                    plt.title(f"DVH: {target_name}")
                    plt.ylabel("Volume (cc)")
                    plt.xlabel("Dose (Gy)")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    print("DVH not generated — structure may not intersect the dose grid.")

    except Exception as e:
                        print("Error during DVH calculation:", str(e))
else:
    print("Missing required datasets (CT, RS, or RD).")

if rs_dataset and rd_dataset:
    try:
        print("\nCalculating DVHs using dvhcalc and extracting stats...")
        dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling

        output_csv_path = os.path.join(base_dir, "structure_dvhcalc_stats.csv")
        with open(output_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Structure Name", "Mean Dose (Gy)", "Max Dose (Gy)", "Min Dose (Gy)", "Volume (cc)"])

            mean_doses = []
            structure_names = []

            for sid, s in structures.items():
                print(f"Processing structure: {s['name']}")
                dvh = dvhcalc.get_dvh(rs_dataset, rd_dataset, sid)

                if dvh:
                    writer.writerow([
                        s['name'],
                        f"{dvh.mean:.2f}",
                        f"{dvh.max:.2f}",
                        f"{dvh.min:.2f}",
                        f"{dvh.volume:.2f}"
                    ])

                    plt.plot(dvh.bins[:-1], dvh.counts, label=s['name'])
                    print(f"  Mean: {dvh.mean:.2f} Gy, Max: {dvh.max:.2f} Gy, Min: {dvh.min:.2f} Gy, Volume: {dvh.volume:.2f} cc")

                    mean_doses.append(dvh.mean)
                    structure_names.append(s['name'])
                else:
                    print(f"  Skipping {s['name']} — DVH could not be generated.")

        plt.title("DVH from dvhcalc")
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume (cc")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(f"\nDVH statistics written to: {output_csv_path}")

    except Exception as e:
        print("Error during dvhcalc DVH analysis:", str(e))
