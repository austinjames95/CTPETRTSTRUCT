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
from shapely.ops import unary_union
from skimage.draw import polygon
import tempfile
import shutil
from shapely.geometry import Polygon as ShapelyPolygon
import SimpleITK as sitk

# load DICOM series
def load_dicom_series(path):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    return image, array

# get contours
def extract_contours(rtstruct, ct_image):
    roi_map = {roi.ROINumber: roi.ROIName for roi in rtstruct.StructureSetROISequence}
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)
    origin = np.array(ct_image.GetOrigin())
    spacing = np.array(ct_image.GetSpacing())
    contours_by_slice = {}

    for roi_contour in rtstruct.ROIContourSequence:
        roi_num = roi_contour.ReferencedROINumber
        roi_name = roi_map.get(roi_num, f"ROI {roi_num}")
        for contour in roi_contour.ContourSequence:
            coords = np.array(contour.ContourData).reshape(-1, 3)
            z = coords[0, 2]
            slice_index = int(round((z - origin[2]) / spacing[2]))
            pixels = []
            for x, y, z in coords:
                v = np.array([x, y, z]) - origin
                ijk = np.linalg.inv(direction).dot(v / spacing)
                pixels.append((ijk[0], ijk[1]))
            contours_by_slice.setdefault(slice_index, []).append({"roi": roi_name, "coords": pixels})
    return contours_by_slice

# Implement Scrolling
def scroll_through_slices(datasets, modality_name, cmap='gray'):
    datasets = sorted(datasets, key=lambda x: float(x.ImagePositionPatient[2]))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    first_img = datasets[0].pixel_array
    if modality_name == "PET":
        window_min, window_max = 0, 5000 
        first_img = np.clip(first_img, window_min, window_max)
        first_img = ((first_img - window_min) / (window_max - window_min)) * 255
        first_img = first_img.astype(np.uint8)

    img = ax.imshow(first_img, cmap=cmap)
    ax.set_title(f"{modality_name} Slice 0")

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(datasets) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        pixel_data = datasets[idx].pixel_array

        if modality_name == "PET":
            pixel_data = np.clip(pixel_data, window_min, window_max)
            pixel_data = ((pixel_data - window_min) / (window_max - window_min)) * 255
            pixel_data = pixel_data.astype(np.uint8)

        img.set_data(pixel_data)
        ax.set_title(f"{modality_name} Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def scroll_overlay(ct_array, pet_array, dose_array, contours_by_slice):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    slice_idx = 0
    im_ct = ax.imshow(ct_array[slice_idx], cmap='gray')
    im_pet = ax.imshow(pet_array[slice_idx], cmap='hot', alpha=0.4)
    im_dose = ax.imshow(dose_array[slice_idx], cmap='jet', alpha=0.3)
    patches = []

    unique_rois = set()
    for contour_list in contours_by_slice.values():
        for contour in contour_list:
            unique_rois.add(contour['roi'])
            color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown']
            roi_colors = {roi: color_list[i % len(color_list)] for i, roi in enumerate(sorted(unique_rois))}

    legend_patches = [Patch(facecolor=to_rgba(c, alpha=0.4), edgecolor=c, label=r) for r, c in roi_colors.items()]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.,
        title="Structures",
        fontsize='small',
        title_fontsize='medium'
    )

    def draw_contours(idx):
        nonlocal patches
        for p in patches:
            p.remove()
        patches = []
        for contour in contours_by_slice.get(idx, []):
            roi_name = contour['roi']
            color = roi_colors.get(roi_name, 'lime')  # default
            poly = Polygon(contour['coords'], fill=False, edgecolor=color, linewidth=1)
            ax.add_patch(poly)
            patches.append(poly)

    draw_contours(slice_idx)

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, ct_array.shape[0]-1, valinit=slice_idx, valstep=1)

    ax.set_title(f"CT/PET/Dose/Contours Slice 0")

    def update(val):
        idx = int(slider.val)
        im_ct.set_data(ct_array[idx])
        im_pet.set_data(pet_array[idx])
        im_dose.set_data(dose_array[idx])
        draw_contours(idx)
        ax.set_title(f"CT/PET/Dose/Contours Slice {idx}")
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    def key(event):
        current = slider.val
        if event.key == "right" and current < slider.valmax:
            slider.set_val(current + 1)
            
        elif event.key == "left" and current > slider.valmin:
            slider.set_val(current - 1)

        elif event.key == "up":
            if current + 5 < slider.valmax:
                slider.set_val(current + 5)
            else:
                slider.set_val(slider.valmax)

        elif event.key == "down":
            if current - 5 > slider.valmin:
                slider.set_val(current - 5)
            else:
                slider.set_val(slider.valmin)
        
        elif event.key == "escape":
            plt.close(event.canvas.figure)
    
    fig.canvas.mpl_connect('key_press_event', key)

    plt.show()

def create_structure_mask_from_contours(contours_by_slice, shape):
    mask = np.zeros(shape, dtype=bool)
    for z_index, contours in contours_by_slice.items():
        for contour in contours:
            coords = np.array(contour['coords'])
            if coords.shape[0] < 3:
                continue
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=(shape[1], shape[2]))
            if 0 <= z_index < shape[0]:
                mask[z_index, rr, cc] = True
    return mask

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
                    dvh = dvhcalc.get_dvh(rs_dataset, rd_dataset, sid)
                    if dvh:
                        plt.plot(dvh.counts, dvh.bins[:-1], label=struct['name'])
                        print(f"{struct['name']}: Max {dvh.max:.2f} Gy, Mean {dvh.mean:.2f} Gy, Volume {dvh.volume:.2f} cc")
                    else:
                        print(f"   Skipped {struct['name']} — DVH not generated.")
                except Exception as e:
                    print(f"   Error processing {struct['name']}: {str(e)}")

            plt.title("DVH for All Structures")
            plt.xlabel("Dose (Gy)")
            plt.ylabel("Volume (cc)")
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
                dvh = dvhcalc.get_dvh(rs_dataset, rd_dataset, target_id)
                if dvh:
                    print(f"DVH for {structures[target_id]['name']}:")
                    print(f"  Max Dose:  {dvh.max:.2f} Gy")
                    print(f"  Mean Dose: {dvh.mean:.2f} Gy")
                    print(f"  Min Dose:  {dvh.min:.2f} Gy")
                    print(f"  Volume:    {dvh.volume:.2f} cc")

                    plt.figure(figsize=(8, 6))
                    plt.plot(dvh.counts, dvh.bins[:-1], label=structures[target_id]['name'])
                    plt.title(f"DVH: {structures[target_id]['name']}")
                    plt.ylabel("Dose (Gy)")
                    plt.xlabel("Volume (cc)")
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
