import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon, Patch
from matplotlib.colors import to_rgba
from skimage.draw import polygon
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

def compute_dvh_from_mask(structure_name, contours_by_slice, dose_array, voxel_volume):
    interpolated_doses = []

    for z_index, contours in contours_by_slice.items():
        for contour in contours:
            if contour['roi'] != structure_name:
                continue

            coords = np.array(contour['coords'])
            if coords.shape[0] < 3:
                continue

            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=(dose_array.shape[1], dose_array.shape[2]))
            slice_coords = np.vstack([np.full(rr.shape, z_index), rr, cc])

            interpolated_values = map_coordinates(dose_array, slice_coords, order=1, mode='nearest')
            interpolated_doses.extend(interpolated_values)

    if not interpolated_doses:
        return None

    interpolated_doses = np.array(interpolated_doses)
    volume_cc = len(interpolated_doses) * voxel_volume / 1000

    bins = np.linspace(0, np.max(interpolated_doses), 100)
    hist, bin_edges = np.histogram(interpolated_doses, bins=bins)
    cumulative = np.cumsum(hist[::-1])[::-1] * voxel_volume / 1000

    stats = {
        'max': np.max(interpolated_doses),
        'mean': np.mean(interpolated_doses),
        'min': np.min(interpolated_doses),
        'volume': volume_cc,
        'hist': hist,
        'bins': bin_edges,
        'cumulative': cumulative,
    }
    return stats

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
