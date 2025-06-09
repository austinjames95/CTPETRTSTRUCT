import os
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from dicompylercore import dicomparser
from scipy import ndimage
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import csv

def get_pet_resolution(pet_volume):
    pet_array = pet_volume.flatten()
    unique_values = np.unique(pet_array[pet_array > 0])  # Only non-zero values
    max_pet = np.max(pet_array)
    
    if len(unique_values) < 2:
        return max_pet / 1000.0, max_pet  # Default to 1000 bins
    
    value_diffs = np.diff(np.sort(unique_values))
    min_step = np.min(value_diffs[value_diffs > 0])
    
    # Prevent extremely small steps that would create too many bins
    max_bins = 10000  # Reasonable maximum number of bins
    min_reasonable_step = max_pet / max_bins
    
    # Use the larger of calculated step or minimum reasonable step
    final_step = max(min_step, min_reasonable_step)
    
    print(f"PET resolution calculation:")
    print(f"  Calculated min step: {min_step}")
    print(f"  Min reasonable step: {min_reasonable_step}")
    print(f"  Final step used: {final_step}")
    
    return final_step, max_pet

def sort_pet_slices(pet_datasets):
    """Sort PET slices by position for proper visualization"""
    pet_with_position = []
    for ds in pet_datasets:
        try:
            # Try to get slice position
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                z_pos = float(ds.SliceLocation)
            else:
                z_pos = 0.0
            pet_with_position.append((z_pos, ds))
        except:
            pet_with_position.append((0.0, ds))
    
    # Sort by z position
    pet_with_position.sort(key=lambda x: x[0])
    return [ds for _, ds in pet_with_position]

def create_pet_volume(pet_datasets):
    """Create 3D volume from PET slices"""
    if not pet_datasets:
        return None, None
    
    # Sort slices
    sorted_pet = sort_pet_slices(pet_datasets)
    first_slice = sorted_pet[0]
    
    # Get dimensions
    rows = first_slice.Rows
    cols = first_slice.Columns
    num_slices = len(sorted_pet)
    
    # Initialize volume
    pet_volume = np.zeros((num_slices, rows, cols))
    
    # Fill volume
    for i, ds in enumerate(sorted_pet):
        try:
            # Get pixel data
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            pet_volume[i] = pixel_array
        except Exception as e:
            print(f"Error processing PET slice {i}: {e}")
            pet_volume[i] = np.zeros((rows, cols))
    
    return pet_volume, sorted_pet

def create_dose_volume(rd_dataset, pet_volume_shape):
    """Create 3D dose volume from RT Dose dataset with proper alignment"""
    if not rd_dataset:
        return None
    
    try:
        # Get dose array
        dose_array = rd_dataset.pixel_array.astype(np.float32)
        
        # Apply dose grid scaling
        if hasattr(rd_dataset, 'DoseGridScaling'):
            dose_array *= float(rd_dataset.DoseGridScaling)
        
        print(f"Original dose shape: {dose_array.shape}")
        print(f"Target PET shape: {pet_volume_shape}")
        
        # Handle dimension mismatch by resizing dose to match PET
        if dose_array.shape != pet_volume_shape:
            print("Resizing dose volume to match PET dimensions...")
            
            # Calculate zoom factors
            zoom_factors = (
                pet_volume_shape[0] / dose_array.shape[0],
                pet_volume_shape[1] / dose_array.shape[1], 
                pet_volume_shape[2] / dose_array.shape[2]
            )
            
            print(f"Zoom factors: {zoom_factors}")
            
            # Resize using scipy
            dose_array = ndimage.zoom(dose_array, zoom_factors, order=1)  # Linear interpolation
            print(f"Resized dose shape: {dose_array.shape}")
        
        return dose_array
        
    except Exception as e:
        print(f"Error creating dose volume: {e}")
        return None

def get_structure_contours(rs_dataset, structures, pet_datasets):
    """Extract structure contours for each slice"""
    if not rs_dataset or not structures:
        return {}
    
    dp_struct = dicomparser.DicomParser(rs_dataset)
    structure_contours = {}
    
    for sid, struct in structures.items():
        try:
            # Get structure data
            structure_data = dp_struct.GetStructureCoordinates(sid)
            if not structure_data:
                continue
                
            structure_contours[sid] = {
                'name': struct['name'],
                'color': struct.get('color', [255, 0, 0]),  # Default to red
                'contours': structure_data
            }
        except Exception as e:
            print(f"Error getting contours for {struct['name']}: {e}")
    
    return structure_contours

class InteractivePETStructureViewer:
    def __init__(self, pet_volume,ct_volume, structure_contours, sorted_pet):
        self.pet_volume = pet_volume
        self.structure_contours = structure_contours
        self.sorted_pet = sorted_pet
        self.current_slice = 0
        self.visible_structures = {sid: True for sid in structure_contours.keys()}
        self.show_pet = True
        self.ct_volume = ct_volume
        self.show_ct = True
        
        self.setup_figure()
        
    def setup_figure(self):
        """Setup the interactive figure"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main image axis
        self.ax_main = plt.axes([0.1, 0.3, 0.6, 0.6])
        
        # Slider for slice selection
        ax_slice = plt.axes([0.1, 0.1, 0.6, 0.03])
        self.slider_slice = Slider(
            ax_slice, 'Slice', 0, self.pet_volume.shape[0] - 1,
            valinit=0, valfmt='%d'
        )
        self.slider_slice.on_changed(self.update_slice)
        
        # Checkboxes for structure visibility
        ax_check = plt.axes([0.75, 0.3, 0.2, 0.6])
        
        # Create checkbox labels and initial states
        structure_names = []
        
        # Add PET and Dose toggles first
        structure_names.extend(['CT Data', 'PET Data'])
        init_states = [self.show_ct, self.show_pet]
        
        self.check_buttons = CheckButtons(ax_check, structure_names, init_states)
        self.check_buttons.on_clicked(self.toggle_visibility)

        self.update_display()
        
    def toggle_visibility(self, label):
        if label == 'CT Data':
            self.show_ct = not self.show_ct
        elif label == 'PET Data':
            self.show_pet = not self.show_pet
        
    def update_slice(self, val):
        """Update the displayed slice"""
        self.current_slice = int(self.slider_slice.val)
        self.update_display()
        
    def update_display(self):
        """Update the main display"""
        self.ax_main.clear()
        
        # Get current slice data
        current_pet = self.pet_volume[self.current_slice]
        
        # Display PET data as base layer
        if self.show_ct and self.ct_volume is not None:
            ct_slice = self.ct_volume[self.current_slice]
            self.ax_main.imshow(ct_slice, cmap='gray', alpha=0.6)

        # Show PET
        if self.show_pet:
            self.ax_main.imshow(current_pet, cmap='hot', alpha=0.4)
    
    def show(self):
        """Display the interactive viewer"""
        plt.show()

def visualize_pet_data(pet_volume, pet_datasets, rd_dataset=None, rs_dataset=None, structures=None, output_dir="dvh_data_combined", ct_volume=None):
    """Create comprehensive PET visualizations with structure overlay"""
    if pet_volume is None:
        print("No PET data to visualize")
        return
    
    print("\nGenerating PET visualizations...")
    
    # Get basic statistics
    pet_max = np.max(pet_volume)
    pet_min = np.min(pet_volume)
    pet_mean = np.mean(pet_volume)
    pet_std = np.std(pet_volume)
    
    print(f"PET Statistics:")
    print(f"  Min: {pet_min:.2f}")
    print(f"  Max: {pet_max:.2f}")
    print(f"  Mean: {pet_mean:.2f}")
    print(f"  Std: {pet_std:.2f}")
    
    # Get structure contours if available
    structure_contours = {}
    if rs_dataset and structures:
        structure_contours = get_structure_contours(rs_dataset, structures, pet_datasets)
    
    # Create interactive viewer
    if structure_contours:
        print("\nLaunching interactive PET + Structure")
        viewer = InteractivePETStructureViewer(
            pet_volume, ct_volume, structure_contours, 
            sort_pet_slices(pet_datasets)
        )
        viewer.show()

    slice_max_activity = np.max(pet_volume, axis=(1, 2))
    hottest_slice_idx = np.argmax(slice_max_activity)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pet_volume[hottest_slice_idx], cmap='hot')
    plt.title(f'Hottest Slice #{hottest_slice_idx+1}\nMax Activity: {slice_max_activity[hottest_slice_idx]:.2f}')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # Thresholded view (show only high activity regions)
    threshold = pet_mean + 2 * pet_std
    plt.subplot(1, 2, 2)
    thresholded = pet_volume[hottest_slice_idx].copy()
    thresholded[thresholded < threshold] = 0
    plt.imshow(thresholded, cmap='hot')
    plt.title(f'High Activity Regions\n(Threshold: {threshold:.2f})')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    plt.tight_layout()
    hotspot_path = os.path.join(output_dir, "PET_Hotspot_Analysis.png")
    plt.savefig(hotspot_path, dpi=300, bbox_inches='tight')
    print(f"PET hotspot analysis saved to: {hotspot_path}")
    plt.show()
    
    # 5. Slice-by-slice activity profile
    plt.figure(figsize=(12, 6))
    
    slice_mean_activity = np.mean(pet_volume, axis=(1, 2))
    slice_numbers = range(1, len(slice_mean_activity) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(slice_numbers, slice_mean_activity, 'b-o', markersize=4)
    plt.xlabel('Slice Number')
    plt.ylabel('Mean Activity')
    plt.title('Mean Activity per Slice')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(slice_numbers, slice_max_activity, 'r-o', markersize=4)
    plt.xlabel('Slice Number')
    plt.ylabel('Max Activity')
    plt.title('Maximum Activity per Slice')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    profile_path = os.path.join(output_dir, "PET_Activity_Profile.png")
    plt.savefig(profile_path, dpi=300, bbox_inches='tight')
    print(f"PET activity profile saved to: {profile_path}")
    plt.show()
    
    return {
        'volume': pet_volume,
        'statistics': {
            'min': pet_min,
            'max': pet_max,
            'mean': pet_mean,
            'std': pet_std
        },
        'hottest_slice': hottest_slice_idx,
        'slice_activities': {
            'mean': slice_mean_activity,
            'max': slice_max_activity
        }
    }

def resample_pet_to_dose(pet_volume, pet_datasets, dose_origin, dose_spacing, dose_shape):
    """Resample PET volume into dose coordinate space using trilinear interpolation"""
    if pet_volume is None or len(pet_datasets) == 0:
        return None

    # PET metadata
    pet_origin = np.array(pet_datasets[0].ImagePositionPatient, dtype=np.float32)
    pet_spacing = list(map(float, pet_datasets[0].PixelSpacing))  # [dy, dx]
    pet_thickness = float(pet_datasets[0].SliceThickness)
    pet_spacing.append(pet_thickness)  # [dy, dx, dz]

    pet_inv_spacing = np.array([1/s for s in pet_spacing])

    # Create 3D grid of points in dose space
    zi, yi, xi = np.meshgrid(
        np.arange(dose_shape[0]),
        np.arange(dose_shape[1]),
        np.arange(dose_shape[2]),
        indexing='ij'
    )

    # Convert dose voxel indices to world coordinates
    dose_x = xi * dose_spacing[1] + dose_origin[0]
    dose_y = yi * dose_spacing[0] + dose_origin[1]
    dose_z = zi * dose_spacing[2] + dose_origin[2]

    # Convert world coordinates to PET voxel indices
    pet_coords_x = (dose_x - pet_origin[0]) / pet_spacing[1]
    pet_coords_y = (dose_y - pet_origin[1]) / pet_spacing[0]
    pet_coords_z = (dose_z - pet_origin[2]) / pet_spacing[2]

    # Convert world coordinates to PET voxel indices
    pet_coords = np.array([pet_coords_z, pet_coords_y, pet_coords_x])

    # Interpolate PET data at these coordinates
    resampled_pet = map_coordinates(
        pet_volume, 
        pet_coords, 
        order=1,  # linear interpolation
        mode='nearest'
    )

    return resampled_pet

def get_dose_grid_coordinates(rd_dataset):
    origin = np.array(rd_dataset.ImagePositionPatient, dtype=np.float32)  # 3D origin
    spacing = list(map(float, rd_dataset.PixelSpacing))  # [dy, dx]
    slice_thickness = float(rd_dataset.GridFrameOffsetVector[1] - rd_dataset.GridFrameOffsetVector[0]) \
        if hasattr(rd_dataset, 'GridFrameOffsetVector') and len(rd_dataset.GridFrameOffsetVector) > 1 \
        else float(rd_dataset.SliceThickness)

    spacing.append(slice_thickness)  # [dy, dx, dz]

    shape = rd_dataset.pixel_array.shape  # [z, y, x]
    return origin, spacing, shape

def get_ct_grid_coordinates(ct_datasets):
    sorted_ct = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
    origin = np.array(sorted_ct[0].ImagePositionPatient, dtype=np.float32)
    
    spacing = list(map(float, sorted_ct[0].PixelSpacing))  # [dy, dx]
    thickness = float(sorted_ct[0].SliceThickness)
    spacing.append(thickness)  # [dy, dx, dz]

    shape = (len(sorted_ct), sorted_ct[0].Rows, sorted_ct[0].Columns)
    return origin, spacing, shape

def resample_pet_to_ct(pet_datasets, ct_datasets, pet_volume, reg_transform=None):
    # Sort datasets by Z position
    pet_sorted = sorted(pet_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
    ct_sorted = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))

    # --- PET metadata ---
    pet_spacing = list(map(float, pet_sorted[0].PixelSpacing))  # [dy, dx]
    if len(pet_sorted) > 1:
        z1 = float(pet_sorted[0].ImagePositionPatient[2])
        z2 = float(pet_sorted[1].ImagePositionPatient[2])
        dz_pet = abs(z2 - z1)
    else:
        dz_pet = float(pet_sorted[0].SliceThickness)

    pet_spacing.append(dz_pet)
    pet_origin = list(map(float, pet_sorted[0].ImagePositionPatient))

    # Correct orientation using cross product
    iop = pet_sorted[0].ImageOrientationPatient
    axis_x = np.array(iop[:3])
    axis_y = np.array(iop[3:])
    axis_z = np.cross(axis_x, axis_y)
    direction = np.concatenate([axis_x, axis_y, axis_z]).tolist()

    pet_img = sitk.GetImageFromArray(pet_volume.astype(np.float32))  # Z,Y,X
    pet_img.SetSpacing([pet_spacing[1], pet_spacing[0], pet_spacing[2]])  # [dx, dy, dz]
    pet_img.SetOrigin(pet_origin)
    pet_img.SetDirection(direction)

    # --- CT metadata ---
    ct_shape = (ct_sorted[0].Columns, ct_sorted[0].Rows, len(ct_sorted))
    ct_spacing = list(map(float, ct_sorted[0].PixelSpacing))  # [dy, dx]
    if len(ct_sorted) > 1:
        z1_ct = float(ct_sorted[0].ImagePositionPatient[2])
        z2_ct = float(ct_sorted[1].ImagePositionPatient[2])
        dz_ct = abs(z2_ct - z1_ct)
    else:
        dz_ct = float(ct_sorted[0].SliceThickness)

    ct_spacing.append(dz_ct)
    ct_origin = list(map(float, ct_sorted[0].ImagePositionPatient))
    iop_ct = ct_sorted[0].ImageOrientationPatient
    axis_x_ct = np.array(iop_ct[:3])
    axis_y_ct = np.array(iop_ct[3:])
    axis_z_ct = np.cross(axis_x_ct, axis_y_ct)
    ct_direction = np.concatenate([axis_x_ct, axis_y_ct, axis_z_ct]).tolist()
    
    print("\nPET origin Z:", pet_origin[2])
    print("CT origin Z:", ct_origin[2])
    print("PET dz:", pet_spacing[2])
    print("CT dz:", ct_spacing[2])

    ct_ref = sitk.Image(ct_shape, sitk.sitkFloat32)
    ct_ref.SetSpacing([ct_spacing[1], ct_spacing[0], ct_spacing[2]])
    ct_ref.SetOrigin(ct_origin)
    ct_ref.SetDirection(ct_direction)

    # --- Resample PET to CT grid ---
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_ref)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    if reg_transform is not None:
        resampler.SetTransform(reg_transform)
    else:
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampled_pet = resampler.Execute(pet_img)

    return sitk.GetArrayFromImage(resampled_pet)

def export_cumulative_pet_histogram(pet_volume, structure_masks, structures, voxel_volume_ml, output_dir):    
    # Get PET resolution with safety limits
    pet_step, max_pet = get_pet_resolution(pet_volume)
    
    # Additional safety check for number of bins
    estimated_bins = int(max_pet / pet_step) + 1
    if estimated_bins > 10000:
        print(f"Warning: Too many bins ({estimated_bins}), adjusting step size")
        pet_step = max_pet / 10000.0
        estimated_bins = 10000
    
    # Create bins using consistent logic with DVH
    custom_bins = np.arange(0, max_pet + pet_step, pet_step)
    
    print(f"PET histogram parameters:")
    print(f"  Step size: {pet_step}")
    print(f"  Max value: {max_pet}")
    print(f"  Number of bins: {len(custom_bins)}")
    
    # Create PVH tables - both absolute and relative
    pvh_table_absolute = {"PET_Bq_per_mL": [float(p) for p in custom_bins.tolist()]}
    pvh_table_relative = {"PET_Bq_per_mL": [float(p) for p in custom_bins.tolist()]}
    
    for sid, mask in structure_masks.items():
        if sid not in structures:
            continue
            
        # Get PET values within the structure
        structure_pet_values = pet_volume[mask]
        
        if structure_pet_values.size == 0:
            print(f"Warning: No voxels found for structure {structures[sid]['name']}")
            continue
        
        # Calculate total structure volume
        total_voxels = structure_pet_values.size
        total_volume_ml = total_voxels * voxel_volume_ml
        
        # Calculate cumulative volumes at each PET level (volume with >= activity)
        cumulative_volumes = []
        relative_volumes = []
        
        for pet_level in custom_bins:
            # Count voxels with PET activity >= pet_level
            voxel_count = np.sum(structure_pet_values >= pet_level)
            volume_ml = voxel_count * voxel_volume_ml
            cumulative_volumes.append(volume_ml)
            
            # Calculate relative volume (fraction of total structure)
            if total_volume_ml > 0:
                relative_volume = volume_ml / total_volume_ml
            else:
                relative_volume = 0.0
            relative_volumes.append(relative_volume)
        
        # Create safe column names
        name = structures[sid]['name']
        safe_name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
        abs_column_name = f"{safe_name}_Volume_mL"
        rel_column_name = f"{safe_name}_RelativeCumVolume"
        
        pvh_table_absolute[abs_column_name] = cumulative_volumes
        pvh_table_relative[rel_column_name] = relative_volumes
        
        print(f"Structure '{name}':")
        print(f"  Total voxels: {structure_pet_values.size}")
        print(f"  Total volume: {total_volume_ml:.2f} mL")
        print(f"  Volume at 0 Bq/mL: {cumulative_volumes[0]:.2f} mL (relative: {relative_volumes[0]:.3f})")
        print(f"  Volume at max activity: {cumulative_volumes[-1]:.2f} mL (relative: {relative_volumes[-1]:.3f})")
    
    # Write Absolute PVH CSV
    absolute_path = os.path.join(output_dir, "CumulativePVH_AllStructures_AbsoluteUnits.csv")
    with open(absolute_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with units information
        writer.writerow(["# Cumulative PET Volume Histogram (PVH) - PET in Bq/mL, Volume in mL"])
        writer.writerow(["# Values represent volume of structure with >= PET activity"])
        writer.writerow(["# Generated from PET and RT Structure data"])
        writer.writerow([])  # Empty row for separation
        
        # Write column headers
        headers = list(pvh_table_absolute.keys())
        writer.writerow(headers)
        
        # Write data rows with appropriate precision
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "PET_Bq_per_mL":
                    row.append("{:.6f}".format(pvh_table_absolute[h][i]))
                else:
                    row.append("{:.4f}".format(pvh_table_absolute[h][i]))
            writer.writerow(row)
    
    # Write Relative PVH CSV
    relative_path = os.path.join(output_dir, "CumulativePVH_AllStructures_RelativeUnits.csv")
    with open(relative_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header with units information
        writer.writerow(["# Cumulative PET Volume Histogram (PVH) - PET in Bq/mL, Relative Cumulative Volume (fraction 0-1)"])
        writer.writerow(["# Values represent fraction of structure volume with >= PET activity"])
        writer.writerow(["# Generated from PET and RT Structure data"])
        writer.writerow([])  # Empty row for separation
        
        # Write column headers
        headers = list(pvh_table_relative.keys())
        writer.writerow(headers)
        
        # Write data rows with appropriate precision
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "PET_Bq_per_mL":
                    row.append("{:.6f}".format(pvh_table_relative[h][i]))
                else:
                    row.append("{:.6f}".format(pvh_table_relative[h][i]))
            writer.writerow(row)
    
    print(f"\nAbsolute PET Volume Histogram exported to: {absolute_path}")
    print(f"Relative PET Volume Histogram exported to: {relative_path}")
    print(f"Data exported with:")
    print(f"   - PET activity in Bq/mL")
    print(f"   - {len(custom_bins)} activity levels")
    print(f"   - {len([k for k in pvh_table_absolute.keys() if k != 'PET_Bq_per_mL'])} structures")
    
    return pvh_table_absolute, pvh_table_relative

def create_structure_masks(structures, rs_dataset, ct_datasets, volume_shape):
    rt = dicomparser.DicomParser(rs_dataset)
    first_ct = ct_datasets[0]
    origin = np.array(first_ct.ImagePositionPatient)
    spacing = list(map(float, first_ct.PixelSpacing))
    thickness = float(first_ct.SliceThickness)
    spacing.append(thickness)

    z_positions = sorted([float(ds.ImagePositionPatient[2]) for ds in ct_datasets])
    masks = {}

    for sid, struct in structures.items():
        coords = rt.GetStructureCoordinates(sid)
        if not coords:
            continue

        mask = np.zeros(volume_shape, dtype=bool)
        for z_str, contours in coords.items():
            z = float(z_str)
            try:
                k = np.argmin(np.abs(np.array(z_positions) - z))
            except:
                continue

            for contour in contours:
                pts = contour['data']
                if len(pts) < 3:
                    continue
                x = [(pt[0] - origin[0]) / spacing[0] for pt in pts]
                y = [(pt[1] - origin[1]) / spacing[1] for pt in pts]
                poly = Path(np.vstack((x, y)).T)
                grid_x, grid_y = np.meshgrid(np.arange(volume_shape[2]), np.arange(volume_shape[1]))
                points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
                mask2d = poly.contains_points(points).reshape((volume_shape[1], volume_shape[2]))
                mask[k] |= mask2d

        masks[sid] = mask
    return masks

def extract_affine_transform_from_reg(reg_dataset):
    try:
        matrix_seq = reg_dataset[0x0070, 0x0308].value[0]  # RegistrationSequence -> Item 0
        matrix_data = matrix_seq[0x3006, 0x00C6].value      # FrameOfReferenceTransformationMatrix

        if len(matrix_data) != 16:
            print("Expected 16 values for 4x4 matrix, got:", len(matrix_data))
            return None

        # Convert 4x4 to rotation (3x3) + translation (3)
        matrix_np = np.array(matrix_data).reshape(4, 4)
        rotation = matrix_np[:3, :3].flatten().tolist()
        translation = matrix_np[:3, 3].tolist()

        transform = sitk.AffineTransform(3)
        transform.SetMatrix(rotation)
        transform.SetTranslation(translation)

        print("REG matrix extracted and converted to SimpleITK transform.")
        return transform

    except Exception as e:
        print(f"Failed to extract transform from REG dataset: {e}")
        return None
