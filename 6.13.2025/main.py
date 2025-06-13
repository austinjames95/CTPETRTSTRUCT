from dicompylercore import dicomparser, dvhcalc
import pydicom
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
from viewer import visualize_pet_data, create_pet_volume, get_ct_grid_coordinates, resample_pet_to_ct, create_structure_masks, extract_affine_transform_from_reg, get_pet_resolution
from ddvh2cdvh import compare_relative_dvh, compare_relative_pvh

def get_dose_resolution(rd_dataset):
    dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling
    unique_doses = np.unique(dose_array.flatten())
    dose_diffs = np.diff(np.sort(unique_doses))
    min_step = np.min(dose_diffs[dose_diffs > 0])
    return min_step, np.max(unique_doses)

base_dir = r'C:\Users\austi\OneDrive\Desktop\COH\BGRT^P_uPTV5mm4cmSphere_20250110_160d82'

ct_datasets = []
pet_datasets = []
rs_dataset = None
rd_dataset = None
reg_dataset = None

print("\nscanning files\n")

dicom_files = glob.glob(os.path.join(base_dir, '*'))
for file_path in dicom_files:
    try:
        ds = pydicom.dcmread(file_path)
        modality = ds.Modality
        print(f"{os.path.basename(file_path)}: Modality = {modality}")

        if modality == 'CT':
            ct_datasets.append(ds)
        elif modality == 'PT':
            pet_datasets.append(ds)
        elif modality == 'RTSTRUCT':
            rs_dataset = ds
        elif modality == 'RTDOSE':
            rd_dataset = ds
        elif modality == 'REG':
            reg_dataset = ds

    except Exception as e:
        print(f"Could not read {file_path}: {e}")

print("\n Load Summary \n")
print(f"Loaded {len(ct_datasets)} CT slices")
print(f"Loaded {len(pet_datasets)} PET slices")
print("\nRTSTRUCT loaded - PASS - Ready" if rs_dataset else "\nRTSTRUCT loaded - FAIL")
print("\nRTDOSE loaded - PASS - Ready" if rd_dataset else "\nRTDOSE loaded - FAIL")
print("\nREG loaded - PASS - Ready" if reg_dataset else "\nREG loaded - FAIL (will use DICOM coordinates only)")

sorted_ct = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
ct_volume = np.stack([ds.pixel_array.astype(np.float32) for ds in sorted_ct])

os.makedirs("generated_data", exist_ok=True)
os.makedirs("extra_data", exist_ok=True)
os.makedirs("generated_data/DVH", exist_ok=True)
os.makedirs("generated_data/PVH", exist_ok=True)
os.makedirs("extra_data/DVH", exist_ok=True)
os.makedirs("extra_data/PVH", exist_ok=True)


if rs_dataset and rd_dataset:
    print("\n Parsing DICOM datasets with dicompylercore:")

    try:
        dp_struct = dicomparser.DicomParser(rs_dataset)
        dp_dose = dicomparser.DicomParser(rd_dataset)
    except Exception as e:
        print("Failed to parse RTSTRUCT or RTDOSE:", e)
        exit(1)

    try:
        structures = dp_struct.GetStructures()
    except Exception as e:
        print("Failed to get structures from RTSTRUCT:", e)
        exit(1)

    print("\nAvailable Structures:")
    for sid, struct in structures.items():
        print(f"  [{sid}] {struct['name']}")
    
        
    dose_step, max_dose = get_dose_resolution(rd_dataset)
    custom_bins = np.arange(0, max_dose + dose_step, dose_step)
    
    dose_units = getattr(rd_dataset, 'DoseUnits', 'GY')
    print(f"\nDose units from DICOM: {dose_units}")
    
    # DVH table - dose and absolute volumes
    dvh_table_absolute = {"GY": [float(d) for d in custom_bins.tolist()]}
    dvh_table_relative = {"GY": [float(d) for d in custom_bins.tolist()]}

    for sid, struct in structures.items():
        try:
            dvh_struct = dvhcalc.get_dvh(dp_struct.ds, dp_dose.ds, sid, calculate_full_volume=True)
            
            if not dvh_struct:
                print(f"Warning: No DVH for {struct['name']}")
                continue

            structure_volume = dvh_struct.volume
            print(f"Structure '{struct['name']}' total volume: {structure_volume:.2f} cm3")

            # Get absolute volumes at each dose level (cumulative - volume receiving >= dose)
            absolute_volumes = []
            for dose in custom_bins:
                vol = dvh_struct.volume_constraint(dose, dose_units="Gy")
                
                # Extract the numerical value
                if hasattr(vol, 'value'):
                    volume_value = float(vol.value)
                else:
                    volume_value = float(vol)
                
                absolute_volumes.append(volume_value)

            # Convert to relative volumes (fraction of total structure volume)
            relative_volumes = []
            if structure_volume > 0:
                relative_volumes = [vol / structure_volume for vol in absolute_volumes]
            else:
                relative_volumes = [0.0] * len(custom_bins)

            # Create safe column names
            safe_name = struct['name'].replace(" ", "_").replace("/", "_").replace("-", "_")
            abs_column_name = f"{safe_name}_Volume_cm3"
            rel_column_name = f"{safe_name}_RelativeCumVolume"
            
            dvh_table_absolute[abs_column_name] = absolute_volumes
            dvh_table_relative[rel_column_name] = relative_volumes

            print(f"  DVH computed: {len(absolute_volumes)} dose points")
            print(f"  Volume at 0 Gy: {absolute_volumes[0]:.2f} cm3 (relative: {relative_volumes[0]:.3f})")
            print(f"  Volume at max dose: {absolute_volumes[-1]:.2f} cm3 (relative: {relative_volumes[-1]:.3f})")

        except Exception as e:
            print(f"Error computing DVH for {struct['name']}: {e}")

    # Write Absolute DVH CSV
    absolute_path = os.path.join("extra_data/DVH", "CumulativeDVH_AllStructures_AbsoluteUnits.csv")
    with open(absolute_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write column headers
        headers = list(dvh_table_absolute.keys())
        writer.writerow(headers)
        
        # Write data rows with appropriate precision
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "GY":
                    row.append("{:.6f}".format(dvh_table_absolute[h][i]))
                else:
                    row.append("{:.4f}".format(dvh_table_absolute[h][i]))
            writer.writerow(row)

    print(f"\nAbsolute DVH written to: {absolute_path}")

    # Write Relative DVH CSV
    relative_path_dvh = os.path.join("generated_data/DVH", "CumulativeDVH_AllStructures_RelativeUnits.csv")
    with open(relative_path_dvh, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write column headers
        headers = list(dvh_table_relative.keys())
        writer.writerow(headers)
        
        # Write data rows with appropriate precision
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "GY":
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
                else:
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
            writer.writerow(row)

    print(f"Relative DVH written to: {relative_path_dvh}")
    print(f"Data exported with:")
    print(f"   - Dose in Gray (Gy)")
    print(f"   - {len(custom_bins)} dose points")
    print(f"   - {len([k for k in dvh_table_absolute.keys() if k != 'GY'])} structures")

    print("\nGenerating DVH plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    num_structures = len([k for k in dvh_table_absolute.keys() if k != 'GY'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    dose_values = dvh_table_absolute["GY"]

    structure_index = 0
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "GY":
            continue
            
        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        
        ax1.plot(dose_values, volumes, 
                label=clean_name, 
                linewidth=2, 
                color=colors[structure_index % len(colors)])
        structure_index += 1
    
    ax1.set_xlabel('Dose (Gy)', fontsize=12)
    ax1.set_ylabel('Volume (cm³)', fontsize=12)
    ax1.set_title('Cumulative DVH - All Structures (Absolute)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    critical_keywords = ['PTV', 'GTV', 'CTV', 'ORGAN', 'BLADDER', 'RECTUM', 'FEMUR', 'BOWEL']
    
    structure_index = 0
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "GY":
            continue
            
        is_critical = any(keyword.upper() in struct_name.upper() for keyword in critical_keywords)
        is_external = any(ext.upper() in struct_name.upper() for ext in ['EXTERNAL', 'BODY', 'SKIN'])
        
        if is_critical and not is_external:
            clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
            ax2.plot(dose_values, volumes, 
                    label=clean_name, 
                    linewidth=2, 
                    color=colors[structure_index % len(colors)])
        structure_index += 1
    
    ax2.set_xlabel('Dose (Gy)', fontsize=12)
    ax2.set_ylabel('Volume (cm³)', fontsize=12)
    ax2.set_title('Cumulative DVH - Critical Structures (Absolute)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    abs_plot_path = os.path.join("extra_data/DVH", "DVH_Absolute_Plot.png")
    plt.savefig(abs_plot_path, dpi=300, bbox_inches='tight')
    print(f" Absolute DVH plot saved to: {abs_plot_path}")
    plt.show()

    # Create relative DVH plot
    plt.figure(figsize=(12, 8))
    structure_index = 0
    for struct_name, volumes in dvh_table_relative.items():
        if struct_name == "GY":
            continue
        
        clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
        plt.plot(dvh_table_relative["GY"], volumes, 
                label=clean_name, 
                linewidth=2,
                color=colors[structure_index % len(colors)])
        structure_index += 1

    plt.xlabel('Dose (Gy)', fontsize=12)
    plt.ylabel('Relative Volume (fraction)', fontsize=12)
    plt.title('Cumulative DVH - Relative Volumes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save relative DVH plot
    rel_plot_path = os.path.join("generated_data/DVH", "DVH_Relative_Plot.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    print(f" Relative DVH plot saved to: {rel_plot_path}")
    plt.show()
    
    print("\nDVH Summary Statistics:")
    print("="*60)
    
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "GY":
            continue
            
        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        max_volume = volumes[0] 

        rel_struct_name = struct_name.replace("_Volume_cm3", "")
        if rel_struct_name in dvh_table_relative:
            rel_volumes = dvh_table_relative[rel_struct_name]
            
            dose_95_vol = None
            dose_50_vol = None
            dose_5_vol = None
            
            for j, rel_vol in enumerate(rel_volumes):
                if dose_95_vol is None and rel_vol <= 0.95:
                    dose_95_vol = dose_values[j]
                if dose_50_vol is None and rel_vol <= 0.50:
                    dose_50_vol = dose_values[j]
                if dose_5_vol is None and rel_vol <= 0.05:
                    dose_5_vol = dose_values[j]
        
        print(f"\n{clean_name}:")
        print(f"  Total Volume: {max_volume:.2f} cm³")
        if 'dose_95_vol' in locals() and dose_95_vol is not None:
            print(f"  D95 (dose to 95% volume): {dose_95_vol:.2f} Gy")
        if 'dose_50_vol' in locals() and dose_50_vol is not None:
            print(f"  D50 (dose to 50% volume): {dose_50_vol:.2f} Gy")
        if 'dose_5_vol' in locals() and dose_5_vol is not None:
            print(f"  D5 (dose to 5% volume): {dose_5_vol:.2f} Gy")
    
    print("\n" + "="*60)
    print("\nProcessing completed successfully!")
else:
    print("Required DICOM files (RTSTRUCT and RTDOSE) not found or could not be loaded.")

if pet_datasets:
    print(f"\nProcessing {len(pet_datasets)} PET slices...")
    
    if rd_dataset:
        print("Aligning PET volume to CT grid...")

        pet_volume, sorted_pet = create_pet_volume(pet_datasets)

        if pet_volume is not None:
            print(f"Original PET shape: {pet_volume.shape}")

            ct_origin, ct_spacing, ct_shape = get_ct_grid_coordinates(ct_datasets)
            reg_transform = None
            if reg_dataset:
                reg_transform = extract_affine_transform_from_reg(reg_dataset)

            pet_volume = resample_pet_to_ct(pet_datasets, ct_datasets, pet_volume, reg_transform=reg_transform)

            print(f"Resampled PET shape (CT grid): {pet_volume.shape}")
    else:
        pet_volume, sorted_pet = create_pet_volume(pet_datasets)
    
    # Replace the existing PVH export section in main.py with this enhanced version:

    if pet_volume is not None:
        structures = None
        if rs_dataset:
            try:
                dp_struct = dicomparser.DicomParser(rs_dataset)
                structures = dp_struct.GetStructures()
            except Exception as e:
                print(f"Could not get structures for overlay: {e}")
        
        pet_analysis = visualize_pet_data(
            pet_volume,
            pet_datasets,
            rd_dataset=rd_dataset,
            rs_dataset=rs_dataset,
            structures=structures,
            ct_volume=ct_volume
        )

        # Generate structure masks and export PVH
        structure_masks = create_structure_masks(structures, rs_dataset, ct_datasets, pet_volume.shape)
        spacing = list(map(float, ct_datasets[0].PixelSpacing))  # [dy, dx]
        thickness = float(ct_datasets[0].SliceThickness)
        voxel_volume_mm3 = spacing[0] * spacing[1] * thickness
        voxel_volume_ml = voxel_volume_mm3 / 1000.0

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
        pvh_table_absolute = {"BQML": [float(p) for p in custom_bins.tolist()]}
        pvh_table_relative = {"BQML": [float(p) for p in custom_bins.tolist()]}
        
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
        absolute_path = os.path.join("extra_data/PVH", "CumulativePVH_AllStructures_AbsoluteUnits.csv")
        with open(absolute_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write column headers
            headers = list(pvh_table_absolute.keys())
            writer.writerow(headers)
            
            # Write data rows with appropriate precision
            for i in range(len(custom_bins)):
                row = []
                for h in headers:
                    if h == "BQML":
                        row.append("{:.6f}".format(pvh_table_absolute[h][i]))
                    else:
                        row.append("{:.4f}".format(pvh_table_absolute[h][i]))
                writer.writerow(row)
        
        # Write Relative PVH CSV
        relative_path = os.path.join("generated_data/PVH", "CumulativePVH_AllStructures_RelativeUnits.csv")
        with open(relative_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write column headers
            headers = list(pvh_table_relative.keys())
            writer.writerow(headers)
            
            # Write data rows with appropriate precision
            for i in range(len(custom_bins)):
                row = []
                for h in headers:
                    if h == "BQML":
                        row.append("{:.6f}".format(pvh_table_relative[h][i]))
                    else:
                        row.append("{:.6f}".format(pvh_table_relative[h][i]))
                writer.writerow(row)
        
        print(f"\nAbsolute PET Volume Histogram exported to: {absolute_path}")
        print(f"Relative PET Volume Histogram exported to: {relative_path}")
        print(f"Data exported with:")
        print(f"   - PET activity in Bq/mL")
        print(f"   - {len(custom_bins)} activity levels")
        print(f"   - {len([k for k in pvh_table_absolute.keys() if k != 'BQML'])} structures")
    
        print("\nGenerating PVH plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        num_structures = len([k for k in pvh_table_absolute.keys() if k != 'BQML'])
        if num_structures <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

        pet_values = pvh_table_absolute["BQML"]
        structure_index = 0
        for struct_name, volumes in pvh_table_absolute.items():
            if struct_name == "BQML":
                continue
                
            clean_name = struct_name.replace("_Volume_mL", "").replace("_", " ")
            
            ax1.plot(pet_values, volumes, 
                    label=clean_name, 
                    linewidth=2, 
                    color=colors[structure_index % len(colors)])
            structure_index += 1
        
        ax1.set_xlabel('PET Activity (Bq/mL)', fontsize=12)
        ax1.set_ylabel('Volume (mL)', fontsize=12)
        ax1.set_title('Cumulative PVH - All Structures (Absolute)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot critical structures only (similar to DVH approach)
        critical_keywords = ['PTV', 'GTV', 'CTV', 'ORGAN', 'BLADDER', 'RECTUM', 'FEMUR', 'BOWEL']
        
        structure_index = 0
        for struct_name, volumes in pvh_table_absolute.items():
            if struct_name == "BQML":
                continue
                
            is_critical = any(keyword.upper() in struct_name.upper() for keyword in critical_keywords)
            is_external = any(ext.upper() in struct_name.upper() for ext in ['EXTERNAL', 'BODY', 'SKIN'])
            
            if is_critical and not is_external:
                clean_name = struct_name.replace("_Volume_mL", "").replace("_", " ")
                ax2.plot(pet_values, volumes, 
                        label=clean_name, 
                        linewidth=2, 
                        color=colors[structure_index % len(colors)])
            structure_index += 1
        
        ax2.set_xlabel('PET Activity (Bq/mL)', fontsize=12)
        ax2.set_ylabel('Volume (mL)', fontsize=12)
        ax2.set_title('Cumulative PVH - Critical Structures (Absolute)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()

        abs_plot_path = os.path.join("extra_data/PVH", "PVH_Absolute_Plot.png")
        plt.savefig(abs_plot_path, dpi=300, bbox_inches='tight')
        print(f"Absolute PVH plot saved to: {abs_plot_path}")
        plt.show()

        # Create relative PVH plot
        plt.figure(figsize=(12, 8))
        structure_index = 0
        for struct_name, volumes in pvh_table_relative.items():
            if struct_name == "BQML":
                continue
            
            clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
            plt.plot(pvh_table_relative["BQML"], volumes, 
                    label=clean_name, 
                    linewidth=2,
                    color=colors[structure_index % len(colors)])
            structure_index += 1

        plt.xlabel('PET Activity (Bq/mL)', fontsize=12)
        plt.ylabel('Relative Volume (fraction)', fontsize=12)
        plt.title('Cumulative PVH - Relative Volumes', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save relative PVH plot
        rel_plot_path = os.path.join("generated_data/PVH", "PVH_Relative_Plot.png")
        plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
        print(f"Relative PVH plot saved to: {rel_plot_path}")
        plt.show()
        
        # Generate PVH Summary Statistics (similar to DVH summary)
        print("\nPVH Summary Statistics:")
        print("="*60)
        
        for struct_name, volumes in pvh_table_absolute.items():
            if struct_name == "BQML":
                continue
                
            clean_name = struct_name.replace("_Volume_mL", "").replace("_", " ")
            max_volume = volumes[0]  # Volume at 0 Bq/mL (total structure volume)

            rel_struct_name = struct_name.replace("_Volume_mL", "")
            if rel_struct_name in pvh_table_relative:
                rel_volumes = pvh_table_relative[rel_struct_name]
                
                # Find activity levels for specific volume fractions
                activity_95_vol = None
                activity_50_vol = None
                activity_5_vol = None
                
                for j, rel_vol in enumerate(rel_volumes):
                    if activity_95_vol is None and rel_vol <= 0.95:
                        activity_95_vol = pet_values[j]
                    if activity_50_vol is None and rel_vol <= 0.50:
                        activity_50_vol = pet_values[j]
                    if activity_5_vol is None and rel_vol <= 0.05:
                        activity_5_vol = pet_values[j]
            
            print(f"\n{clean_name}:")
            print(f"  Total Volume: {max_volume:.2f} mL")
            if 'activity_95_vol' in locals() and activity_95_vol is not None:
                print(f"  A95 (activity for 95% volume): {activity_95_vol:.2f} Bq/mL")
            if 'activity_50_vol' in locals() and activity_50_vol is not None:
                print(f"  A50 (activity for 50% volume): {activity_50_vol:.2f} Bq/mL")
            if 'activity_5_vol' in locals() and activity_5_vol is not None:
                print(f"  A5 (activity for 5% volume): {activity_5_vol:.2f} Bq/mL")
        
        print("\n" + "="*60)
        print("PVH processing completed successfully!")

    else:
        print("Failed to create PET volume")
else:
    print("\nNo PET data found - skipping PET visualization")

compare_relative_dvh()
compare_relative_pvh()

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)
