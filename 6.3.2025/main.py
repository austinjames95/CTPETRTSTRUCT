from dicompylercore import dicomparser, dvhcalc
import pydicom
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
from submain import load_pet_with_transforms, sample_pet_at_structures_with_transforms, get_dose_resolution, normalize_name

base_dir = r'C:\\Users\\austi\\OneDrive\\Desktop\\COH\\BGRT^P_uPTV5mm4cmSphere_20250110_160d82'

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
    
    # Load PET and get statistics
    pet_image, reg_transform = load_pet_with_transforms(pet_datasets, reg_dataset)
    pet_stats = sample_pet_at_structures_with_transforms(pet_image, structures, dp_struct, reg_transform)
        
    dose_step, max_dose = get_dose_resolution(rd_dataset)
    custom_bins = np.arange(0, max_dose + dose_step, dose_step)
    
    dose_units = getattr(rd_dataset, 'DoseUnits', 'GY')
    print(f"\nDose units from DICOM: {dose_units}")
    
    # Create output directory FIRST
    os.makedirs("dvh_data_combined", exist_ok=True)
    
    # DVH table - dose and absolute volumes
    dvh_table_absolute = {"Dose_Gy": [float(d) for d in custom_bins.tolist()]}
    dvh_table_relative = {"Dose_Gy": [float(d) for d in custom_bins.tolist()]}

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
    absolute_path = os.path.join("dvh_data_combined", "CumulativeDVH_AllStructures_AbsoluteUnits.csv")
    with open(absolute_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header with units information
        writer.writerow(["# Cumulative DVH Data - Dose in Gy, Volume in cm3"])
        writer.writerow(["# Values represent volume of structure receiving >= dose"])
        writer.writerow(["# Generated from DICOM RT data"])
        writer.writerow([])  # Empty row for separation
        
        # Write column headers
        headers = list(dvh_table_absolute.keys())
        writer.writerow(headers)
        
        # Write data rows with appropriate precision
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "Dose_Gy":
                    row.append("{:.6f}".format(dvh_table_absolute[h][i]))
                else:
                    row.append("{:.4f}".format(dvh_table_absolute[h][i]))
            writer.writerow(row)

    print(f"\nAbsolute DVH written to: {absolute_path}")

    # Write Relative DVH CSV
    relative_path = os.path.join("dvh_data_combined", "CumulativeDVH_AllStructures_RelativeUnits.csv")
    with open(relative_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header with units information
        writer.writerow(["# Cumulative DVH Data - Dose in Gy, Relative Cumulative Volume (fraction 0-1)"])
        writer.writerow(["# Values represent fraction of structure volume receiving >= dose"])
        writer.writerow(["# Generated from DICOM RT data"])
        writer.writerow([])  # Empty row for separation
        
        # Write column headers
        headers = list(dvh_table_relative.keys())
        writer.writerow(headers)
        
        # Write data rows with appropriate precision
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "Dose_Gy":
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
                else:
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
            writer.writerow(row)

    print(f"Relative DVH written to: {relative_path}")
    print(f"Data exported with:")
    print(f"   - Dose in Gray (Gy)")
    print(f"   - {len(custom_bins)} dose points")
    print(f"   - {len([k for k in dvh_table_absolute.keys() if k != 'Dose_Gy'])} structures")

    # Write separate PET CSV file
    if pet_stats:
        print(f"\nPET statistics found for {len(pet_stats)} structures")
        pet_csv_path = os.path.join("dvh_data_combined", "PET_Structure_Statistics.csv")
        
        with open(pet_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header with clear description
            writer.writerow(["# PET Activity Statistics by Structure"])
            writer.writerow(["# Units: Bq/ml (Becquerels per milliliter)"])
            writer.writerow([])  # Empty row for separation
            
            # Column headers
            writer.writerow([
                "Structure_Name", 
                "Mean_Bq_ml", 
                "Max_Bq_ml", 
                "Min_Bq_ml",
                "Std_Bq_ml", 
                "Median_Bq_ml", 
                "N_Sample_Points"
            ])
            
            # Data rows
            for struct_name, stats in pet_stats.items():
                writer.writerow([
                    struct_name,
                    f"{stats['mean_bqml']:.4f}",
                    f"{stats['max_bqml']:.4f}",
                    f"{stats['min_bqml']:.4f}",
                    f"{stats['std_bqml']:.4f}",
                    f"{stats['median_bqml']:.4f}",
                    stats['n_samples']
                ])

        print(f"PET statistics CSV written to: {pet_csv_path}")
        print(f"PET data includes statistics for {len(pet_stats)} structures")
    else:
        print("\nNo PET statistics available - check PET data loading and structure sampling")

    print("\nGenerating DVH plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    num_structures = len([k for k in dvh_table_absolute.keys() if k != 'Dose_Gy'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    dose_values = dvh_table_absolute["Dose_Gy"]

    structure_index = 0
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "Dose_Gy":
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
        if struct_name == "Dose_Gy":
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

    abs_plot_path = os.path.join("dvh_data_combined", "DVH_Absolute_Plot.png")
    plt.savefig(abs_plot_path, dpi=300, bbox_inches='tight')
    print(f" Absolute DVH plot saved to: {abs_plot_path}")
    plt.show()

    # Create relative DVH plot
    plt.figure(figsize=(12, 8))
    structure_index = 0
    for struct_name, volumes in dvh_table_relative.items():
        if struct_name == "Dose_Gy":
            continue
        
        clean_name = struct_name.replace("_RelativeCumVolume", "").replace("_", " ")
        plt.plot(dvh_table_relative["Dose_Gy"], volumes, 
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
    rel_plot_path = os.path.join("dvh_data_combined", "DVH_Relative_Plot.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    print(f" Relative DVH plot saved to: {rel_plot_path}")
    plt.show()
    
    print("\nDVH Summary Statistics:")
    print("="*60)
    
    for struct_name, volumes in dvh_table_absolute.items():
        if struct_name == "Dose_Gy":
            continue
            
        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        max_volume = volumes[0] 

        rel_struct_name = struct_name.replace("_Volume_cm3", "_RelativeCumVolume")
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
        
        if pet_stats:
            pet_stats_normalized = {normalize_name(k): v for k, v in pet_stats.items()}
            norm_clean_name = normalize_name(clean_name)
            if norm_clean_name in pet_stats_normalized:
                pet_data = pet_stats_normalized[norm_clean_name]
                print(f"  PET Mean Activity: {pet_data['mean_bqml']:.2f} Bq/ml")
                print(f"  PET Max Activity: {pet_data['max_bqml']:.2f} Bq/ml")
                print(f"  PET Sample Points: {pet_data['n_samples']}")
    
    print("\n" + "="*60)
    print("\nProcessing completed successfully!")
else:
    print("Required DICOM files (RTSTRUCT and RTDOSE) not found or could not be loaded.")
