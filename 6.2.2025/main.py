from dicompylercore import dicomparser, dvhcalc
import pydicom
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import csv

def get_dose_resolution(rd_dataset):
    dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling
    unique_doses = np.unique(dose_array.flatten())
    dose_diffs = np.diff(np.sort(unique_doses))
    min_step = np.min(dose_diffs[dose_diffs > 0])
    return min_step, np.max(unique_doses)

base_dir = r'C:\\Users\\austi\\OneDrive\\Desktop\\COH\\BGRT^P_uPTV5mm4cmSphere_20250110_160d82'

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
            ct_datasets.append(ds)
        elif modality == 'PT':
            pet_datasets.append(ds)
        elif modality == 'RTSTRUCT':
            rs_dataset = ds
        elif modality == 'RTDOSE':
            rd_dataset = ds

    except Exception as e:
        print(f"Could not read {file_path}: {e}")

print("\n Load Summary \n")
print(f"Loaded {len(ct_datasets)} CT slices")
print(f"Loaded {len(pet_datasets)} PET slices")
print("\nRTSTRUCT loaded - PASS - Ready" if rs_dataset else "\nRTSTRUCT loaded - FAIL")
print("\nRTDOSE loaded - PASS - Ready" if rd_dataset else "\nRTDOSE loaded - FAIL")

if rs_dataset and rd_dataset:
    print("\n🔍 Parsing DICOM datasets with dicompylercore:")

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
        
    # Get dose resolution and create bins in Gy
    dose_step, max_dose = get_dose_resolution(rd_dataset)
    custom_bins = np.arange(0, max_dose + dose_step, dose_step)
    
    # Verify dose units from DICOM
    dose_units = getattr(rd_dataset, 'DoseUnits', 'GY')
    print(f"\nDose units from DICOM: {dose_units}")
    
    dvh_table = {"Dose_Gy": [float(d) for d in custom_bins.tolist()]}

    for sid, struct in structures.items():
        try:
            dvh_struct = dvhcalc.get_dvh(dp_struct.ds, dp_dose.ds, sid, calculate_full_volume=True)
            if not dvh_struct:
                print(f"Warning: No DVH for {struct['name']}")
                continue

            structure_volume = dvh_struct.volume
            print(f"Structure '{struct['name']}' total volume: {structure_volume:.2f} cm³")

            volumes_struct = []
            for dose in custom_bins:
                vol = dvh_struct.volume_constraint(dose, dose_units="Gy")
                
                if hasattr(vol, 'value'):
                    volume_value = float(vol.value)
                else:
                    volume_value = float(vol)
                
                volumes_struct.append(volume_value)

            safe_name = struct['name'].replace(" ", "_").replace("/", "_").replace("-", "_")
            column_name = f"{safe_name}_Volume_cm3"
            dvh_table[column_name] = volumes_struct

        except Exception as e:
            print(f"Error computing DVH for {struct['name']}: {e}")

    os.makedirs("dvh_data_combined", exist_ok=True)
    combined_path = os.path.join("dvh_data_combined", "CumulativeDVH_AllStructures_AbsoluteUnits.csv")

    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        writer.writerow(["# DVH Data - Dose in Gy, Volume in cm³"])
        writer.writerow(["# Generated from DICOM RT data"])
        writer.writerow([])  
        
        headers = list(dvh_table.keys())
        writer.writerow(headers)
        
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "Dose_Gy":
                    row.append("{:.6f}".format(dvh_table[h][i]))
                else:
                    row.append("{:.4f}".format(dvh_table[h][i]))
            writer.writerow(row)

    print(f"\nCombined DVH written to: {combined_path}")
    print(f"Data exported with:")
    print(f"   - Dose in Gray (Gy)")
    print(f"   - Volume in cubic centimeters (cm³)")
    print(f"   - {len(custom_bins)} dose points")
    print(f"   - {len([k for k in dvh_table.keys() if k != 'Dose_Gy'])} structures")

    print("\nGenerating DVH plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    num_structures = len([k for k in dvh_table.keys() if k != 'Dose_Gy'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    dose_values = dvh_table["Dose_Gy"]
    
    structure_index = 0
    for struct_name, volumes in dvh_table.items():
        if struct_name == "Dose_Gy":
            continue

        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        
        ax1.plot(dose_values, volumes, 
                label=clean_name, 
                linewidth=2, 
                color=colors[structure_index])
        structure_index += 1
    
    ax1.set_xlabel('Dose (Gy)', fontsize=12)
    ax1.set_ylabel('Volume (cm³)', fontsize=12)
    ax1.set_title('Dose-Volume Histogram (All Structures)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    critical_keywords = ['PTV', 'GTV', 'CTV', 'ORGAN', 'BLADDER', 'RECTUM', 'FEMUR', 'BOWEL']
    
    structure_index = 0
    for struct_name, volumes in dvh_table.items():
        if struct_name == "Dose_Gy":
            continue
            
        is_critical = any(keyword.upper() in struct_name.upper() for keyword in critical_keywords)
        is_external = any(ext.upper() in struct_name.upper() for ext in ['EXTERNAL', 'BODY', 'SKIN'])
        
        if is_critical and not is_external:
            clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
            ax2.plot(dose_values, volumes, 
                    label=clean_name, 
                    linewidth=2, 
                    color=colors[structure_index])
        structure_index += 1
    
    ax2.set_xlabel('Dose (Gy)', fontsize=12)
    ax2.set_ylabel('Volume (cm³)', fontsize=12)
    ax2.set_title('DVH - Critical Structures Only', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    plot_path = os.path.join("dvh_data_combined", "DVH_Plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"DVH plot saved to: {plot_path}")
    
    plt.show()
    
    print("\nDVH Summary Statistics:")
    print("="*60)
    
    for struct_name, volumes in dvh_table.items():
        if struct_name == "Dose_Gy":
            continue
            
        clean_name = struct_name.replace("_Volume_cm3", "").replace("_", " ")
        max_volume = max(volumes)
        
        dose_95_vol = None
        dose_50_vol = None
        dose_5_vol = None
        
        if max_volume > 0:
            target_95 = 0.95 * max_volume
            target_50 = 0.50 * max_volume
            target_5 = 0.05 * max_volume
            
            for j, vol in enumerate(volumes):
                if dose_95_vol is None and vol <= target_95:
                    dose_95_vol = dose_values[j]
                if dose_50_vol is None and vol <= target_50:
                    dose_50_vol = dose_values[j]
                if dose_5_vol is None and vol <= target_5:
                    dose_5_vol = dose_values[j]
        
        print(f"\n{clean_name}:")
        print(f"  Total Volume: {max_volume:.2f} cm³")
        if dose_95_vol is not None:
            print(f"  D95 (dose to 95% volume): {dose_95_vol:.2f} Gy")
        if dose_50_vol is not None:
            print(f"  D50 (dose to 50% volume): {dose_50_vol:.2f} Gy")
        if dose_5_vol is not None:
            print(f"  D5 (dose to 5% volume): {dose_5_vol:.2f} Gy")
    
    print("\n" + "="*60)
