
from dicompylercore import dicomparser, dvhcalc
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from helper.resolution import Resolution

def plot_dvh(dvh_table_absolute, dvh_table_relative):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) 
    num_structures = len([k for k in dvh_table_absolute.keys() if k != 'GY'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    dose_values = dvh_table_absolute["GY"]

    structure_index = 0

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

    rel_plot_path = os.path.join("generated_data/DVH", "DVH_Relative_Plot.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    print(f" Relative DVH plot saved to: {rel_plot_path}")
    
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

def export_csv(relative_path_dvh, dvh_table_relative, custom_bins):
    with open(relative_path_dvh, "w", newline="") as f:
        writer = csv.writer(f)
        
        headers = list(dvh_table_relative.keys())
        writer.writerow(headers)
        
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "GY":
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
                else:
                    row.append("{:.6f}".format(dvh_table_relative[h][i]))
            writer.writerow(row)

def process_dvh(rd_dataset, rs_dataset):
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
        
            
        dose_step, max_dose = Resolution.get_dose_resolution(rd_dataset)
        custom_bins = np.arange(0, max_dose + dose_step, dose_step)
        
        dose_units = getattr(rd_dataset, 'DoseUnits', 'GY')
        print(f"\nDose units from DICOM: {dose_units}")
        
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

                absolute_volumes = []
                for dose in custom_bins:
                    vol = dvh_struct.volume_constraint(dose, dose_units="Gy")
                    
                    if hasattr(vol, 'value'):
                        volume_value = float(vol.value)
                    else:
                        volume_value = float(vol)
                    
                    absolute_volumes.append(volume_value)

                relative_volumes = []
                if structure_volume > 0:
                    relative_volumes = [vol / structure_volume for vol in absolute_volumes]
                else:
                    relative_volumes = [0.0] * len(custom_bins)

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

        relative_path_dvh = os.path.join("generated_data/DVH", "CumulativeDVH_AllStructures_RelativeUnits.csv")
        
        export_csv(relative_path_dvh, dvh_table_relative, custom_bins)

        print(f"Relative DVH written to: {relative_path_dvh}")
        print(f"Data exported with:")
        print(f"   - Dose in Gray (Gy)")
        print(f"   - {len(custom_bins)} dose points")
        print(f"   - {len([k for k in dvh_table_absolute.keys() if k != 'GY'])} structures")

        print("\nGenerating DVH plots...")
        
        plot_dvh(dvh_table_absolute, dvh_table_relative)

    else:
        print("Required DICOM files (RTSTRUCT and RTDOSE) not found or could not be loaded.")
