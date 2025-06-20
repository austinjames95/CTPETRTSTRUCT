
from dicompylercore import dicomparser, dvhcalc
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from helper.resolution import Resolution, extract_affine_transform_from_reg
from helper.resample import Volume
from helper.binary_masks import create_structure_masks

def plot_pvh(pvh_table_absolute, pvh_table_relative):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
    num_structures = len([k for k in pvh_table_absolute.keys() if k != 'BQML'])
    if num_structures <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_structures))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_structures, 20)))

    pet_values = pvh_table_absolute["BQML"]
    structure_index = 0

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

    rel_plot_path = os.path.join("generated_data/PVH", "PVH_Relative_Plot.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    print(f"Relative PVH plot saved to: {rel_plot_path}")
    
    print("\nPVH Summary Statistics:")
    print("="*60)
    
    for struct_name, volumes in pvh_table_absolute.items():
        if struct_name == "BQML":
            continue
            
        clean_name = struct_name.replace("_Volume_mL", "").replace("_", " ")
        max_volume = volumes[0]  

        rel_struct_name = struct_name.replace("_Volume_mL", "")
        if rel_struct_name in pvh_table_relative:
            rel_volumes = pvh_table_relative[rel_struct_name]
            
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

def export_csv(relative_path, pvh_table_relative, custom_bins):
    with open(relative_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        headers = list(pvh_table_relative.keys())
        writer.writerow(headers)
        
        for i in range(len(custom_bins)):
            row = []
            for h in headers:
                if h == "BQML":
                    row.append("{:.6f}".format(pvh_table_relative[h][i]))
                else:
                    row.append("{:.6f}".format(pvh_table_relative[h][i]))
            writer.writerow(row)

def process_pvh(ct_datasets, pet_datasets, rd_dataset, reg_dataset, rs_dataset):
    if pet_datasets:
        print(f"\nProcessing {len(pet_datasets)} PET slices...")
        
        if rd_dataset:
            print("Aligning PET volume to CT grid...")

            pet_volume, sorted_pet = Volume.create_pet_volume(pet_datasets)

            if pet_volume is not None:
                print(f"Original PET shape: {pet_volume.shape}")

                sorted_ct = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
                ct_origin = np.array(sorted_ct[0].ImagePositionPatient, dtype=np.float32)
                
                ct_spacing = list(map(float, sorted_ct[0].PixelSpacing)) 
                thickness = float(sorted_ct[0].SliceThickness)
                ct_spacing.append(thickness) 

                ct_shape = (len(sorted_ct), sorted_ct[0].Rows, sorted_ct[0].Columns)

                reg_transform = None
                if reg_dataset:
                    reg_transform = extract_affine_transform_from_reg(reg_dataset)

                pet_volume = Volume.resample_pet_to_ct(pet_datasets, ct_datasets, pet_volume, reg_transform=reg_transform)

                print(f"Resampled PET shape (CT grid): {pet_volume.shape}")
        else:
            pet_volume, sorted_pet = Volume.create_pet_volume(pet_datasets)
        

        if pet_volume is not None:
            structures = None
            if rs_dataset:
                try:
                    dp_struct = dicomparser.DicomParser(rs_dataset)
                    structures = dp_struct.GetStructures()
                except Exception as e:
                    print(f"Could not get structures for overlay: {e}")

            structure_masks = create_structure_masks(structures, rs_dataset, ct_datasets, pet_volume.shape)
            spacing = list(map(float, ct_datasets[0].PixelSpacing)) 
            thickness = float(ct_datasets[0].SliceThickness)
            voxel_volume_mm3 = spacing[0] * spacing[1] * thickness
            voxel_volume_ml = voxel_volume_mm3 / 1000.0

            pet_step, max_pet = Resolution.get_pet_resolution(pet_volume)
        
            estimated_bins = int(max_pet / pet_step) + 1
            if estimated_bins > 10000:
                print(f"Warning: Too many bins ({estimated_bins}), adjusting step size")
                pet_step = max_pet / 10000.0
                estimated_bins = 10000
            
            custom_bins = np.arange(0, max_pet + pet_step, pet_step)
            
            print(f"PET histogram parameters:")
            print(f"  Step size: {pet_step}")
            print(f"  Max value: {max_pet}")
            print(f"  Number of bins: {len(custom_bins)}")
            
            pvh_table_absolute = {"BQML": [float(p) for p in custom_bins.tolist()]}
            pvh_table_relative = {"BQML": [float(p) for p in custom_bins.tolist()]}
            
            for sid, mask in structure_masks.items():
                if sid not in structures:
                    continue
                    
                structure_pet_values = pet_volume[mask]
                
                if structure_pet_values.size == 0:
                    print(f"Warning: No voxels found for structure {structures[sid]['name']}")
                    continue
                
                total_voxels = structure_pet_values.size
                total_volume_ml = total_voxels * voxel_volume_ml
                
                cumulative_volumes = []
                relative_volumes = []
                
                for pet_level in custom_bins:
                    voxel_count = np.sum(structure_pet_values >= pet_level)
                    volume_ml = voxel_count * voxel_volume_ml
                    cumulative_volumes.append(volume_ml)
                    
                    if total_volume_ml > 0:
                        relative_volume = volume_ml / total_volume_ml
                    else:
                        relative_volume = 0.0
                    relative_volumes.append(relative_volume)
                
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
            
            relative_path = os.path.join("generated_data/PVH", "CumulativePVH_AllStructures_RelativeUnits.csv")
            
            export_csv(relative_path, pvh_table_relative, custom_bins)

            print(f"Relative PET Volume Histogram exported to: {relative_path}")
            print(f"Data exported with:")
            print(f"   - PET activity in Bq/mL")
            print(f"   - {len(custom_bins)} activity levels")
            print(f"   - {len([k for k in pvh_table_absolute.keys() if k != 'BQML'])} structures")
        
            print("\nGenerating PVH plots...")

            plot_pvh(pvh_table_absolute, pvh_table_relative)

        else:
            print("Failed to create PET volume")
    else:
        print("\nNo PET data found - skipping PET visualization")

