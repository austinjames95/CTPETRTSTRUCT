import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def compare_relative_dvh():
    df = pd.read_csv(r'GivenData\absolute_dvh_export.csv', header=None)

    blank_indices = df.index[df.isnull().all(axis=1)].tolist()

    if 0 not in blank_indices:
        blank_indices = [0] + blank_indices

    # Add the start (0) and end (len(df)) to help split
    split_points = [0] + blank_indices + [len(df)]

    # Split into chunks between blank rows
    sections = []

    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = df.iloc[start:end].dropna(how='all')  # Drop any full-NaN rows just in case
        if not chunk.empty:
            sections.append(chunk.reset_index(drop=True))

    cdvh_results = {}

    for section in sections:
        if section.shape[0] < 3:
            continue  # Too small to be useful

        # Assume first row is the structure name
        structure_name = str(section.iloc[0, 0]).strip()
        
        # Locate the header row (should start with "GY")
        dose_row_index = section[section[0].astype(str).str.upper() == "GY"].index
        if dose_row_index.empty:
            continue  # No header, skip
        
        header_idx = dose_row_index[0]
        data_rows = section.iloc[header_idx + 1:]
        data_rows = data_rows.iloc[:, :2]  # Select only first two columns
        data_rows.columns = ["GY", "cm3"]
        data_rows = data_rows.dropna()
        
        try:
            # Ensure numeric
            data_rows["GY"] = pd.to_numeric(data_rows["GY"], errors='coerce')
            data_rows["cm3"] = pd.to_numeric(data_rows["cm3"], errors='coerce')
            data_rows = data_rows.dropna()

            # Step 3: Convert to cumulative DVH (relative)
            data_rows = data_rows.sort_values(by="GY").reset_index(drop=True)
            total_volume = data_rows["cm3"].sum()
            data_rows["CumulativeVolume_cm3"] = data_rows["cm3"][::-1].cumsum()[::-1]
            data_rows["RelativeCumVolume"] = data_rows["CumulativeVolume_cm3"] / total_volume

            cdvh_results[structure_name] = data_rows

        except Exception as e:
            print(f"Skipping {structure_name} due to error: {e}")

    # Step 4: Example — plot or export
    for name, cdvh in cdvh_results.items():
        plt.plot(cdvh["GY"], cdvh["RelativeCumVolume"], label=name)

    plt.xlabel("Dose (Gy)", fontsize=12)
    plt.ylabel("Relative Volume (fraction)", fontsize=12)
    plt.title("Cumulative DVH - Relative Volumes", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    rel_plot_path = os.path.join("generated_data/DVH", "CDVH_Converted.png")
    plt.savefig(rel_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Relative DVH plot saved to: {rel_plot_path}")

    try:
        ref_path = os.path.join("generated_data/DVH", "CumulativeDVH_AllStructures_RelativeUnits.csv")
        ref_df = pd.read_csv(ref_path, comment='#')
        dose_ref = ref_df["GY"].values

        diff_output = {"GY": dose_ref}

        for name, cdvh in cdvh_results.items():
            rel_col = f"{name.replace(' ', '_')}_RelativeCumVolume"
            if rel_col not in ref_df.columns:
                print(f"Structure '{name}' not found in DICOM DVH. Skipping diff.")
                continue

            # Interpolate CDVH to match reference dose points
            interp = np.interp(dose_ref, cdvh["GY"], cdvh["RelativeCumVolume"], left=np.nan, right=np.nan)
            ref_values = ref_df[rel_col].values
            difference = interp - ref_values

            diff_output[f"{name}_Diff"] = difference

        # Save the difference as a CSV
        diff_df = pd.DataFrame(diff_output)
        diff_path = os.path.join("generated_data/DVH", "DVH_Differences.csv")
        diff_df.to_csv(diff_path, index=False)
        print(f"DVH differences saved to: {diff_path}")

    except Exception as e:
        print(f"Error comparing to reference DVH: {e}")

def compare_relative_pvh():
    df = pd.read_csv(r'GivenData\pet_volume_histogrm.csv', header=None)

    # Identify blank rows
    blank_indices = df.index[df.isnull().all(axis=1)].tolist()

    # Ensure first block is included
    if 0 not in blank_indices:
        blank_indices = [0] + blank_indices

    # Add final row index
    split_points = blank_indices + [len(df)]

    # Split into sections
    sections = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = df.iloc[start:end].dropna(how='all')
        if not chunk.empty:
            sections.append(chunk.reset_index(drop=True))

    # Parse PVH data and convert
    pvh_results = {}

    for section in sections:
        if section.shape[0] < 3:
            continue  # Skip tiny sections

        structure_name = str(section.iloc[0, 0]).strip()
        
        # Find the header row (usually "Bq/mL" or "PET" or something like that)
        value_row_index = section[section[0].astype(str).str.upper().str.contains("BQ|SUV|PET", na=False)].index
        if value_row_index.empty:
            continue

        header_idx = value_row_index[0]
        data_rows = section.iloc[header_idx + 1:]
        data_rows = data_rows.iloc[:, :2]  # Use first two columns only
        data_rows.columns = ["BQML", "cm3"]
        data_rows = data_rows.dropna()

        try:
            data_rows["BQML"] = pd.to_numeric(data_rows["BQML"], errors='coerce')
            data_rows["cm3"] = pd.to_numeric(data_rows["cm3"], errors='coerce')
            data_rows = data_rows.dropna()

            data_rows = data_rows.sort_values(by="BQML").reset_index(drop=True)
            total_volume = data_rows["cm3"].sum()

            # Convert to cumulative and relative
            data_rows["CumulativeVolume_mL"] = data_rows["cm3"][::-1].cumsum()[::-1]
            data_rows["RelativeCumVolume"] = data_rows["CumulativeVolume_mL"] / total_volume

            pvh_results[structure_name] = data_rows

        except Exception as e:
            print(f"Skipping {structure_name} due to error: {e}")

    # Plot example
    for name, pvh in pvh_results.items():
        plt.plot(pvh["BQML"], pvh["RelativeCumVolume"], label=name)

    plt.xlabel("PET Activity (BQML)")
    plt.ylabel("Relative Volume")
    plt.title("Cumulative PVHs")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    rel_plot_path_PVH = os.path.join("generated_data/PVH", "CPVH_Converted.png")
    plt.savefig(rel_plot_path_PVH, dpi=300, bbox_inches='tight')
    plt.show()

    print(f" Relative DVH plot saved to: {rel_plot_path_PVH}")

    try:
        ref_path = os.path.join("generated_data/PVH", "CumulativePVH_AllStructures_RelativeUnits.csv")
        ref_df = pd.read_csv(ref_path, comment='#')
        
        # Detect dose column automatically
        bq_col = next((c for c in ref_df.columns if "bq" in c.lower()), None)
        if not bq_col:
            raise KeyError("No PET activity column found in reference PVH.")

        activity_ref = ref_df[bq_col].values
        diff_output = {bq_col: activity_ref}

        for name, pvh in pvh_results.items():
            rel_col = f"{name.replace(' ', '_')}_RelativeCumVolume"
            if rel_col not in ref_df.columns:
                print(f"Structure '{name}' not found in reference PVH. Skipping diff.")
                continue

            interp = np.interp(activity_ref, pvh["BQML"], pvh["RelativeCumVolume"], left=np.nan, right=np.nan)
            ref_values = ref_df[rel_col].values
            difference = interp - ref_values

            diff_output[f"{name}_Diff"] = difference

        diff_df = pd.DataFrame(diff_output)
        diff_path = os.path.join("generated_data/PVH", "PVH_Differences.csv")
        diff_df.to_csv(diff_path, index=False)
        print(f"PVH differences saved to: {diff_path}")

    except Exception as e:
        print(f"Error comparing to reference PVH: {e}")

