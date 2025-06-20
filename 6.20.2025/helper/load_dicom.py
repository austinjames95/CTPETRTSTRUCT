import pydicom
import glob
import os
from config import base_dir

""" 
Loads all of the DICOM Data and returns them as 
ct_datasets, pet_datasets, rs_dataset, rd_dataset, reg_dataset
"""


def read_dicom():
    ct_datasets = []
    pet_datasets = []
    rs_dataset = None
    rd_dataset = None
    reg_dataset = None

    dicom_files = glob.glob(os.path.join(base_dir, '**', '*'), recursive=True)
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

    os.makedirs("generated_data", exist_ok=True)
    os.makedirs("generated_data/DVH", exist_ok=True)
    os.makedirs("generated_data/PVH", exist_ok=True)

    return ct_datasets, pet_datasets, rs_dataset, rd_dataset, reg_dataset
