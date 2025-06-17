from ddvh2cdvh import compare_relative_dvh, compare_relative_pvh
from helper.load_dicom import read_dicom
from processing.pvh import process_pvh
from processing.dvh import process_dvh


ct_datasets, pet_datasets, rs_dataset, rd_dataset, reg_dataset = read_dicom()

process_dvh(rd_dataset, rs_dataset)
process_pvh(ct_datasets, pet_datasets, rd_dataset, reg_dataset, rs_dataset)


compare_relative_dvh()
compare_relative_pvh()

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)
