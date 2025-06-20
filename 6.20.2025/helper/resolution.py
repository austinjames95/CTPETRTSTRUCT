import numpy as np
import SimpleITK as sitk

""" GETS RESOLUTION OF DIFFERENT DATASETS """

class Resolution:
    def get_dose_resolution(rd_dataset):
        dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling
        unique_doses = np.unique(dose_array.flatten())
        dose_diffs = np.diff(np.sort(unique_doses))
        min_step = np.min(dose_diffs[dose_diffs > 0])
        return min_step, np.max(unique_doses)

    def get_pet_resolution(pet_volume):
        pet_array = pet_volume.flatten()
        unique_values = np.unique(pet_array[pet_array > 0]) 
        max_pet = np.max(pet_array)
        
        if len(unique_values) < 2:
            return max_pet / 1000.0, max_pet 
        
        value_diffs = np.diff(np.sort(unique_values))
        min_step = np.min(value_diffs[value_diffs > 0])
        
        
        max_bins = 10000 # Able to change 
        min_reasonable_step = max_pet / max_bins
        
        
        final_step = max(min_step, min_reasonable_step)
        
        print(f"PET resolution calculation:")
        print(f"  Calculated min step: {min_step}")
        print(f"  Min reasonable step: {min_reasonable_step}")
        print(f"  Final step used: {final_step}")
        
        return final_step, max_pet
    

def extract_affine_transform_from_reg(reg_dataset):
    """ Extracts a 3D affine transformation matrix from a DICOM registration (REG) object and converts it into a SimpleITK.AffineTransform  """
    try:
        matrix_seq = reg_dataset[0x0070, 0x0308].value[0]  
        matrix_data = matrix_seq[0x3006, 0x00C6].value     

        if len(matrix_data) != 16:
            print("Expected 16 values for 4x4 matrix, got:", len(matrix_data))
            return None

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
