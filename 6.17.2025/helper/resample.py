import SimpleITK as sitk
import numpy as np

def sort_pet_slices(pet_datasets):
    """Sort PET slices by position for proper visualization"""
    pet_with_position = []
    for ds in pet_datasets:
        try:
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                z_pos = float(ds.SliceLocation)
            else:
                z_pos = 0.0
            pet_with_position.append((z_pos, ds))
        except:
            pet_with_position.append((0.0, ds))
    
    pet_with_position.sort(key=lambda x: x[0])
    return [ds for _, ds in pet_with_position]

class Volume:

    def create_pet_volume(pet_datasets):
        """Create 3D volume from PET slices"""
        if not pet_datasets:
            return None, None
        
        sorted_pet = sort_pet_slices(pet_datasets)
        first_slice = sorted_pet[0]

        rows = first_slice.Rows
        cols = first_slice.Columns
        num_slices = len(sorted_pet)
        
        pet_volume = np.zeros((num_slices, rows, cols))

        for i, ds in enumerate(sorted_pet):
            try:
                pixel_array = ds.pixel_array.astype(np.float32)

                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                
                pet_volume[i] = pixel_array
            except Exception as e:
                print(f"Error processing PET slice {i}: {e}")
                pet_volume[i] = np.zeros((rows, cols))
        
        return pet_volume, sorted_pet

    
    def resample_pet_to_ct(pet_datasets, ct_datasets, pet_volume, reg_transform=None):
        """ Resamples a 3D PET image volume to align spatially with a CT image grid """
        pet_sorted = sorted(pet_datasets, key=lambda x: float(x.ImagePositionPatient[2]))
        ct_sorted = sorted(ct_datasets, key=lambda x: float(x.ImagePositionPatient[2]))

        pet_spacing = list(map(float, pet_sorted[0].PixelSpacing)) 
        if len(pet_sorted) > 1:
            z1 = float(pet_sorted[0].ImagePositionPatient[2])
            z2 = float(pet_sorted[1].ImagePositionPatient[2])
            dz_pet = abs(z2 - z1)
        else:
            dz_pet = float(pet_sorted[0].SliceThickness)

        pet_spacing.append(dz_pet)
        pet_origin = list(map(float, pet_sorted[0].ImagePositionPatient))

        iop = pet_sorted[0].ImageOrientationPatient
        axis_x = np.array(iop[:3])
        axis_y = np.array(iop[3:])
        axis_z = np.cross(axis_x, axis_y)
        direction = np.concatenate([axis_x, axis_y, axis_z]).tolist()

        pet_img = sitk.GetImageFromArray(pet_volume.astype(np.float32)) 
        pet_img.SetSpacing([pet_spacing[1], pet_spacing[0], pet_spacing[2]])  
        pet_img.SetOrigin(pet_origin)
        pet_img.SetDirection(direction)

        ct_shape = (ct_sorted[0].Columns, ct_sorted[0].Rows, len(ct_sorted))
        ct_spacing = list(map(float, ct_sorted[0].PixelSpacing))  
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
    
