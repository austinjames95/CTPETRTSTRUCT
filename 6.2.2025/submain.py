import numpy as np
import SimpleITK as sitk

def get_dose_resolution(rd_dataset):
    dose_array = rd_dataset.pixel_array * rd_dataset.DoseGridScaling
    unique_doses = np.unique(dose_array.flatten())
    dose_diffs = np.diff(np.sort(unique_doses))
    min_step = np.min(dose_diffs[dose_diffs > 0])
    return min_step, np.max(unique_doses)

def load_pet_with_transforms(pet_datasets, reg_dataset=None, target_reference_frame=None):
    """
    Load PET data and apply coordinate transformations from REG file if available
    Returns PET image in Bq/ml units with proper coordinate system
    
    Args:
        pet_datasets: List of PET DICOM datasets
        reg_dataset: Registration DICOM dataset (optional)
        target_reference_frame: Target frame of reference UID for alignment
    """
    if not pet_datasets:
        return None, None
    
    print(f"\n Processing {len(pet_datasets)} PET slices...")
    
    try:
        pet_datasets.sort(key=lambda x: float(getattr(x, 'InstanceNumber', 0)))
        
        # Get the PET frame of reference
        pet_frame_of_reference = pet_datasets[0].FrameOfReferenceUID if hasattr(pet_datasets[0], 'FrameOfReferenceUID') else None
        print(f"   PET Frame of Reference: {pet_frame_of_reference}")
        
        reader = sitk.ImageSeriesReader()
        dicom_names = [ds.filename for ds in pet_datasets if hasattr(ds, 'filename')]
        
        if not dicom_names:
            # Manual PET reconstruction with proper coordinate handling
            print("Using pydicom approach for PET reconstruction...")
            
            rows, cols = pet_datasets[0].Rows, pet_datasets[0].Columns
            num_slices = len(pet_datasets)
            pet_array = np.zeros((num_slices, rows, cols), dtype=np.float32)
            
            pixel_spacing = pet_datasets[0].PixelSpacing
            slice_thickness = float(getattr(pet_datasets[0], 'SliceThickness', 1.0))
            
            # Process each slice with proper SUV/activity conversion
            for i, ds in enumerate(pet_datasets):
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                
                slice_data = ds.pixel_array.astype(np.float32)
                slice_data = slice_data * slope + intercept
                
                # Apply SUV to Bq/ml conversion if radiopharmaceutical info is available
                if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
                    # This would require patient weight, injection dose, etc.
                    # For now, assume the rescale slope/intercept handles the conversion
                    pass
                
                pet_array[i] = slice_data
            
            pet_image = sitk.GetImageFromArray(pet_array)
            pet_image.SetSpacing([float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness])
            
            if hasattr(pet_datasets[0], 'ImagePositionPatient'):
                origin = [float(x) for x in pet_datasets[0].ImagePositionPatient]
                pet_image.SetOrigin(origin)
            
            if hasattr(pet_datasets[0], 'ImageOrientationPatient'):
                orientation = [float(x) for x in pet_datasets[0].ImageOrientationPatient]
                direction = np.eye(3)
                direction[0, :2] = orientation[:3]
                direction[1, :2] = orientation[3:]
                direction[2] = np.cross(direction[0], direction[1])
                pet_image.SetDirection(direction.flatten())
        else:
            reader.SetFileNames(dicom_names)
            pet_image = reader.Execute()
        
        print(f"   PET image loaded: {pet_image.GetSize()} voxels")
        print(f"   Spacing: {pet_image.GetSpacing()} mm")
        print(f"   Origin: {pet_image.GetOrigin()} mm")
        
        # Store the original transform for coordinate conversion
        original_transform = None
        
        # Apply registration transformation if REG file is available
        if reg_dataset:
            print("  Applying registration transformation...")
            pet_image, original_transform = apply_registration_transform(pet_image, reg_dataset, return_transform=True)
        
        return pet_image, original_transform
        
    except Exception as e:
        print(f"Error loading PET data: {e}")
        return None, None

def apply_registration_transform(image, reg_dataset, return_transform=False):
    """
    Apply transformation from REG DICOM file to image
    Returns transformed image and optionally the transform itself
    """
    try:
        transform_applied = None
        
        if hasattr(reg_dataset, 'RegistrationSequence'):
            reg_sequence = reg_dataset.RegistrationSequence[0]
            
            # Check source and target frame of reference
            source_frame = getattr(reg_sequence, 'SourceFrameOfReferenceUID', None)
            target_frame = getattr(reg_sequence, 'TargetFrameOfReferenceUID', None)
            print(f"   Registration: {source_frame} -> {target_frame}")
            
            if hasattr(reg_sequence, 'MatrixRegistrationSequence'):
                matrix_seq = reg_sequence.MatrixRegistrationSequence[0]
                
                if hasattr(matrix_seq, 'MatrixSequence'):
                    transform_matrix = matrix_seq.MatrixSequence[0].FrameOfReferenceTransformationMatrix
                    matrix_4x4 = np.array(transform_matrix).reshape(4, 4)
                    
                    print(f"   Transformation matrix:\n{matrix_4x4}")
                    
                    sitk_transform = sitk.AffineTransform(3)
                    sitk_transform.SetMatrix(matrix_4x4[:3, :3].flatten())
                    sitk_transform.SetTranslation(matrix_4x4[:3, 3])
                    
                    # Apply transformation
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetTransform(sitk_transform)
                    resampler.SetReferenceImage(image)
                    resampler.SetInterpolator(sitk.sitkLinear)
                    resampler.SetDefaultPixelValue(0.0)
                    
                    transformed_image = resampler.Execute(image)
                    transform_applied = sitk_transform
                    
                    print("   Registration transformation applied successfully")
                    
                    if return_transform:
                        return transformed_image, transform_applied
                    else:
                        return transformed_image
        
        print("   No valid transformation found in REG file, using original image")
        if return_transform:
            return image, None
        else:
            return image
        
    except Exception as e:
        print(f"Error applying registration: {e}")
        if return_transform:
            return image, None
        else:
            return image

def sample_pet_at_structures_with_transforms(pet_image, structures_dict, dp_struct, reg_transform=None):
    """
    Sample PET values at structure locations with proper coordinate transformations
    Handles both DICOM coordinate systems and registration transformations
    """
    if pet_image is None:
        print("No PET image available for sampling.")
        return {}
    
    print("\nSampling PET values at structure locations with coordinate transforms...")
    pet_stats = {}
    
    # Get structure frame of reference
    struct_frame_ref = getattr(dp_struct.ds, 'FrameOfReferenceUID', None)
    print(f"Structure Frame of Reference: {struct_frame_ref}")
    
    for sid, struct in structures_dict.items():
        try:
            struct_name = struct['name']
            print(f"Processing structure: {struct_name}")
            
            contour_data = dp_struct.GetStructureCoordinates(sid)
            if not contour_data:
                continue
            
            pet_values = []
            total_points = 0
            valid_points = 0
            
            # Process each slice/contour in the dictionary
            for slice_key, coordinate_data in contour_data.items():
                try:
                    points = []
                    
                    if isinstance(coordinate_data, str):
                        # Parse coordinate string
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', coordinate_data)
                        coords = [float(x) for x in numbers if x]
                        
                        if len(coords) >= 3 and len(coords) % 3 == 0:
                            points = [[coords[i], coords[i+1], coords[i+2]] 
                                    for i in range(0, len(coords), 3)]
                    
                    elif isinstance(coordinate_data, (list, np.ndarray)):
                        if len(coordinate_data) > 0:
                            if isinstance(coordinate_data[0], (list, np.ndarray)) and len(coordinate_data[0]) >= 3:
                                points = coordinate_data
                            elif len(coordinate_data) % 3 == 0:
                                points = [[coordinate_data[i], coordinate_data[i+1], coordinate_data[i+2]] 
                                        for i in range(0, len(coordinate_data), 3)]
                    
                    # Process points with coordinate transformations
                    for point in points:
                        total_points += 1
                        try:
                            # Start with structure coordinates (DICOM mm coordinates)
                            physical_point = [float(point[0]), float(point[1]), float(point[2])]
                            
                            # If we have a registration transform, we might need to apply inverse
                            # to transform structure coordinates to PET space
                            if reg_transform is not None:
                                try:
                                    # Apply inverse transform to structure coordinates
                                    inverse_transform = reg_transform.GetInverse()
                                    transformed_point = inverse_transform.TransformPoint(physical_point)
                                    physical_point = list(transformed_point)
                                except Exception as e:
                                    # If inverse transform fails, use original coordinates
                                    print(f"    Warning: Could not apply inverse transform to point: {e}")
                            
                            # Transform physical coordinates to PET image indices
                            try:
                                index = pet_image.TransformPhysicalPointToIndex(physical_point)
                                size = pet_image.GetSize()
                                
                                if (0 <= index[0] < size[0] and 
                                    0 <= index[1] < size[1] and 
                                    0 <= index[2] < size[2]):
                                    pet_value = pet_image.GetPixel(index)
                                    pet_values.append(pet_value)
                                    valid_points += 1
                                    
                            except Exception as e:
                                # Point transformation failed
                                continue
                                
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    print(f"  Warning: Could not process slice {slice_key}: {e}")
                    continue
            
            print(f"  Processed {total_points} total points, {valid_points} valid samples")
            
            if pet_values:
                pet_values = np.array(pet_values)
                
                pet_stats[struct_name] = {
                    'mean_bqml': np.mean(pet_values),
                    'max_bqml': np.max(pet_values),
                    'min_bqml': np.min(pet_values),
                    'std_bqml': np.std(pet_values),
                    'median_bqml': np.median(pet_values),
                    'n_samples': len(pet_values),
                    'sampling_efficiency': valid_points / total_points if total_points > 0 else 0
                }
                
                print(f"  Mean: {pet_stats[struct_name]['mean_bqml']:.2f} Bq/ml")
                print(f"  Max:  {pet_stats[struct_name]['max_bqml']:.2f} Bq/ml")
                print(f"  Samples: {pet_stats[struct_name]['n_samples']}")
                print(f"  Efficiency: {pet_stats[struct_name]['sampling_efficiency']:.1%}")
            else:
                print(f"  No valid PET values found for {struct_name}")
        
        except Exception as e:
            print(f"Error sampling PET for {struct['name']}: {e}")
    
    return pet_stats

def normalize_name(name):
    return name.replace("_", " ").replace("-", " ").replace("/", " ").strip().lower()
