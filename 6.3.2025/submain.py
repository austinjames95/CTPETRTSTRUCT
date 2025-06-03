import numpy as np
import SimpleITK as sitk
import os
import csv

def safe_float_from_dicom(value, default=0.0):
    """Safely extract float value from DICOM attribute that might be dict, list, or scalar"""
    if value is None:
        return default
    
    # Handle dictionary types (common with some DICOM parsers)
    if isinstance(value, dict):
        # Try common keys that might contain the actual value
        for key in ['value', 'Value', 'val', 'Val', 0, '0']:
            if key in value:
                return safe_float_from_dicom(value[key], default)
        return default
    
    # Handle list/sequence types
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return safe_float_from_dicom(value[0], default)
    
    # Handle string representations
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    
    # Handle numeric types
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_list_from_dicom(value, expected_length=None, default=None):
    """Safely extract list from DICOM attribute that might be dict, nested, or scalar"""
    if value is None:
        return default if default is not None else []
    
    # Handle dictionary types
    if isinstance(value, dict):
        for key in ['value', 'Value', 'val', 'Val', 0, '0']:
            if key in value:
                return safe_list_from_dicom(value[key], expected_length, default)
        return default if default is not None else []
    
    # Handle already-list types
    if isinstance(value, (list, tuple)):
        # Flatten nested structures if needed
        result = []
        for item in value:
            if isinstance(item, (list, tuple)):
                result.extend(item)
            else:
                result.append(safe_float_from_dicom(item))
        
        if expected_length and len(result) >= expected_length:
            return result[:expected_length]
        return result
    
    # Handle single values
    single_val = safe_float_from_dicom(value)
    if expected_length:
        return [single_val] * expected_length
    return [single_val]

def safe_normalize_vector(vector, default_vector=None):
    """Safely normalize a vector, handling zero magnitude cases"""
    try:
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        
        # Check for zero or very small magnitude
        if norm < 1e-10:  # Very small threshold to avoid division by zero
            if default_vector is not None:
                return np.array(default_vector)
            else:
                # Return a unit vector along the first axis
                unit_vector = np.zeros_like(vector)
                if len(unit_vector) > 0:
                    unit_vector[0] = 1.0
                return unit_vector
        
        return vector / norm
    except:
        # If any error occurs, return default or unit vector
        if default_vector is not None:
            return np.array(default_vector)
        else:
            unit_vector = np.zeros(len(vector) if hasattr(vector, '__len__') else 3)
            if len(unit_vector) > 0:
                unit_vector[0] = 1.0
            return unit_vector

def get_dose_resolution(rd_dataset):
    """Get dose resolution with safe DICOM attribute handling"""
    try:
        # Handle DoseGridScaling safely
        dose_scaling = safe_float_from_dicom(getattr(rd_dataset, 'DoseGridScaling', 1.0), 1.0)
        dose_array = rd_dataset.pixel_array * dose_scaling
        
        unique_doses = np.unique(dose_array.flatten())
        dose_diffs = np.diff(np.sort(unique_doses))
        
        # Find minimum non-zero difference
        non_zero_diffs = dose_diffs[dose_diffs > 0]
        if len(non_zero_diffs) > 0:
            min_step = np.min(non_zero_diffs)
        else:
            min_step = 0.01  # Default step size
            
        return min_step, np.max(unique_doses)
    except Exception as e:
        print(f"Error in get_dose_resolution: {e}")
        return 0.01, 100.0  # Safe defaults

def create_transform_from_reg(reg_dataset):
    """Extract transformation matrix from REG DICOM file and create SimpleITK transform"""
    try:
        if not hasattr(reg_dataset, 'RegistrationSequence'):
            print("   No RegistrationSequence found in REG file")
            return None
            
        reg_sequence = reg_dataset.RegistrationSequence[0]
        
        # Check source and target frame of reference
        source_frame = getattr(reg_sequence, 'SourceFrameOfReferenceUID', None)
        target_frame = getattr(reg_sequence, 'TargetFrameOfReferenceUID', None)
        print(f"   Registration: {source_frame} -> {target_frame}")
        
        if hasattr(reg_sequence, 'MatrixRegistrationSequence'):
            matrix_seq = reg_sequence.MatrixRegistrationSequence[0]
            
            if hasattr(matrix_seq, 'MatrixSequence'):
                transform_matrix_raw = matrix_seq.MatrixSequence[0].FrameOfReferenceTransformationMatrix
                
                # Handle the transformation matrix safely
                if isinstance(transform_matrix_raw, (list, tuple)):
                    transform_matrix = [safe_float_from_dicom(x) for x in transform_matrix_raw]
                else:
                    # If it's a single value or dict, try to extract
                    transform_matrix = safe_list_from_dicom(transform_matrix_raw, 16)
                
                if len(transform_matrix) >= 16:
                    matrix_4x4 = np.array(transform_matrix[:16]).reshape(4, 4)
                    
                    print(f"   Transformation matrix:\n{matrix_4x4}")
                    
                    # Create SimpleITK transform
                    sitk_transform = sitk.AffineTransform(3)
                    # Set rotation/scaling (upper 3x3)
                    sitk_transform.SetMatrix(matrix_4x4[:3, :3].flatten())
                    # Set translation (last column, first 3 elements)
                    sitk_transform.SetTranslation(matrix_4x4[:3, 3])
                    
                    return sitk_transform
                    
        print("   No valid transformation matrix found in REG file")
        return None
        
    except Exception as e:
        print(f"Error extracting registration transform: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_pet_with_transforms(pet_datasets, reg_dataset=None, target_reference_frame=None):
    """
    Load PET data and apply coordinate transformations from REG file if available
    Returns PET image in Bq/ml units with proper coordinate system
    """
    if not pet_datasets:
        return None, None
    
    print(f"\nProcessing {len(pet_datasets)} PET slices...")
    
    try:
        # Sort by instance number or slice location
        def get_sort_key(ds):
            # Try ImagePositionPatient Z-coordinate first
            if hasattr(ds, 'ImagePositionPatient'):
                try:
                    position = safe_list_from_dicom(ds.ImagePositionPatient, 3, [0.0, 0.0, 0.0])
                    if len(position) >= 3:
                        return float(position[2])  # Z-coordinate
                except:
                    pass
            
            # Try SliceLocation
            slice_loc = safe_float_from_dicom(getattr(ds, 'SliceLocation', None))
            if slice_loc != 0.0:
                return slice_loc
            
            # Fall back to InstanceNumber
            return safe_float_from_dicom(getattr(ds, 'InstanceNumber', 0))
        
        pet_datasets.sort(key=get_sort_key)
        
        # Get the PET frame of reference
        pet_frame_of_reference = getattr(pet_datasets[0], 'FrameOfReferenceUID', None)
        print(f"   PET Frame of Reference: {pet_frame_of_reference}")
        
        # Reconstruct PET volume
        print("   Reconstructing PET volume from DICOM slices...")
        
        rows, cols = pet_datasets[0].Rows, pet_datasets[0].Columns
        num_slices = len(pet_datasets)
        pet_array = np.zeros((num_slices, rows, cols), dtype=np.float32)
        
        # Get spacing information with safe extraction
        pixel_spacing_raw = getattr(pet_datasets[0], 'PixelSpacing', [1.0, 1.0])
        pixel_spacing = safe_list_from_dicom(pixel_spacing_raw, 2, [1.0, 1.0])
        if len(pixel_spacing) < 2:
            pixel_spacing = [1.0, 1.0]
        
        # Calculate slice spacing from actual positions
        slice_positions = []
        for ds in pet_datasets:
            if hasattr(ds, 'ImagePositionPatient'):
                try:
                    position = safe_list_from_dicom(ds.ImagePositionPatient, 3, [0.0, 0.0, 0.0])
                    if len(position) >= 3:
                        slice_positions.append(position[2])
                except:
                    slice_positions.append(0.0)
            else:
                slice_loc = safe_float_from_dicom(getattr(ds, 'SliceLocation', 0.0))
                slice_positions.append(slice_loc)
        
        # Calculate slice thickness from positions
        slice_thickness = safe_float_from_dicom(getattr(pet_datasets[0], 'SliceThickness', 1.0), 1.0)
        if len(set(slice_positions)) > 1:
            sorted_positions = sorted(slice_positions)
            if len(sorted_positions) > 1:
                calculated_spacing = abs(sorted_positions[1] - sorted_positions[0])
                if calculated_spacing > 0:
                    slice_thickness = calculated_spacing
                    print(f"   Calculated slice spacing: {slice_thickness:.2f} mm")
        
        # Process each slice with proper scaling
        print("   Processing PET slices with scaling factors...")
        for i, ds in enumerate(pet_datasets):
            # Handle rescale slope/intercept safely
            slope = safe_float_from_dicom(getattr(ds, 'RescaleSlope', 1.0), 1.0)
            intercept = safe_float_from_dicom(getattr(ds, 'RescaleIntercept', 0.0), 0.0) 
            slice_data = ds.pixel_array.astype(np.float32) * slope + intercept

            if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
                try:
                    radio_info = ds.RadiopharmaceuticalInformationSequence[0]
                    if hasattr(radio_info, 'RadionuclideTotalDose'):
                        total_dose = safe_float_from_dicom(radio_info.RadionuclideTotalDose)
                        if total_dose > 0:
                            print(f"   Found radionuclide dose: {total_dose} Bq")
                except:
                    pass
            
            # Additional scaling factors
            if hasattr(ds, 'DoseGridScaling'):
                scaling = safe_float_from_dicom(ds.DoseGridScaling, 1.0)
                slice_data *= scaling
            
            pet_array[i] = slice_data
        
        # Create SimpleITK image with proper orientation
        pet_image = sitk.GetImageFromArray(pet_array)
        
        # Set spacing
        try:
            spacing_x = float(pixel_spacing[0]) if len(pixel_spacing) > 0 and pixel_spacing[0] > 0 else 1.0
            spacing_y = float(pixel_spacing[1]) if len(pixel_spacing) > 1 and pixel_spacing[1] > 0 else 1.0
            spacing_z = float(slice_thickness) if slice_thickness > 0 else 1.0
            
            pet_image.SetSpacing([spacing_x, spacing_y, spacing_z])
            print(f"   PET Spacing: [{spacing_x:.2f}, {spacing_y:.2f}, {spacing_z:.2f}] mm")
        except Exception as e:
            print(f"   Spacing error. Using defaults (1.0 mm): {e}")
            pet_image.SetSpacing([1.0, 1.0, 1.0])
        
        # Set origin from first slice
        if hasattr(pet_datasets[0], 'ImagePositionPatient'):
            origin_raw = pet_datasets[0].ImagePositionPatient
            origin = safe_list_from_dicom(origin_raw, 3, [0.0, 0.0, 0.0])
            if len(origin) >= 3:
                pet_image.SetOrigin(origin)
                print(f"   PET Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}] mm")
            else:
                print("   Using default origin [0.0, 0.0, 0.0]")
                pet_image.SetOrigin([0.0, 0.0, 0.0])
        else:
            pet_image.SetOrigin([0.0, 0.0, 0.0])
        
        # Set direction/orientation with safe normalization
        if hasattr(pet_datasets[0], 'ImageOrientationPatient'):
            orientation_raw = pet_datasets[0].ImageOrientationPatient
            orientation = safe_list_from_dicom(orientation_raw, 6, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            
            if len(orientation) >= 6:
                # Extract and safely normalize direction vectors
                row_dir = safe_normalize_vector(orientation[:3], [1.0, 0.0, 0.0])
                col_dir = safe_normalize_vector(orientation[3:6], [0.0, 1.0, 0.0])
                
                # Calculate slice direction with safe cross product
                try:
                    slice_dir_raw = np.cross(row_dir, col_dir)
                    slice_dir = safe_normalize_vector(slice_dir_raw, [0.0, 0.0, 1.0])
                except:
                    slice_dir = np.array([0.0, 0.0, 1.0])
                
                # Create direction matrix
                direction = np.column_stack([row_dir, col_dir, slice_dir])
                
                # Validate direction matrix
                try:
                    if np.linalg.matrix_rank(direction) == 3 and np.allclose(np.linalg.det(direction), 1.0, atol=0.1):
                        pet_image.SetDirection(direction.flatten())
                        print(f"   PET Direction matrix set from DICOM orientation")
                    else:
                        print(f"   Invalid direction matrix (rank={np.linalg.matrix_rank(direction)}, det={np.linalg.det(direction):.3f}). Using identity.")
                        pet_image.SetDirection(np.eye(3).flatten())
                except:
                    print(f"   Direction matrix validation failed. Using identity.")
                    pet_image.SetDirection(np.eye(3).flatten())
            else:
                print(f"   Insufficient orientation data. Using identity matrix.")
                pet_image.SetDirection(np.eye(3).flatten())
        else:
            print(f"   No orientation data found. Using identity matrix.")
            pet_image.SetDirection(np.eye(3).flatten())
        
        print(f"   PET image loaded: {pet_image.GetSize()} voxels")
        print(f"   Value range: {sitk.GetArrayFromImage(pet_image).min():.2f} to {sitk.GetArrayFromImage(pet_image).max():.2f}")
        
        # Extract registration transform if available
        registration_transform = None
        if reg_dataset:
            print("   Processing registration transformation...")
            registration_transform = create_transform_from_reg(reg_dataset)
            if registration_transform:
                print("   Registration transform created successfully")
            else:
                print("   No valid registration transform found")
        
        return pet_image, registration_transform
        
    except Exception as e:
        print(f"Error loading PET data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def sample_pet_at_structures_with_transforms(pet_image, structures_dict, dp_struct, reg_transform=None):
    """
    Sample PET values at structure locations with proper coordinate transformations
    Uses improved coordinate handling and transformation chain
    """
    if pet_image is None:
        print("No PET image available for sampling.")
        return {}
    
    print("\nSampling PET values at structure locations with coordinate transforms...")
    print(f"PET Image Properties:")
    print(f"  Size: {pet_image.GetSize()}")
    print(f"  Spacing: {pet_image.GetSpacing()}")
    print(f"  Origin: {pet_image.GetOrigin()}")
    
    pet_stats = {}
    
    # Get structure frame of reference
    struct_frame_ref = getattr(dp_struct.ds, 'FrameOfReferenceUID', None)
    print(f"Structure Frame of Reference: {struct_frame_ref}")
    
    for sid, struct in structures_dict.items():
        try:
            struct_name = struct['name']
            print(f"\nProcessing structure: {struct_name}")
            
            # Get structure coordinates using dicompylercore
            contour_data = dp_struct.GetStructureCoordinates(sid)
            if not contour_data:
                print(f"  No contour data found for {struct_name}")
                continue
            
            pet_values = []
            total_points = 0
            valid_points = 0
            coord_debug_count = 0
            
            # Process each slice/contour 
            for slice_key, coordinate_data in contour_data.items():
                try:
                    points = []
                    
                    # Parse coordinate data into 3D points
                    if isinstance(coordinate_data, str):
                        # Parse coordinate string "x1 y1 z1 x2 y2 z2 ..."
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
                    
                    # Process each point with proper coordinate transformation
                    for point_idx, point in enumerate(points):
                        total_points += 1
                        try:
                            # Structure coordinates are in DICOM patient coordinate system (mm)
                            dicom_point = [safe_float_from_dicom(point[0]), 
                                         safe_float_from_dicom(point[1]), 
                                         safe_float_from_dicom(point[2])]
                            
                            # Start with the original DICOM coordinates
                            current_point = dicom_point.copy()
                            
                            # Apply registration transform if available
                            if reg_transform is not None:
                                try:
                                    # Apply the registration transform
                                    transformed_point = reg_transform.TransformPoint(current_point)
                                    current_point = list(transformed_point)
                                    
                                    # Debug first few transformations
                                    if coord_debug_count < 3:
                                        print(f"    Transform debug {coord_debug_count + 1}: "
                                              f"DICOM({dicom_point[0]:.1f}, {dicom_point[1]:.1f}, {dicom_point[2]:.1f}) -> "
                                              f"Transformed({current_point[0]:.1f}, {current_point[1]:.1f}, {current_point[2]:.1f})")
                                        coord_debug_count += 1
                                        
                                except Exception as transform_error:
                                    current_point = dicom_point.copy()
                                    if coord_debug_count < 3:
                                        print(f"    Transform failed: {transform_error}")
                                        coord_debug_count += 1
                            
                            try:
                                index = pet_image.TransformPhysicalPointToIndex(current_point)
                                size = pet_image.GetSize()
                                
                                if (0 <= index[0] < size[0] and 
                                    0 <= index[1] < size[1] and 
                                    0 <= index[2] < size[2]):
                                    
                                    pet_value = pet_image.GetPixel(index)

                                    pet_values.append(pet_value)
                                    valid_points += 1
                                    
                                    if valid_points <= 3:
                                        print(f"    Sample {valid_points}: Physical({current_point[0]:.1f}, {current_point[1]:.1f}, {current_point[2]:.1f}) -> "
                                              f"Index({index[0]}, {index[1]}, {index[2]}) -> PET={pet_value:.2f}")
                                else:
                                    if total_points <= 5:  
                                        print(f"    Point {total_points} out of bounds: Physical({current_point[0]:.1f}, {current_point[1]:.1f}, {current_point[2]:.1f}) -> "
                                              f"Index({index[0]}, {index[1]}, {index[2]}) vs Size({size[0]}, {size[1]}, {size[2]})")
                                    
                            except Exception as coord_error:
                                if total_points <= 5:
                                    print(f"    Coord transform failed for point {total_points}: {current_point} -> {coord_error}")
                                continue
                                
                        except Exception as point_error:
                            if total_points <= 5:  # Debug first few parsing errors
                                print(f"    Point parsing error {total_points}: {point} -> {point_error}")
                            continue
                            
                except Exception as slice_error:
                    print(f"  Warning: Could not process slice {slice_key}: {slice_error}")
                    continue
            
            sampling_efficiency = valid_points / total_points if total_points > 0 else 0
            print(f"  Sampling results: {valid_points}/{total_points} points ({sampling_efficiency:.1%} efficiency)")
            
            if pet_values and len(pet_values) > 0:
                pet_values = np.array(pet_values)
                
                q99 = np.percentile(pet_values, 99)
                q01 = np.percentile(pet_values, 1)
                filtered_values = pet_values[(pet_values >= q01) & (pet_values <= q99)]
                
                output_dir = "dvh_data_combined/pet_samples"
                os.makedirs(output_dir, exist_ok=True)
                raw_csv_path = os.path.join(output_dir, f"{struct_name.replace(' ', '_')}_raw_values.csv")
                with open(raw_csv_path, "w", newline="") as raw_file:
                    writer = csv.writer(raw_file)
                    writer.writerow(["Sample_Index", "PET_Value_Bq_ml"])
                    for idx, val in enumerate(pet_values):
                        writer.writerow([idx, val])
                print(f"    Raw PET values saved to: {raw_csv_path}")

                # Save statistics if any valid values
                if len(filtered_values) > 0:
                    pet_stats[struct_name] = {
                        'mean_bqml': float(np.mean(pet_values)),
                        'max_bqml': float(np.max(pet_values)),
                        'min_bqml': float(np.min(pet_values)),
                        'std_bqml': float(np.std(pet_values)),
                        'median_bqml': float(np.median(pet_values)),
                        'percentile_95_bqml': float(np.percentile(pet_values, 95)),
                        'percentile_05_bqml': float(np.percentile(pet_values, 5)),
                        'n_samples': len(pet_values),
                        'n_nonzero_samples': int(np.sum(pet_values > 0)),
                        'sampling_efficiency': sampling_efficiency,
                        'total_points_processed': total_points
                    }

                    print(f"  PET Statistics: ...")
                else:
                    print(f"  No valid PET values after filtering for {struct_name}")
            else:
                print(f"  No valid PET values found for {struct_name}")
                print(f"    Check coordinate system alignment between PET and structures")
                
                # Provide diagnostic information
                if total_points > 0:
                    print(f"    Processed {total_points} contour points but none were inside PET image bounds")
                    print(f"    This suggests a coordinate system mismatch")
        
        except Exception as e:
            print(f"Error sampling PET for {struct['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nPET sampling completed. Found statistics for {len(pet_stats)} structures.")
    return pet_stats

def normalize_name(name):
    """Normalize structure names for consistent matching"""
    return name.replace("_", " ").replace("-", " ").replace("/", " ").strip().lower()
