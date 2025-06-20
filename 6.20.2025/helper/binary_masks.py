from dicompylercore import dicomparser, dvhcalc
import numpy as np
from matplotlib.path import Path

def create_structure_masks(structures, rs_dataset, ct_datasets, volume_shape):
    """ Generates binary 3D masks for each ROI (structure) defined """
    rt = dicomparser.DicomParser(rs_dataset)
    first_ct = ct_datasets[0]
    origin = np.array(first_ct.ImagePositionPatient)
    spacing = list(map(float, first_ct.PixelSpacing))
    thickness = float(first_ct.SliceThickness)
    spacing.append(thickness)

    z_positions = sorted([float(ds.ImagePositionPatient[2]) for ds in ct_datasets])
    masks = {}

    for sid, struct in structures.items():
        coords = rt.GetStructureCoordinates(sid)
        if not coords:
            continue

        mask = np.zeros(volume_shape, dtype=bool)
        for z_str, contours in coords.items():
            z = float(z_str)
            try:
                k = np.argmin(np.abs(np.array(z_positions) - z))
            except:
                continue

            for contour in contours:
                pts = contour['data']
                if len(pts) < 3:
                    continue
                x = [(pt[0] - origin[0]) / spacing[0] for pt in pts]
                y = [(pt[1] - origin[1]) / spacing[1] for pt in pts]
                poly = Path(np.vstack((x, y)).T)
                grid_x, grid_y = np.meshgrid(np.arange(volume_shape[2]), np.arange(volume_shape[1]))
                points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
                mask2d = poly.contains_points(points).reshape((volume_shape[1], volume_shape[2]))
                mask[k] |= mask2d

        masks[sid] = mask
    return masks
