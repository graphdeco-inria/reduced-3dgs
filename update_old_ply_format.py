from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
import sys
import numpy as np
import math
from pathlib import Path

# Creates a list with the ply names of all the essential fields
# that describe the 3d primitives
def construct_list_of_attributes(sh_band_order):
    return ['x', 'y', 'z',
            'f_dc_0','f_dc_1','f_dc_2',
            *[f"f_rest_{i}" for i in range(3*((sh_band_order+1)**2 - 1))],
            'opacity',
            'scale_0','scale_1','scale_2',
            'rot_0','rot_1','rot_2','rot_3']

# Unused fields of the previous version
unused_fields = ['nx', 'ny', 'nz']

# Automatically detects the number of SH bands of an old 
# format (with unused normals) pcd
def infer_max_sh_order(vertex_group):
    f_rest_count = len(vertex_group.properties) - len(construct_list_of_attributes(0)) - len(unused_fields)
    n_bands = math.log(f_rest_count/3 + 1, 2)
    if f_rest_count % 3 != 0 or not n_bands.is_integer():
        raise Exception(f"Invalid num of sh bands")
    return int(n_bands) - 1

def validate_vertex_group(vertex_group, sh_band_order):
    for field in construct_list_of_attributes(sh_band_order):
        if field not in vertex_group:
            print(f"Required field {field} missing in vertex_group of {sh_band_order} band", file=sys.stderr)
            return False
    return True

def validate_old_format(plydata, max_sh_order=None):
    vertex_group = plydata.elements[0]

    if max_sh_order is None:
        max_sh_order = infer_max_sh_order(vertex_group)

    if not validate_vertex_group(vertex_group, max_sh_order):
        return False
    return True

def validate_new_format(plydata):
    if 'vertex_0' not in plydata:
        print(f"Not compliant with new format", file=sys.stderr)
        return False
    for idx, vertex_group in enumerate(plydata.elements, 0):
        if not vertex_group.name.startswith('vertex_'):
            continue
        if not validate_vertex_group(vertex_group, idx):
            return False
    return True

# Converts an old format pcd to a new one
# that has one pcd per SH band
def convert_ply(plydata, path, max_sh_order=None):
    vertex_group = plydata.elements[0]
    if max_sh_order is None:
        max_sh_order = infer_max_sh_order(vertex_group)
    
    elements_list = []
    for sh_band_order in range(max_sh_order+1):
        coeff_num = (sh_band_order+1)**2 - 1

        # Just store zero-sized tensors
        if sh_band_order != max_sh_order:
            n_primitives = 0
            xyz = np.empty((0, 3))
            f_dc = np.empty((0, 3))
            f_rest = np.empty((0, 3*coeff_num))
            opacities = np.empty((0, 1))
            scales = np.empty((0, 3))
            rots = np.empty((0, 4))
        else:
            # Read the parameters
            n_primitives = vertex_group.count
            xyz = np.stack((np.asarray(vertex_group["x"]),
                            np.asarray(vertex_group["y"]),
                            np.asarray(vertex_group["z"])), axis=1)

            f_dc = np.stack((np.asarray(vertex_group["f_dc_0"]),
                             np.asarray(vertex_group["f_dc_1"]),
                             np.asarray(vertex_group["f_dc_2"])), axis=1)

            extra_f_names = [p.name for p in vertex_group.properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

            f_rest = np.zeros((n_primitives, len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                f_rest[:, idx] = np.asarray(vertex_group[attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            f_rest = f_rest.reshape((n_primitives, 3 * ((max_sh_order + 1) ** 2 - 1)))

            opacities = np.asarray(vertex_group["opacity"])[..., np.newaxis]

            scales = np.stack((np.asarray(vertex_group["scale_0"]),
                               np.asarray(vertex_group["scale_1"]),
                               np.asarray(vertex_group["scale_2"])), axis=1)

            rots = np.stack((np.asarray(vertex_group["rot_0"]),
                             np.asarray(vertex_group["rot_1"]),
                             np.asarray(vertex_group["rot_2"]),
                             np.asarray(vertex_group["rot_3"])), axis=1)
        
        # Create an PlyElement with the vertex group
        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(sh_band_order)]

        elements = np.empty(n_primitives, dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        elements_list.append(PlyElement.describe(elements, f'vertex_{sh_band_order}'))

    PlyData(elements_list).write(path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Checks the compliance of the point cloud with"
                            "the new format and converts it if necessary")
    parser.add_argument("--path", "-p", help="Path of the ply file to convert", type=str, required=True)
    parser.add_argument("--name", "-n", help="Name of the newly created point cloud", type=str, required=True)
    parser.add_argument("--max_sh_order", help="Max order of SH bands. 0 if just DC colour"
                        "1 if 3 coefficients per channel etc", type=int, default=3)
    args = parser.parse_args(sys.argv[1:])
    plydata = PlyData.read(args.path)
    if validate_new_format(plydata):
        print("Compliant with new format")
    elif validate_old_format(plydata, max_sh_order=args.max_sh_order):
        print("Begin conversion")
        name = args.name
        if not name.endswith('.ply'):
            name += '.ply'
        convert_ply(plydata, Path(args.path).parent / Path(name), max_sh_order=args.max_sh_order)
        print("Done")
    
