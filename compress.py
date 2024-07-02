from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
import sys
import os
from pathlib import Path
from scene import GaussianModel

# Load and quantise ply by applying the quantisation scheme
def quantise(gaussian_model, path=None):
    print("Begin of k-means")
    gaussian_model.produce_clusters(store_dict_path=path)
    gaussian_model.apply_clustering(gaussian_model._codebook_dict)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Compresses a ply by applying a k-means clustering \
                            and optionally half float quantisation")
    parser.add_argument("--path", "-p", help="Path of the ply file to compress. Should be an unquantised one", type=str, required=True)
    parser.add_argument("--half_float", "-hf", help="Additionally apply half float quantisation", action="store_true")
    parser.add_argument("--store_codebook_path", help="If set, it will save the produced k-means\
                        codebook at the given directory", type=str, default=None)
    parser.add_argument("--max_sh_order", help="Max order of SH bands. 0 if just DC colour"
                        "1 if 3 coefficients per channel etc", type=int, default=3)
    
    args = parser.parse_args(sys.argv[1:])
    gaussian_model = GaussianModel(sh_degree=args.max_sh_order, variable_sh_bands=False)
    print("Loading ply")
    gaussian_model.load_ply(args.path)

    quantise(gaussian_model, path=args.store_codebook_path)

    ply_name = "point_cloud_quantised"
    if args.half_float:
        ply_name += "_half"
    ply_name += ".ply"
    gaussian_model.save_ply(os.path.join(Path(args.path).parent / Path(ply_name)), True, args.half_float)


    print("Done")
    
