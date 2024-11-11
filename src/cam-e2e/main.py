from sklearn.neighbors import NearestNeighbors
import numpy as np
import argparse, pathlib
import matplotlib.pyplot as plt

project_path = pathlib.Path(__file__).parents[2]

def knn(k, x):
    x = x.reshape(-1, x.shape[-1])
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(x)
    res = nbrs.kneighbors(x)
    return tuple(i[..., 1:] for i in res)

def knn_cli(args):
    arr = np.load(args.input_file)
    distances, indices = knn(args.k, arr)
    np.save(args.output_file, indices)

def knn_parser(parser):
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--k", default=15, type=int)
    parser.set_defaults(call=knn_cli)

def flatten_cli(args):
    arr = np.load(args.input_file)
    arr = arr.reshape(-1, arr.shape[-1])
    np.save(args.output_file, arr)

def flatten_parser(parser):
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.set_defaults(call=flatten_cli)

def plot_cli(args):
    arr = np.load(args.knn_file)
    plt.hist(arr[:, -1], 200)
    plt.show()

def plot_parser(parser):
    parser.add_argument("knn_file", help="reformatted knn file from umap-cam")
    parser.set_defaults(call=plot_cli)

helper = lambda parser: lambda args: parser.parse_args(["-h"])

def main(parser):
    subparsers = parser.add_subparsers()
    knn_parser(subparsers.add_parser("knn"))
    flatten_parser(subparsers.add_parser("flatten"))
    plot_parser(subparsers.add_parser("plot"))
    parser.set_defaults(call=helper(parser))
    args = parser.parse_args()
    return args.call(args)

if __name__ == "__main__":
    main(argparse.ArgumentParser())
