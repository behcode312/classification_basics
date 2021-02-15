import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=int, default=0, help="device")
#you shouls calculate the given dataset weight_loss if the dataset is imbalanced
parser.add_argument(
    "--weight_loss",
    nargs=5,
    metavar=("wc1", "wc2", "wc3", "wc4", "wc5"),
    type=float,
    default=[7.64, 0.34, 4.24, 1.30, 1.06],
)
parser.add_argument("--data_dir", type=Path, default=Path("data"))
parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))

# arguments for distributed testing
parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
parser.add_argument("-nr", "--nr", default=0, type=int, help="ranking within the nodes")

args = parser.parse_args()
