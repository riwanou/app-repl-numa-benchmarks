import ann.lib
import config
import argparse

TAG = "default"
NUM_THREADS = config.NUM_THREADS
DATA_DIR = "ann/data"
INDEX_DIR = "ann/indices"
RESULT_DIR = config.RESULT_DIR_ANN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--threads", type=int, default=NUM_THREADS, help="Number of threads"
)
parser.add_argument("--tag", default=TAG, help="CSV filename tag")
parser.add_argument(
    "--datasets",
    nargs="*",
    default=ann.lib.DATASETS,
    help="List of datasets to use (default: all datasets)",
)
parser.add_argument(
    "--recreate-index", action="store_true", help="Re create the indices"
)
parser.add_argument("--bench", action="store_true", help="Bench the ann search")
parser.add_argument(
    "--faiss", action="store_true", help="Evaluate faiss benchmark"
)
parser.add_argument(
    "--annoy", action="store_true", help="Evaluate annoy benchmark"
)
parser.add_argument(
    "--usearch", action="store_true", help="Evaluate usearch benchmark"
)
args = parser.parse_args()

ann.lib.run(
    DATA_DIR,
    INDEX_DIR,
    RESULT_DIR,
    args.datasets,
    args.faiss,
    args.annoy,
    args.usearch,
    args.bench,
    args.recreate_index,
    args.tag,
    args.threads,
)
