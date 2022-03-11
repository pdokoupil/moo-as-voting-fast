import argparse
import mlflow
import itertools
import random
import time

from datetime import datetime

from main import main as experiment_entry
from main import * # We need to populate globals()

import numpy as np

def get_argument_combinations(args):
    for weights, shift, seed, normalization, algorithm, baseline, diversity in itertools.product(
        args.weights, args.shifts, args.seeds, args.normalizations, args.algorithms, args.baselines, args.diversities
    ):
        yield {
            "weights": weights,
            "shift": shift,
            "seed": seed,
            "normalization": normalization,
            "algorithm": algorithm,
            "baseline": baseline,
            "diversity": diversity,
            "experiment_name": args.experiment_name,
            "train_fold_path": args.train_fold_path,
            "test_fold_path": args.test_fold_path,
            "metadata_path": args.metadata_path,
            "cache_dir": ""
        }

def main(args):
    args_combinations = get_argument_combinations(args)
    num_args_combinations = len(args_combinations)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    for i, arg_combination in enumerate(args_combinations):
        if not args.no_cache:
            arg_combination.cache_dir = f""
        
        start_time = time.perf_counter()
        start_time_formated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"Starting next experiment ({i + 1} out of {num_args_combinations}) at time: {start_time_formated}")
        run_experiment(args.mlflow_project_path, args.experiment_name, arg_combination)
        print(f"Experiment {i + 1} took {time.perf_counter() - start_time}")

def run_experiment(project_path, experiment_name, args):
    job = mlflow.run(project_path, parameters=args, use_conda=False, experiment_name=experiment_name)
    job.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="moo-as-voting-fast")
    parser.add_argument("--weights")
    parser.add_argument("--shifts")
    parser.add_argument("--seeds")
    parser.add_argument("--normalizations")
    parser.add_argument("--algorithms")
    parser.add_argument("--train_fold_path")
    parser.add_argument("--test_fold_path")
    parser.add_argument("--metadata_path")
    parser.add_argument("--baselines")
    parser.add_argument("--diversities")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://gpulab.ms.mff.cuni.cz:7022")
    parser.add_argument("--output_path_prefix", type=str, default="/mnt/1/outputs")
    parser.add_argument("--mlflow_project_path", type=str, default="/mnt/1/MLproject")
    args = parser.parse_args()

    # Modify arguments to correct type and structure
    args.weights = [[float(x) for x in w.split(',')] for w in args.weights.split(';')]
    args.seeds = [int(s) for s in args.seeds.split(';')]
    args.shifts = [float(s) for s in args.shifts.split(';')]
    args.algorithms = [globals()[alg] for alg in args.algorithms.split(';')]
    args.normalizations = [globals()[norm] for norm in args.normalizations.split(';')]
    args.baselines = [globals()[baseline] for baseline in args.baselines.split(';')]
    args.diversities = [globals()[div] for div in args.diversities.split(';')]

    main(args)