# ==============================================================================
# Name    : Anay Abhijit Joshi
# Date    : November 22, 2026
# Course  : ISyE 6767 - Interim Project 2 (Virtual Bidding at ERCOT HB_NORTH)
# File    : main.py
# Purpose : One entry point. Every result of the project is regenerated from
#           the raw ERCOT files by:   python main.py --stage all
# ==============================================================================

import argparse
import time

import config
from data_pipeline import ErcotDatasetBuilder, load_hourly_dataset
from data_sanity import DataSanityInspector
from feature_factory import run_feature_stage
from final_evaluation import run_final_evaluation
from run_part1 import run_part1
from run_part2 import run_part2
from run_part3 import run_part3


def run_data_stage(force_rebuild):
    # So, the dataset is always rebuilt from the raw files; --rebuild also re-reads the price workbooks.
    dataset_builder = ErcotDatasetBuilder(use_cache=not force_rebuild)
    dataset_builder.run_full_pipeline()
    # Let's do now the sanity checks on the CSV exactly as it was saved (round trip included).
    DataSanityInspector(load_hourly_dataset()).run_all_checks()


def main():
    argument_parser = argparse.ArgumentParser(description="ISyE 6767 Interim Project 2 - ERCOT HB_NORTH DART trading")
    argument_parser.add_argument("--stage", default="data", choices=["data", "features", "part1", "part2", "part3", "evaluate", "all"])
    argument_parser.add_argument("--rebuild", action="store_true",
                                 help="re-read every raw workbook and refit every model (ignore all caches)")
    parsed_arguments = argument_parser.parse_args()

    config.ensure_output_folders()
    start_clock = time.perf_counter()
    if parsed_arguments.stage in ("data", "all"):
        run_data_stage(parsed_arguments.rebuild)
    if parsed_arguments.stage in ("features", "all"):
        run_feature_stage(use_cache=not parsed_arguments.rebuild)
    # So, --rebuild also refits every model instead of reusing the cached predictions.
    use_cache = not parsed_arguments.rebuild
    if parsed_arguments.stage in ("part1", "all"):
        run_part1(use_cache=use_cache)
        use_cache = True        # Let's do now: the models were just (re)fitted, later stages reuse them.
    if parsed_arguments.stage in ("part2", "all"):
        run_part2(use_cache=use_cache)
    if parsed_arguments.stage in ("part3", "all"):
        run_part3(use_cache=use_cache)
    if parsed_arguments.stage in ("evaluate", "all"):
        run_final_evaluation()
    print(f"\nFinal Conclusion: stage '{parsed_arguments.stage}' finished in {time.perf_counter() - start_clock:,.1f} s.")


if __name__ == "__main__":
    main()
