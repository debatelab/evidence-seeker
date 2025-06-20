#!/usr/bin/env python3

from os import path, makedirs
import sys
from loguru import logger
import argparse
import shutil  # For copying files
import importlib.resources as pkg_resources
from pathlib import Path
import asyncio

from evidence_seeker import (
    IndexBuilder,
)
from evidence_seeker import EvidenceSeeker


_PACKAGE_DATA_MODULE = "evidence_seeker.package_data"
_DEFAULT_NAME = "my-evidence-seeker"
_BASE_DIR = "."

_CONFIG_DIR_NAME = "config"
_RETRIEVAL_CONFIG_FILE_NAME = "retrieval_config.yaml"
_PREPROCESSING_CONFIG_FILE_NAME = "preprocessing_config.yaml"
_CONFIRMATION_ANALYSIS_CONFIG_FILE_NAME = "confirmation_analysis_config.yaml"

_KNOWLEDGE_BASE_DIR_NAME = "knowledge_base"
_SCRIPTS_DIR_NAME = "scripts"
_INDEX_DIR_NAME = "embeddings"
_LOGS_DIR_NAME = "logs"

# Remove the default handler
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format=(
        "<white>{time:DD.MM.YYYY}</white> <level>{level} | {message}</level>"
    )
)
logger.add(f"{_LOGS_DIR_NAME}/evse_cli.log",
           rotation="10 MB",
           retention="10 days",)


def main():
    parser = argparse.ArgumentParser(description="EvidenceSeeker CLI tool.")
    parser.add_argument(
        "action",
        choices=['init', 'build-index', 'run'],
        help="Action to perform")
    parser.add_argument(
        "-n", "--name",
        default=_DEFAULT_NAME,
        help="Name of the evidence seeker instance"
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Input string for running the pipeline (via 'evse run')"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="File for persisting the results of the  'run' action"
    )

    args = parser.parse_args()

    if args.action == "init":
        name = args.name.strip()
        if name == _DEFAULT_NAME:
            logger.warning(
                f"Using default name '{_DEFAULT_NAME}'. "
                "Consider specifying a custom name "
                "with -n or --name during `evse init`. "
            )
        target_directory = path.join(_BASE_DIR, name)

        logger.info(
            f"Initializing directory structure in {target_directory}..."
        )
        _init_directory_structure(target_directory)
        logger.info(
            "Copying default configs and scripts ..."
        )
        _copy_package_data(target_directory)
        logger.info("Initialization complete.")
    elif args.action == "build-index":
        _build_index()
    elif args.action == "run":
        input_string = args.input.strip()
        output_file = args.output.strip() if args.output else None
        if not input_string:
            logger.error(
                "No input string provided. Use -i or --input to specify input."
            )
            sys.exit(1)
        logger.info(f"Running pipeline with input: {input_string}")
        _run_pipeline(input_string, output_file)
    else:
        parser.print_help()


def _init_directory_structure(target_directory: str = "."):

    directories = [
        _CONFIG_DIR_NAME,
        _KNOWLEDGE_BASE_DIR_NAME,
        _KNOWLEDGE_BASE_DIR_NAME + "/data_files",
        _SCRIPTS_DIR_NAME,
        _INDEX_DIR_NAME,
        _LOGS_DIR_NAME,
    ]
    makedirs(target_directory, exist_ok=True)

    for dir_name in directories:
        dir_name = path.join(target_directory, dir_name)
        if not path.exists(dir_name):
            makedirs(dir_name)
            logger.info(f"Created directory: {dir_name}")
        else:
            logger.info(f"Directory already exists: {dir_name}")


def _copy_package_data(target_directory: str = "."):
    """
    Copies package data files to the specified base directory.
    """
    package_data_files = [
        "config/preprocessing_config.yaml",
        "config/retrieval_config.yaml",
        "config/confirmation_analysis_config.yaml",
        "config/api_keys.txt"
    ]

    for file_name in package_data_files:
        try:
            source_path = pkg_resources.files(
                _PACKAGE_DATA_MODULE
            ).joinpath(file_name)
            target_path = path.join(target_directory, file_name)
            # Copy the file
            if not path.exists(target_path):
                shutil.copy(source_path, target_path)
                logger.info(f"Copied {file_name} to {target_path}")
            else:
                logger.info(f"File already exists: {target_path}")
        except FileNotFoundError:
            logger.warning(
                f"Source file '{source_path}' not found in package data"
                f" module '{_PACKAGE_DATA_MODULE}'. Skipping.")
        except Exception as e:
            logger.error(
                f"Error copying '{source_path}' to '{target_path}': {e}"
            )


def _build_index(base_dir: str = "."):

    config_file = Path(base_dir, _CONFIG_DIR_NAME, _RETRIEVAL_CONFIG_FILE_NAME)

    if not config_file.exists():
        logger.error(
            f"Configuration file '{config_file}' does not exist. "
            "Please run 'evse init' first, and call this command from the "
            "root directory of your evidence seeker instance."
        )
        sys.exit(1)

    index_builder = IndexBuilder.from_config_file(config_file)
    index_builder.build_index()


def _run_pipeline(
        input_string: str,
        output_file: str = None,
        base_dir: str = "."
):

    config_dir = path.join(base_dir, _CONFIG_DIR_NAME)
    pipeline = EvidenceSeeker(
        retrieval_config_file=path.join(
            config_dir, _RETRIEVAL_CONFIG_FILE_NAME
        ),
        confirmation_analysis_config_file=path.join(
            config_dir, _CONFIRMATION_ANALYSIS_CONFIG_FILE_NAME
        ),
        preprocessing_config_file=path.join(
            config_dir, _PREPROCESSING_CONFIG_FILE_NAME
        )
    )

    results = asyncio.run(pipeline(input_string))

    if results:
        logger.info("Pipeline executed successfully. Results:")
        results_str = describe_results(input_string, results)
        # write results to a file
        if output_file is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = path.join(
                base_dir,
                _LOGS_DIR_NAME,
                f"results_{timestamp}.md"
            )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(results_str)
        logger.info(f"Results written to {output_file}")
    else:
        logger.warning(
            "No results returned from the pipeline. "
            "Please check your input and configuration."
        )


def describe_results(claim: str, results: list) -> str:
    preamble_template = (
        '## EvidenceSeeker Results\n\n'
        '### Input\n\n'
        '**Submitted claim:** {claim}\n\n'
        '### Results\n\n'
    )
    result_template = (
        '**Clarified claim:** <font color="orange">{text}</font> [_{statement_type}_]\n\n'
        '**Status**: {verbalized_confirmation}\n\n'
        '|Metric|Value|\n'
        '|:---|---:|\n'
        '|Average confirmation|{average_confirmation:.3f}|\n'
        '|Evidential divergence|{evidential_uncertainty:.3f}|\n'
        '|Width of evidential base|{n_evidence}|\n\n'
    )
    markdown = []
    markdown.append(preamble_template.format(claim=claim))
    for claim_dict in results:
        rdict = claim_dict.copy()
        rdict["statement_type"] = rdict["statement_type"].value
        markdown.append(result_template.format(**claim_dict))
    return ("\n".join(markdown))

if __name__ == "__main__":
    main()
