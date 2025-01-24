#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2025-01-02 19:20:05
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import os
import json
import sys
from pathlib import Path
import glob
from argparse import ArgumentParser


def merge_json_files(input_files: list[str], output_file: str) -> None:
    """Merge multiple JSON files into one.
    All input files must have the same top-level type (either list or dict).

    Args:
        input_files: List of input JSON file paths
        output_file: Output JSON file path
    """
    merged_data: list | dict | None = None

    # Expand glob patterns and flatten the list
    expanded_files = []
    for pattern in input_files:
        expanded_files.extend(glob.glob(pattern))

    if not expanded_files:
        print("Error: No input files found")
        sys.exit(1)

    for input_file in expanded_files:
        if not os.path.exists(input_file):
            print(f"Error: File {input_file} does not exist")
            sys.exit(1)

        try:
            with open(input_file, "r") as f:
                data = json.load(f)

                # Check and set top-level type
                current_type = (
                    "list" if isinstance(data, list) else "dict" if isinstance(data, dict) else None
                )
                if current_type is None:
                    print(f"Error: File {input_file} must contain either a list or dict at top level")
                    sys.exit(1)

                if merged_data is None:
                    merged_data = [] if current_type == "list" else {}
                elif current_type != merged_data.__class__.__name__:
                    print(
                        f"Error: File {input_file} has different top-level type ({current_type}) than previous files ({merged_data.__class__.__name__})"
                    )
                    sys.exit(1)

                # Merge based on type
                if isinstance(merged_data, list):
                    merged_data.extend(data)
                else:  # dict
                    merged_data.update(data)

        except json.JSONDecodeError:
            print(f"Error: File {input_file} is not a valid JSON file")
            sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write merged data to output file
    with open(output_file, "w") as f:
        json.dump(merged_data, f)


def main():
    parser = ArgumentParser(
        description="Merge multiple JSON files with the same top-level type (list or dict)"
    )
    parser.add_argument("input", nargs="+", help="Input JSON files (glob patterns supported)")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")

    args = parser.parse_args()
    merge_json_files(args.input, args.output)
    print(f"Successfully merged files into {args.output}")


if __name__ == "__main__":
    main()
