#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The main script for running a single replication.
"""
import errno
import json
import os
import sys
from typing import Any, Dict

import torch

from discrete_mixed_bo.run_one_replication import run_one_replication


def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Here function_name is independent from the "function_name" in the json file for get_problem
    function_name = sys.argv[1]
    exp_dir = os.path.join(current_dir, function_name)

    config_path = os.path.join(exp_dir, "config.json")
    label = sys.argv[2]
    seed = int(float(sys.argv[3]))
    last_arg = sys.argv[4] if len(sys.argv) > 4 else None
    output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")

    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    save_callback = lambda data: torch.save(data, output_path)
    save_frequency = 5
    fetch_data(kwargs=kwargs)
    run_one_replication(
        seed=seed,
        label=label,
        save_callback=save_callback,
        save_frequency=save_frequency,
        **kwargs,
    )
