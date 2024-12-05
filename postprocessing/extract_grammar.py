import argparse
from pathlib import Path

import torch
from torch_support.train_support import (
    get_config_from,
)
from torch_support.load_model import get_model_args
import torch_support.load_model as load_model


def main(config, model_path):
    model_args = get_config_from(config)
    path = Path(model_path)

    # Get rule distribution from model
    load_model.set_model_dir("parser/model")
    model = get_model_args(model_args.model, device="cuda:0")
    with path.open("rb") as f:
        chkpt = torch.load(f, map_location="cuda:0")
    model.load_state_dict(chkpt["model"], strict=False)
    model.eval()
    rules = model.forward()

    # Save
    for k, v in rules.items():
        p = path.parent / f"{k}.pt"
        with p.open("wb") as f:
            torch.save(v.detach().cpu(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="log/n_english_std_norig/NPCFG2022-12-27-17_31_28/config.yaml",
    )
    parser.add_argument(
        "--model_path",
        default="log/n_english_std_norig/NPCFG2022-12-27-17_31_28/last.pt",
    )
    args = parser.parse_args()

    main(args.config, args.model_path)
