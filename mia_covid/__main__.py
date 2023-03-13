from mia_covid.experiment import runExperiment
import argparse
from typing import Dict, Any


def parse_arguments():
    parser = argparse.ArgumentParser(description="Membership-Inference Attack Experimenting Tool", prog="mia-covid", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", choices=["mnist", "covid"], default="mnist", type=str, help="Dataset to be used")
    parser.add_argument("-m", "--model", choices=["resnet18", "resnet50"], default="resnet18", type=str, help="Model to be used")
    parser.add_argument("-e", "--eps", default=1, type=float, help="Epsilon value for experiments [None, 10, 1, 0.1] or any epsilon")
    parser.add_argument("-w", "--wandb", action="store_false", default=False)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # turn args namespace to dict
    arg_dict: Dict["str", Any] = vars(args)
    runExperiment(arg_dict)


if __name__ == "__main__":
    main()
