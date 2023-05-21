import argparse
import json

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)  # positional argument
parser.add_argument("--project", type=str)  # positional argument

args = parser.parse_args()

wandb.init(
    project=args.project,
    job_type="test",
    group="test",
    name=f"{args.name}_test",  # "laxtnn_alm_baseline_test",
)

step_axis = "Sequence Length"
wandb.define_metric(step_axis)
# define which metrics will be plotted against it
wandb.define_metric("perplexity", step_metric=step_axis)
wandb.define_metric("loss", step_metric=step_axis)


with open(f"{args.name}.test.json") as f:

    data = json.load(f)

for k, v in data.items():
    log_dict = {step_axis: int(k)}
    log_dict.update(v)
    wandb.log(log_dict)


wandb.finish()
