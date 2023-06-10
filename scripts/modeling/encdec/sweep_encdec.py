import wandb
import yaml

SWEEP_CONFIG_FILEPATH = "sweep_encdec.yaml"

with open(SWEEP_CONFIG_FILEPATH, "r") as fh:
    sweep_config = yaml.safe_load(fh)

print(yaml.dump(sweep_config))


sweep_id = wandb.sweep(sweep=sweep_config, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=4)
