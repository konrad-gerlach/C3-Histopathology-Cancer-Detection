import argparse
import config
import trainer
import sweep
import feature_visualization

if __name__ == "__main__":
    helper.define_dataset_location()
    if config.TRAINER_CONFIG["mode"] == "training":
        trainer.classifier()
    elif config.TRAINER_CONFIG["mode"] == "sweeps":
        sweep.run_sweep()
    elif config.TRAINER_CONFIG["mode"] == "feature_visualization":
        feature_visualization.visualizer()
    else:
        raise Exception("unsupported mode")