import argparse
import config
import trainer
import sweep
import feature_visualization
import helper
import visualize_saliency_maps

if __name__ == "__main__":
    helper.define_dataset_location()
    if config.TRAINER_CONFIG["mode"] == "training":
        trainer.run_trainer()
    elif config.TRAINER_CONFIG["mode"] == "sweeps":
        sweep.run_sweep()
    elif config.TRAINER_CONFIG["mode"] == "feature_visualization":
        feature_visualization.run_visualizer()
    elif config.TRAINER_CONFIG["mode"] == "saliency_maps":
        visualize_saliency_maps.run_saliency_visualizer()
    else:
        raise Exception("unsupported mode")