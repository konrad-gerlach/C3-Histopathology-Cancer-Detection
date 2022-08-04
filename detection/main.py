import argparse
import config
import trainer
import sweep
import feature_visualization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configure project')
    parser.add_argument('--ds_path', default=config.DATA_CONFIG["ds_path"],
                        help='the location where the dataset is or should be located')

    args = parser.parse_args()
    config.DATA_CONFIG["ds_path"] = args.ds_path
    print(config.DATA_CONFIG["ds_path"])
    
    if config.TRAINER_CONFIG["mode"] == "training":
        trainer.classifier()
    elif config.TRAINER_CONFIG["mode"] == "sweeps":
        sweep.run_sweep()
    elif config.TRAINER_CONFIG["mode"] == "feature_visualization":
        feature_visualization.visualizer()
    else:
        raise Exception("unsupported mode")