import argparse
from habitat.datasets import make_dataset
import sys

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLN_CE.vlnce_baselines.config.default import get_config
from navid_agent import evaluate_agent, VLM
import rospy

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset_split = dataset.get_splits(split_num)[split_id]
    evaluate_agent(config, split_id, dataset_split, model_path, result_path)
  


if __name__ == "__main__":
    rospy.init_node('vlm', anonymous=True)
    vlm = VLM()
    rate = rospy.Rate(0.25)  

    from pathlib import Path
    _THIS = Path(__file__).resolve()
    _PROJECT_ROOT = None
    for p in _THIS.parents:
        if (p / "model_zoo").exists() and (p / "navid").exists():
            _PROJECT_ROOT = p
            break
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = _THIS.parents[1]

    navid_model_path = _PROJECT_ROOT / "model_zoo" / "navid-7b-full-224-video-fps-1-grid-2-eva-encoder-09-10-previous-navid-r2r-rxr-0915-epoch5-20250915T060726Z-1-001"

    while not rospy.is_shutdown():
        vlm.evaluate_agent_CMU(str(navid_model_path), None)
        rate.sleep()


    
