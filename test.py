"""Driver script.
"""

import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['DISPLAY'] = 'localhost:10.0'
# os.environ['SDL_VIDEODRIVER'] = 'dummy'

import sys
sys.path.append('./')

import coloredlogs
import gymnasium as gym
import highway_env

from pathlib import Path
from gymnasium.wrappers.record_video import RecordVideo

from src.args import info, load_args, demo_args
from src.inference import Inference
from src.deterministic import set_deterministic, set_seed
from src.metrics import MetricLogger
from src.nvidia import cuda_check
from src.utils import get_chkpt
from src.engine import Engine


if __name__ == "__main__":
    # Parse command line arguments
    args = demo_args()
    
    # Print info
    if args.info:
        info()
    
    # check for CUDA compatible device
    cuda_check(True)

    # utilize deterministic algorithms for reproducibility and speed-up
    if args.use_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
        set_deterministic(seed=args.seed)

    # build full path for sync
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    args.sync = Path(current_file_path) / Path(args.sync)

    # Seed random number generators
    set_seed(args.seed)

    # Install logger
    coloredlogs.install(
        level='INFO', fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

    # Sync arguments to file
    if args.verbose:
        print(f"Syncing options with: {Path(args.sync) / 'log' / 'options.json'}")
    load_args(Path(args.sync) / 'log' / 'options.json', args)

    # Get checkpoint
    model_checkpoint, mem_checkpoint = get_chkpt(args.sync)

    # Configure render mode
    if args.render:
        render_mode = "human" 
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "rgb_array"

    # Build environment
    env = gym.make("highway-v0", render_mode=render_mode)
    # Apply Wrappers to environment
    env = RecordVideo(env, video_folder="run",
              episode_trigger=lambda e: True)
    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    env.unwrapped.set_record_video_wrapper(env)
    if args.en_cnn:
        env.unwrapped.configure({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (84, 84),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
            "lanes_count": 4,
            "vehicles_density": 1,
            "high_speed_reward": 10,
        })
    else:
        env.unwrapped.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"
            },
            "high_speed_reward": 10,
            "right_lane_reward": 0.4,
            # "collision_reward": -5,
        })
    env.unwrapped.configure({"policy_frequency": 15, "duration": 40})
    # Reset environment
    env.reset(seed=args.seed)
    # Create Rainbow agent
    agent = Inference(
        env=env, v_max=args.v_max, v_min=args.v_min, n_atoms=args.n_atoms, 
        model_checkpoint=model_checkpoint, verbose=args.verbose, 
        num_hiddens=args.num_hiddens, device=args.device, en_cnn=args.en_cnn)

    # Declare a logger instance to output experiment data
    logger = MetricLogger(Path(args.sync) / 'log', args.name, verbosity=args.verbosity)

    # Log machine details
    logger.log_env_info()

    # Fire up engine
    Engine(
        env=env, agent=agent, logger=logger, en_eval=True,
        en_visual=args.render, episodes=args.episodes, 
        verbosity=args.verbosity, 
    )
