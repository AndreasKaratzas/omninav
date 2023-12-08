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

from src.args import get_options, info, load_args, store_args
from src.agent import Agent
from src.deterministic import set_deterministic, set_seed
from src.metrics import MetricLogger
from src.nvidia import cuda_check
from src.utils import create_data_dir, get_chkpt, experiment_data_plots
from src.engine import Engine


if __name__ == "__main__":
    # Parse command line arguments
    args = get_options()
    
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

    # Seed random number generators
    set_seed(args.seed)

    # Install logger
    coloredlogs.install(
        level='INFO', fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

    # Initialize output data directory
    export_data_dir = Path(args.data_dir)

    # Initialize output data filepaths
    model_save_dir, memory_save_dir, log_save_dir, datetime_tag = create_data_dir(
        export_data_dir)
    
    # Store arguments in a dictionary
    if args.store:
        if args.verbose:
            print(f"Exporting options to: {log_save_dir}")
        store_args(Path(log_save_dir) / Path(args.store + '.json'), args)
    
    # Sync arguments to file
    if args.sync:
        if args.verbose:
            print(f"Syncing options with: {log_save_dir}")
        load_args(Path(log_save_dir) / Path(args.sync + '.json'), args)

    # Get checkpoint
    model_checkpoint, mem_checkpoint = get_chkpt(args.resume)

    # Configure render mode
    if args.render:
        render_mode = "human" 
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "rgb_array"

    # Build environment
    env = gym.make("highway-fast-v0", render_mode=render_mode)
    # Apply Wrappers to environment
    if args.en_cnn:
        env.unwrapped.configure({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
            "lanes_count": 4,
            "duration": 60,
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
            "duration": 60,
            "high_speed_reward": 10,
            "right_lane_reward": 0.4,
            # "collision_reward": -5,
        })
    # Reset environment
    env.reset(seed=args.seed)
    # Create Rainbow agent
    agent = Agent(
        env=env, batch_size=args.batch_size, target_sync=args.target_sync, gamma=args.gamma,
        num_of_steps_to_checkpoint_model=args.num_of_steps_to_checkpoint_model, beta=args.beta, 
        mem_capacity=args.mem_capacity, alpha=args.alpha, n_step=args.n_step, v_max=args.v_max,
        v_min=args.v_min, n_atoms=args.n_atoms, learning_rate=args.learning_rate, episodes=args.episodes,
        model_save_dir=model_save_dir, memory_save_dir=memory_save_dir, model_checkpoint=model_checkpoint,
        mem_checkpoint=mem_checkpoint, clip_grad_norm=args.clip_grad_norm, topk=args.top_k, verbose=args.verbose, 
        learning_starts=args.learning_starts, num_hiddens=args.num_hiddens, device=args.device,
        prior_eps=args.prior_eps, num_of_steps_to_checkpoint_memory=args.num_of_steps_to_checkpoint_memory,
        en_cnn=args.en_cnn)

    # Declare a logger instance to output experiment data
    logger = MetricLogger(log_save_dir, args.name, args.wandb, args.tensorboard, args.verbosity)

    # Log machine details
    logger.log_env_info()

    # Fire up engine
    Engine(
        env=env, agent=agent, logger=logger, en_train=True,
        en_visual=args.render, episodes=args.episodes, 
        verbosity=args.verbosity,
    )

    # Plot log data
    experiment_data_plots(export_data_dir, datetime_tag)
