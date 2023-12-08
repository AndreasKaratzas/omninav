# OmniNav

Rainbow - DQN Algorithm Applications Towards Self-Driving Cars

![Methodology](docs/methodology.png)

### Abstract

Decision making for self-driving cars is challenging due to the complex policies that govern such high-dimensional and multi-agent environments. Most methods rely on hard-coded rules assigned by the developers, thus yielding suboptimal agent policies. In this work, we present **OmniNav**, a Deep Reinforcement Learning agent that tackles the difficulty of learning an optimal policy by leveraging the Rainbow - DQN algorithm. Our framework constantly evaluates both a *value* and an *advantage* function using fully-connected networks. It employs NoisyNets to explore the intricate patterns of any environment without the need of a $\epsilon$ - greedy algorithm. We evaluate OmniNav on the [highway-v0](https://github.com/Farama-Foundation/HighwayEnv) environment to observe a $xx$ average agent return value with a response time of $yy$ seconds on a NVIDIA GTX 1660 Ti GPU.

### References

When using any of this project's source code, please cite:
```bibtex
@misc{karatzas2023omninav,
      title={OmniNav: Rainbow - DQN Algorithm Applications Towards Self-Driving Cars},
      author={Andreas Karatzas},
      year={2023},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/AndreasKaratzas/omninav}},
}
```

### Installation

```bash
conda env create --file environment.yml
conda activate omninav
```

If you either added or removed packages, then you can save a checkpoint of the `conda` environment by:
```bash
conda env export --no-builds > environment.yml
```

### Usage


### Results

To monitor with TensorBoard, use:

```bash
tensorboard --logdir=./data/datetime/ --host localhost --port 8888
```



