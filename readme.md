# Lift-H: Pick and Throw Task

Code for training and running a pick and throw task on a Franka Emika Panda robot in IsaacLab using reinforcement learning.

## Prerequisites

- Installed IsaacLab
- Installed IsaacSim

## Installation

1. Clone this repository
2. Copy the `lift_h` directory into your IsaacLab installation:
   ```
   IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/
   ```

## Usage

### Training

Run the training script with the following command:

```bash
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Lift-h-Cube-Franka-v0 --headless
```

### Testing

After training, test the performance using:

```bash
python scripts/reinforcement_learning/skrl/play.py --task Isaac-Lift-h-Cube-Franka-Play-v0 --checkpoint your_checkpoint
```

Replace `your_checkpoint` with the path to your trained model checkpoint.