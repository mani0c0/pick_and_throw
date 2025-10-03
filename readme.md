# Lift-H: Pick and Throw Task

Code for training and running a pick and throw task on a Franka Emika Panda robot in IsaacLab using reinforcement learning.

## Prerequisites

- Installed IsaacLab
- Installed IsaacSim

## Installation

1. Clone this repository
2. Make a directory called `lift_h` in your IsaacLab installation (see below) and copy the contents into that directory:
   ```
   cp -r ./* /home/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_h/
   ```

## Demo Videos

Demonstration video showing the trained robot performing the pick and throw task:


https://github.com/user-attachments/assets/d25c312f-61a5-43a1-bbb0-75c2de267b3f



## Usage

### Training

Run the training script with the following command in your IsaacLab installation:

```bash
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Lift-h-Cube-Franka-v0 --headless
```

### Testing

After training, test the performance using:

```bash
python scripts/reinforcement_learning/skrl/play.py --task Isaac-Lift-h-Cube-Franka-Play-v0 --checkpoint your_checkpoint
```

Replace `your_checkpoint` with the path to your trained model checkpoint.
