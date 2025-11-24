# Voxel4D: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving

[![Custom badge](https://img.shields.io/badge/Project-Page-blue)](https://rtennety.github.io/Voxel4D/)

> **Author:** Rohan Tennety

**Voxel4D: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving**

## Abstract
![teaser](assets/figures/teaser.png)

**4D Occupancy Forecasting and Planning via World Model**. Voxel4D takes observations and trajectories as input, incorporating flexible action conditions for **action-controllable generation**. By leveraging world knowledge and the generative capacity of the world model, I further integrate it with a planner for **continuous forecasting and planning**.


## Getting Started

- [Installation](DOCS/INSTALL.MD) 

- [Prepare Dataset](DOCS/DATASET.MD)

- [Train and Evaluation](DOCS/TRAIN_EVAL.MD)

## Demo of 4D Occupancy and Flow Forecasting

Voxel4D understands how the world evolves by accurately modeling the dynamics of movable objects and the future states of the static environment.

### Scene 1 (Lane Change)
<div style="text-align:center;">
    <img src="assets/figures/forecasting_1.gif" alt="Local GIF" width="600px" />
</div>

### Scene 2 (Pedestrian Crossing)
<div style="text-align:center;">
    <img src="assets/figures/forecasting_2.gif" alt="Local GIF" width="600px" />
</div>

### Scene 3 (Vehicle Following)
<div style="text-align:center;">
    <img src="assets/figures/forecasting_3.gif" alt="Local GIF" width="600px" />
</div>


## Demo of Continuous Forecasting and Planning (E2E Planning)

Voxel4D plans trajectories through forecasting future occupancy state and selecting optimal trajectory based on a comprehensive occupancy-based cost function.

### Scene 1 (Turn Left to Avoid Stopped Vehicle)
<div style="text-align:center;">
    <img src="assets/figures/planning_1.png" width="600px" />
</div>

<div style="text-align:center;">
    <img src="assets/figures/planning_1.gif" alt="Local GIF" width="300px" />
</div>

### Scene 2 (Slowing Down to Wait for Crossing Pedestrians)
<div style="text-align:center;">
    <img src="assets/figures/planning_2.png" width="600px" />
</div>

<div style="text-align:center;">
    <img src="assets/figures/planning_2.gif" alt="Local GIF" width="300px" />
</div>

### Scene 3 (Turn Right to Avoid Stopped Vehicle)
<div style="text-align:center;">
    <img src="assets/figures/planning_3.png" width="600px" />
</div>

<div style="text-align:center;">
    <img src="assets/figures/planning_3.gif" alt="Local GIF" width="300px" />
</div>


## Citation

If you use Voxel4D in an academic work, please cite my paper:

```bibtext
@inproceedings{tennety2024voxel4d,
  author = {Rohan Tennety},
  title = {{Voxel4D: Vision-Centric 4D Occupancy Forecasting and Planning via World Models for Autonomous Driving}},
  year = {2025}
}
```

## Contact

**Author:** Rohan Tennety  
**Email:** rtennety@gmail.com
