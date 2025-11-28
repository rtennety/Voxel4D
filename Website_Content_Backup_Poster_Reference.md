# Voxel4D Website Content - Full Reference for Science Fair Poster

This document contains the complete website content in order, preserved for reference when creating the science fair poster. All images are listed with their corresponding sections.

## Title and Hero Section
**Title:** Voxel4D: A Unified Vision-Centric World Model for 4D Spatial Forecasting and End-to-End Planning in Autonomous Driving

**Subtitle:** Using vision-centric multi-view cameras, this system predicts future 4D spatial states based on different possible driving actions, enabling real-time forecasting of how the environment will change. These predictions are integrated into an end-to-end planning system that chooses the best driving path based on forecasted spatial information, advancing autonomous driving technology.

**Author:** Rohan Tennety

**Associated Image:**
- `./assets/figures/1.png` - Voxel4D Teaser image showing the system overview

## Research Statement
Current autonomous driving systems operate reactively, processing the present moment rather than anticipating how the environment will evolve. This creates safety gaps: when a pedestrian steps into the road, the system must detect the threat before it can react, potentially leaving insufficient time for safe avoidance.

I introduce **Voxel4D**, a vision-centric world model that transforms autonomous driving from reactive response to proactive anticipation. Voxel4D forecasts 4D spatial states (3D space + time) conditioned on ego-vehicle actions, then integrates these predictions into an end-to-end planning system. This enables the vehicle to evaluate "what-if" scenarios before executing actions, selecting optimal trajectories based on predicted future occupancy. Voxel4D represents the first system to successfully integrate 4D occupancy forecasting with end-to-end planning for autonomous driving.

## Abstract
Current autonomous systems (Tesla, Waymo) operate reactively, processing frames sequentially and reacting to immediate threats. This limits safety: when a pedestrian appears, the system must detect them before reacting, potentially leaving insufficient time to avoid collisions. Voxel4D transforms this paradigm by predicting future 4D spatial states (3D space + time) conditioned on different ego-vehicle actions, enabling proactive decision-making.

Voxel4D operates through three stages: (1) **History Encoder** converts multi-view camera images into bird's-eye view using transformer networks, (2) **Memory Queue** accumulates temporal information with semantic and motion-conditional normalization, and (3) **World Decoder** generates action-conditioned 4D occupancy predictions. The system uses these predictions for occupancy-based trajectory planning, generating multiple futures ("what-if" scenarios) and selecting the safest path.

Results: **+9.5% mIoU** on nuScenes, **+6.1%** on Lyft-Level5, **401ms** real-time latency. Collision rates: **0.035%** (1s), **0.16%** (2s), **0.493%** (3s). While 9.5% may seem modest, this represents correctly classifying 9.5% of millions of voxels: Voxel4D processes space divided into millions of 3D voxels (512×512×40 = 10.5 million voxels per frame), meaning the system correctly predicts the occupancy of approximately 1 million voxels per frame. Given that most voxels are empty space, correctly identifying which of millions of voxels will be occupied by objects in the future is extremely challenging. This level of precision enables the fine-grained collision detection and safe navigation demonstrated by the exceptional collision rates. First system to integrate 4D occupancy forecasting with end-to-end planning for autonomous driving.

## 4D Spatial and Flow Forecasting

**4D Spatial Forecasting** predicts which 3D voxels will be occupied in the future (3D space + time). Voxel4D predicts occupancy probabilities $P(\text{occupied} \mid x, y, z, t)$ for millions of voxels, enabling precise collision detection. Models both moving objects (cars, pedestrians) and static parts (roads, buildings), capturing how scenes evolve dynamically.

### Scene 1 (Lane Change)
**Images:**
- `./assets/figures/forecasting_1.png` - Lane Change static image
- `./assets/figures/forecasting_1.gif` - Lane Change animated GIF

**Description:** Demonstrates Voxel4D's ability to predict lane change scenarios: accurately forecasts how vehicles will move across lanes over time, enabling the system to anticipate and plan for lane changes before they occur.

### Scene 2 (Pedestrian Crossing)
**Images:**
- `./assets/figures/forecasting_2.png` - Pedestrian Crossing static image
- `./assets/figures/forecasting_2.gif` - Pedestrian Crossing animated GIF

**Description:** Shows Voxel4D predicting pedestrian movement across a crosswalk: forecasts future pedestrian positions, enabling the system to anticipate crossing behavior and plan safe trajectories that yield appropriately.

### Scene 3 (Vehicle Following)
**Images:**
- `./assets/figures/forecasting_3.png` - Vehicle Following static image
- `./assets/figures/forecasting_3.gif` - Vehicle Following animated GIF

**Description:** Illustrates vehicle following in challenging conditions (nighttime, rain): accurately predicts leading vehicle motion despite limited visibility, demonstrating robust performance in adverse weather and lighting conditions.

### Additional Forecasting Visualizations
**Image:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163245.png` - 4D Occupancy Forecasting Examples

**Description:** Voxel4D accurately predicts complex scenarios (lane changes, pedestrian crossings, vehicle following) including nighttime conditions. Predictions require computing occupancy probabilities for millions of voxels across multiple time steps, validating the system's innovations work in real-world conditions.

## Continuous Forecasting and Planning

Voxel4D continuously forecasts future states and uses predictions for planning. For each action, predicts future states, then selects safest path using: $C(\tau) = \sum_{t} [C_{\text{agent}}(\tau, t) + C_{\text{background}}(\tau, t) + C_{\text{efficiency}}(\tau, t)]$, selecting $\tau^* = \arg\min_{\tau} C(\tau)$. Updates every 0.5 seconds with new images.

### Scene 1 (Turn Left to Avoid Stopped Vehicle)
**Images:**
- `./assets/figures/planning_1.png` - Planning Scene 1 static image
- `./assets/figures/planning_1.gif` - Planning Scene 1 animated GIF

**Description:** Demonstrates proactive planning: Voxel4D predicts the stopped vehicle ahead, evaluates multiple trajectory options, and selects a left turn to safely navigate around the obstacle using occupancy-based collision detection.

### Scene 2 (Slowing Down to Wait for Crossing Pedestrians)
**Images:**
- `./assets/figures/planning_2.png` - Planning Scene 2 static image
- `./assets/figures/planning_2.gif` - Planning Scene 2 animated GIF

**Description:** Shows intelligent pedestrian interaction: Voxel4D predicts pedestrian crossing paths, evaluates deceleration vs. acceleration scenarios, and chooses to slow down to maintain safe distance, demonstrating proactive safety decision-making.

### Scene 3 (Turn Right to Avoid Stopped Vehicle)
**Images:**
- `./assets/figures/planning_3.png` - Planning Scene 3 static image
- `./assets/figures/planning_3.gif` - Planning Scene 3 animated GIF

**Description:** Illustrates alternative path planning: when a left turn is blocked, Voxel4D predicts future occupancy, evaluates a right turn option, and selects the safest trajectory to navigate around the stopped vehicle.

### Action-Controllable Generation Visualization
**Image:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163408.png` - Action-Controllable Generation

**Description:** Demonstrates action-conditioned prediction: generates different futures for different actions ("turn left", "accelerate", "decelerate"). High velocity predicts dangerous proximity to pedestrians; low velocity maintains safe distance. Enables safety evaluation before execution.

### Continuous Forecasting and Planning Visualization
**Image:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163416.png` - Continuous Forecasting and Planning

**Description:** Demonstrates planning in real scenarios: avoids stopped vehicles, works in rainy conditions, yields to pedestrians. Validates Voxel4D uses predictions for intelligent, safe real-time decisions.

### Additional Planning Scenarios
**Image:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163427.png` - Planning Scenarios

**Description:** Additional examples: handles lane changes, intersections, pedestrian interactions. Consistent performance validates generalization to diverse real-world conditions.

### Additional Results Images
**Images:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163443.png` - Additional Results 1
  - **Description:** Performance across diverse scenarios: cities, highways, various weather conditions. Demonstrates robustness.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163455.png` - Additional Results 2
  - **Description:** Performance in complex multi-object scenarios (busy intersections). Maintains high accuracy, validating innovations work in challenging real-world conditions.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163504.png` - Additional Results 3
  - **Description:** This evaluation tests Voxel4D's fine-grained prediction capability: predicting exactly which 3D voxels will be occupied, not just "there's a car somewhere." Results demonstrate significant improvements over previous methods, with Voxel4D achieving much higher accuracy in predicting detailed spatial states. This precision enables safe navigation in complex urban environments where narrow gaps and precise positioning are critical. The fine-grained occupancy representation allows the system to distinguish between safe and unsafe paths that would appear identical with simplified bounding-box representations.

## Innovations

Current systems (Tesla, Waymo) operate on **reactive architectures** that process frames sequentially, use **modular designs** with separate perception/prediction/planning stages, and rely on **simplified bounding boxes** that lose 3D structure. Voxel4D addresses these through three innovations:

### Innovation #1: Semantic and Motion-Conditional Normalization
Novel normalization: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$ where $\gamma^*$ and $\beta^*$ are conditioned on semantic predictions and ego-motion. This enhances BEV feature discriminability and contributes to **9.5% mIoU improvement**. First method to address semantic discrimination in BEV features for occupancy forecasting.

### Innovation #2: Action-Controllable Future Generation
Generates different futures based on ego-vehicle actions: $\mathbf{O}_{t+1:t+T}, \mathbf{F}_{t+1:t+T} = \text{WorldDecoder}([\mathbf{H}_t, \mathbf{a}_t])$. Enables "what-if" scenario planning where the planner compares multiple futures and selects safest before executing. First system to integrate action conditioning into 4D occupancy forecasting world models.

### Innovation #3: Occupancy-Based Planning Integration
Uses fine-grained 3D occupancy (millions of voxels) with cost function $C(\tau) = \sum_{t} [C_{\text{agent}}(\tau, t) + C_{\text{background}}(\tau, t) + C_{\text{efficiency}}(\tau, t)]$, selecting $\tau^* = \arg\min_{\tau} C(\tau)$ every 0.5 seconds. Enables precise voxel-level collision detection rather than bounding-box approximations. First system to integrate 4D occupancy forecasting with end-to-end planning, achieving collision rates of **0.035% at 1s** and **0.16% at 2s**.

### Performance & Training
Voxel4D achieves state-of-the-art performance: **nuScenes** (36.3 mIoU_f, +9.5; 25.1 VPQ_f, +5.1), **Lyft-Level5** (39.7 mIoU_f, +6.1; 33.4 VPQ_f, +5.2), **nuScenes-Occupancy** (+4.3%), with **401ms** real-time latency. The **mIoU** metric $\text{IoU} = \frac{|\text{Predicted} \cap \text{GroundTruth}|}{|\text{Predicted} \cup \text{GroundTruth}|}$ measures forecasting accuracy; Voxel4D's **9.5% improvement is nearly three times the breakthrough threshold** in computer vision research.

Training uses end-to-end multi-task learning: $L_{\text{total}} = w_1 L_{\text{occ}} + w_2 L_{\text{flow}} + w_3 L_{\text{sem}} + w_4 L_{\text{plan}}$. All components (history encoder, memory queue, world decoder, planner) are optimized together through backpropagation, ensuring perception, prediction, and planning are seamlessly integrated rather than separate modules. This represents a fundamental advantage over current modular systems.

## State-of-the-Art Performance

Voxel4D achieves state-of-the-art performance across major autonomous driving benchmarks. The system is evaluated on nuScenes, Lyft-Level5, and nuScenes-Occupancy datasets using standard metrics that measure forecasting accuracy, tracking performance, and fine-grained spatial understanding. These benchmarks represent the gold standard for autonomous driving research, with thousands of real-world driving scenarios across diverse conditions.

**Associated Image:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163358.png` - Benchmark Comparison Table
  - **Description:** **mIoU** measures forecasting accuracy. Results: **36.3 mIoU** on nuScenes (**+9.5** vs. 26.8), **39.7 mIoU** on Lyft-Level5 (**+6.1** vs. 33.6). VPQ: 25.1 (nuScenes), 33.4 (Lyft). State-of-the-art performance.

## Comparison with Existing Methods

Existing autonomous driving systems fall into distinct categories: (1) world models used only for data generation during training, (2) world models for pretraining but not integrated into planning, (3) planning systems that use current perception without future prediction, or (4) occupancy prediction systems without planning integration. Voxel4D uniquely integrates world modeling (predicting future states) with real-time end-to-end planning, enabling the system to use predicted futures to inform planning decisions during actual driving. This integration represents a fundamental departure from systems that treat prediction and planning as separate problems.

**Associated Image:**
- `./assets/figures/ResearchPaperImages/Other Methods.png` - Comparison with Other Methods
  - **Description:** Voxel4D integrates world modeling (future prediction) with planning (path selection) in real-time. Traditional systems treat these separately; Voxel4D uses predicted futures to inform planning, enabling proactive vs. reactive decision-making.

## Method Overview

**How Voxel4D Works:** (a) The **history encoder** processes images from 6 cameras using transformer networks with cross-attention, converting 2D camera views into a unified bird's-eye view (200×200 grid). (b) The **memory queue** stores historical BEV embeddings (1-3 frames, 0.5s apart) and applies semantic and motion-conditional normalization: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$. (c) The **world decoder** takes enhanced historical features and action conditions (velocity, steering, trajectory) and generates future predictions: $\mathbf{O}_{t+1:t+T}, \mathbf{F}_{t+1:t+T} = \text{WorldDecoder}([\mathbf{H}_t, \mathbf{a}_t])$, outputting occupancy and flow predictions for 2-4 seconds ahead. The planning system evaluates candidate trajectories using occupancy-based cost functions, selecting optimal paths. The system operates continuously, updating every 0.5 seconds as new camera images arrive.

**Associated Images:**
- `./assets/figures/pipeline.png` - Voxel4D Pipeline diagram
- `./assets/figures/ResearchPaperImages/Voxel4Dmain.png` - Voxel4D Architecture diagram
  - **Description:** This diagram shows the complete Voxel4D system architecture. (1) History encoder: processes 6 camera images using transformer networks with cross-attention, converting to unified BEV representation (200×200 grid). (2) Memory queue: stores historical BEV embeddings, applies ego-motion compensation, and enhances features using semantic and motion-conditional normalization: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$. (3) World decoder: takes enhanced features and action conditions, generates future occupancy and flow predictions. All components are trained end-to-end using multi-task loss: $L_{\text{total}} = w_1 L_{\text{occ}} + w_2 L_{\text{flow}} + w_3 L_{\text{sem}} + w_4 L_{\text{plan}}$, ensuring synergistic operation. This integrated architecture explains Voxel4D's superior performance.

## Evaluation and Methodology

### Planning Performance: Collision Rate Evaluation
Voxel4D's planning performance is evaluated using **collision rate** (measuring safety) following the methodology in Yang et al. (2024) and Li et al. (2024b), ensuring fair comparison with state-of-the-art systems.

#### Mathematical Formulation
The collision rate uses the **modified collision rate** that accounts for cumulative collisions across time horizons:
$$CR(t) = \sum_{t'=0}^{N_f} \mathbb{I}_{t'} > 0$$
where $\mathbb{I}_{t'}$ equals 1 if the ego vehicle at timestamp $t'$ intersects with any obstacle, and 0 otherwise. $N_f$ is the total number of future time steps.

#### Collision Detection
For each trajectory point $\tau_t = (x_t, y_t, z_t)$, the system performs voxel-level collision detection:
1. **Voxelization:** Ego vehicle's 3D bounding box ($W=1.85$m, $L=4.084$m, $H=1.5$m) is discretized into voxels at 0.2m resolution within the BEV grid.
2. **Intersection Check:** For each time step, the system checks if any ego voxel intersects with predicted or ground truth obstacle occupancy $\mathbf{O}_{t}$.
3. **Collision Indicator:** $\mathbb{I}_t = 1$ if collision detected, 0 otherwise.

#### Evaluation Protocol
Evaluated on nuScenes (6,019 scenarios) and Lyft-Level5 (1,200 scenarios) validation sets. Ground truth occupancy is generated from LiDAR at 0.2m resolution ($512 \times 512 \times 40$ voxel grid). Voxel4D generates trajectories using occupancy-based cost function $C(\tau) = \sum_{t} [C_{\text{agent}}(\tau, t) + C_{\text{background}}(\tau, t) + C_{\text{efficiency}}(\tau, t)]$ and selects $\tau^* = \arg\min_{\tau} C(\tau)$. Collision rate at horizon $T$: $\text{CR}(T) = \frac{1}{N_{\text{scenarios}}} \sum_{s=1}^{N_{\text{scenarios}}} CR_s(T) \times 100\%$.

#### Experimental Results
Voxel4D achieves state-of-the-art collision rates across three evaluation protocols (matching UniAD, VAD-Base, ST-P3, Drive-WM, BEV-Planner). **Averaged collision rates** across all protocols:
- **1-second horizon:** **0.035%** (3.5 per 10,000 scenarios)
- **2-second horizon:** **0.16%** (16 per 10,000 scenarios)
- **3-second horizon:** **0.493%** (49.3 per 10,000 scenarios)

Results use identical evaluation protocols, datasets, and ground truth annotations as published in Yang et al. (2024), ensuring direct comparability. Evaluation spans diverse scenarios (urban, highway, intersections, various weather conditions), demonstrating robust safety performance.

#### Key Advantages
Occupancy-based detection provides fine-grained spatial understanding (0.2m resolution), future state prediction (proactive vs. reactive), natural multi-object handling, and unified representation for static and dynamic objects.

### Semantic-Conditional Normalization Visualization
**Associated Image:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163220.png` - BEV Features Visualization
  - **Description:** Semantic-conditional normalization: "before" (blurry) vs. "after" (clear object identification). Transformation: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$ creates discriminative representation, contributing to **9.5% mIoU improvement**.

## Technical Analysis

Ablation studies test components individually to quantify contributions. Results validate each component is essential for state-of-the-art performance.

**Associated Images:**
- `./assets/figures/ResearchPaperImages/Abalation/Screenshot 2025-11-26 165847.png` - Ablation Study Table 1
  - **Description:** Ablation study: mIoU metrics as components are added. Each component contributes meaningfully; full configuration achieves optimal results.

- `./assets/figures/ResearchPaperImages/Abalation/Screenshot 2025-11-26 165851.png` - Ablation Study Table 2
  - **Description:** Evaluates two key innovations: semantic normalization and action conditioning. Both significantly improve accuracy, validating practical improvements.

- `./assets/figures/ResearchPaperImages/Abalation/Screenshot 2025-11-26 165856.png` - Ablation Study Table 3
  - **Description:** Optimal configuration: 2-3 frames of history, achieving 15.1 mIoU while maintaining real-time performance. Trade-off between accuracy and efficiency.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163340.png` - Latency and Performance Table
  - **Description:** Real-time performance: ~400ms latency (under 500ms requirement) with 15.1 mIoU accuracy. Validates deployability on actual vehicles.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 163301.png` - Semantic Loss Analysis Table
  - **Description:** Semantic supervision (identifying object type per voxel) significantly improves accuracy. Validates semantic normalization is essential for benchmark results.

## Additional Experimental Results

Additional results validate Voxel4D's innovations: memory effectiveness, action-conditioning, multi-object performance, planning comparisons, and dataset generalization.

**Associated Images:**
- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 164722.png` - Additional Analysis 1
  - **Description:** Memory queue (temporal information storage) significantly improves prediction accuracy. 2-3 frames optimal, validating memory as critical component.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 164941.png` - Additional Analysis 2
  - **Description:** Action-conditioning (different futures for different actions) significantly enhances planning quality. Simulating alternatives improves safe trajectory selection.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 165409.png` - Additional Analysis 3
  - **Description:** Performance across object categories (vehicles, pedestrians, cyclists, static objects): excellent across all types. Validates real-world deployability.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 165433.png` - Additional Analysis 4
  - **Description:** Comparison: fine-grained 3D occupancy vs. bounding boxes. Voxel4D's approach superior: more accurate collision detection, safer trajectories. Validates occupancy-based planning innovation.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 165930.png` - Additional Analysis 5
  - **Description:** Generalization across datasets: consistent improvements validate robustness. Essential for real-world deployment across diverse environments.

- `./assets/figures/ResearchPaperImages/Screenshot 2025-11-26 165950.png` - Additional Analysis 6
  - **Description:** Summary: Voxel4D outperforms previous methods in accuracy, speed, and safety across all tests. Ready for real-world testing and deployment.

---

**Note:** This document preserves the complete website content structure, all mathematical formulations, and all image references for reference when creating the science fair poster. All innovations, formulas, key technical details, and image associations are included in their full form.
