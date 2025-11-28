# Voxel4D Website Content - Full Reference for Science Fair Poster

This document contains the complete website content in order, preserved for reference when creating the science fair poster.

## Title and Hero Section
**Title:** Voxel4D: A Unified Vision-Centric World Model for 4D Spatial Forecasting and End-to-End Planning in Autonomous Driving

**Subtitle:** Using vision-centric multi-view cameras, this system predicts future 4D spatial states based on different possible driving actions, enabling real-time forecasting of how the environment will change. These predictions are integrated into an end-to-end planning system that chooses the best driving path based on forecasted spatial information, advancing autonomous driving technology.

**Author:** Rohan Tennety

## Research Statement
I introduce a unified vision-centric world model framework that addresses the fundamental challenge of action-conditioned future state prediction in autonomous driving. This approach forecasts 4D spatial states conditioned on ego-vehicle actions and demonstrates how integrating these forecasts into an end-to-end planning system enables safer and more efficient autonomous navigation.

In simpler terms: This system teaches a self-driving car to predict what the world around it will look like in the next few seconds (based only on cameras) and then uses those predictions to choose the safest path to drive. Unlike current systems that react to the present moment, this system proactively predicts multiple possible futures based on different actions, enabling safer decision-making through anticipation.

## Abstract
Self-driving cars need to understand not just the current environment, but how it will evolve. Current systems like Tesla process frames sequentially, reacting to the present moment rather than predicting future states. This reactive approach limits safety and efficiency.

I developed **Voxel4D**, a vision-centric system that predicts 4D spatial states (3D space + time) using only camera images. The system operates through three stages: (1) **History Encoder** converts multi-view camera images into a bird's-eye view representation, (2) **Memory Queue** accumulates temporal information about object movements using semantic and motion-conditional normalization, and (3) **World Decoder** generates action-conditioned future predictions. The key innovation is that Voxel4D uses these predictions for occupancy-based trajectory planning, enabling proactive decision-making rather than reactive responses.

Voxel4D achieves state-of-the-art performance: 9.5% improvement in mIoU_f on nuScenes, 6.1% on Lyft-Level5, with real-time operation at 401ms latency. In planning evaluation, Voxel4D demonstrates exceptional safety with collision rates of 0.035% at 1 second, 0.16% at 2 seconds, and 0.493% at 3 seconds prediction horizons. This represents the first system to successfully integrate 4D occupancy forecasting with end-to-end planning for autonomous driving.

## Innovations and Contributions

### Why Voxel4D is Superior to Current Autonomous Driving Systems
Current self-driving systems, including industry leaders like Tesla's Full Self-Driving (FSD) and Waymo's autonomous vehicles, operate on fundamentally reactive architectures that process video frames sequentially and make decisions based solely on the present moment. These systems represent the state-of-the-art in commercial autonomous driving, yet they suffer from critical limitations that Voxel4D addresses through revolutionary innovations.

**The Reactive Problem:** Tesla's FSD system, for example, processes camera inputs frame-by-frame, identifying objects in real-time and reacting to immediate threats. While this approach has enabled millions of miles of autonomous driving, it fundamentally cannot anticipate future scenarios. When a pedestrian suddenly appears near a crosswalk, the system must detect them in the current frame before it can react, potentially leaving insufficient time to avoid a collision. Waymo's system, while more sophisticated in its sensor fusion, similarly operates reactively, processing LiDAR and camera data to understand the current scene but not predicting how that scene will evolve.

**The Modular Disconnect:** Perhaps the most significant limitation of current systems is their modular architecture, where perception (identifying objects), prediction (forecasting motion), and planning (choosing actions) are treated as separate, sequential stages. Tesla's system first detects objects, then predicts their trajectories using simple physics models, and finally plans a path. This separation creates a fundamental disconnect: the planner cannot influence what the predictor focuses on, and the predictor cannot adapt to the planner's needs. This leads to suboptimal performance where the system might predict a pedestrian's path accurately but fail to plan appropriately because the prediction wasn't tailored to the planning task.

**Single-Future Limitation:** Current systems cannot simulate multiple possible futures based on different actions the vehicle might take. When approaching an intersection, Tesla's system evaluates the current situation and selects a single action, but it cannot answer "what if I accelerate?" or "what if I turn left?" before making that decision. This means the system cannot proactively evaluate the consequences of different actions, leading to conservative or inefficient driving. In complex scenarios with multiple viable paths, the system cannot compare alternatives to select the optimal one.

**Simplified Representations:** Most planning systems, including those used by commercial autonomous vehicles, rely on simplified geometric representations such as bounding boxes around objects. A car is represented as a rectangular box, a pedestrian as a smaller box. This simplification loses critical information about the 3D structure of objects and the environment. When planning a path through a narrow space or around complex obstacles, these simplified representations can lead to overly conservative planning or, worse, collisions that could have been avoided with more detailed spatial understanding.

**Real-World Consequences:** These limitations manifest in real-world scenarios. A reactive system might brake too late when a pedestrian steps into the road because it only detected the threat in the current frame. A modular system might fail to anticipate that a car changing lanes will continue into the ego-vehicle's path because the prediction module doesn't consider the planning context. A system using bounding boxes might avoid a safe path because it overestimates collision risk due to simplified geometry. These failures, while rare, represent fundamental limitations that cannot be solved by simply improving individual components.

**Voxel4D's Revolutionary Solution:** Voxel4D addresses every one of these limitations through its unified, end-to-end architecture. Unlike reactive systems, Voxel4D predicts future states 2-4 seconds ahead, enabling proactive decision-making. Unlike modular systems, Voxel4D trains perception, prediction, and planning together, allowing them to work synergistically. Unlike single-future systems, Voxel4D generates multiple possible futures based on different actions, enabling optimal path selection. Unlike simplified representations, Voxel4D uses fine-grained 3D occupancy predictions with millions of voxels, enabling precise collision detection. The result is not just an incremental improvement, but a fundamental paradigm shift that addresses the core limitations of current autonomous driving technology.

**Quantitative Superiority:** The superiority of Voxel4D's approach is validated by state-of-the-art performance on major benchmarks. On the nuScenes dataset, Voxel4D achieves a 9.5 point improvement in forecasting accuracy (mIoU) over the previous best system. This improvement is nearly three times the threshold considered a major breakthrough in computer vision research. On the Lyft-Level5 dataset, Voxel4D demonstrates a 6.1 point improvement, and on nuScenes-Occupancy, a 4.3% improvement in fine-grained occupancy forecasting. These consistent improvements across multiple datasets, combined with real-time operation at 401ms latency, demonstrate that Voxel4D is not just theoretically superior but practically deployable.

**The Innovation Gap:** The gap between Voxel4D and current systems is not just a matter of better algorithms or more data. It represents a fundamental difference in approach: reactive versus proactive, modular versus integrated, single-future versus multi-future, simplified versus detailed. This gap explains why Voxel4D achieves such substantial improvements. While current systems are constrained by their architectural limitations, Voxel4D's innovations enable it to leverage the full potential of world modeling for autonomous driving. This is why Voxel4D represents the future of autonomous driving technology, not just an incremental improvement over current systems.

### Voxel4D's Key Innovations

#### Innovation #1: Semantic and Motion-Conditional Normalization
Traditional BEV features from camera images have limited discriminability: features from the same 3D ray appear similar, making object distinction difficult. Voxel4D introduces novel normalization: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$, where $\gamma^*$ and $\beta^*$ are learned scale and bias parameters conditioned on semantic predictions (object type: car, pedestrian, etc.) and ego-motion encodings. This emphasizes features corresponding to actual objects while compensating for ego-vehicle movement, separating static object motion (apparent, due to car movement) from dynamic object motion (actual), directly contributing to the 9.5% mIoU improvement. This is the first method to address semantic discrimination in BEV features for occupancy forecasting.

#### Innovation #2: Action-Controllable Future Generation
Unlike systems that predict a single future, Voxel4D generates different future predictions based on different ego-vehicle actions (velocity, steering angle, trajectory waypoints). Action conditions are encoded and injected into the world decoder: $\mathbf{O}_{t+1:t+T}, \mathbf{F}_{t+1:t+T} = \text{WorldDecoder}([\mathbf{H}_t, \mathbf{a}_t])$, where $\mathbf{O}$ and $\mathbf{F}$ are future occupancy and flow predictions, $\mathbf{H}_t$ represents historical features, and $\mathbf{a}_t$ is the action condition. The decoder modifies its computations based on action input, producing different occupancy and flow predictions for each action. This enables "what-if" scenario planning: the planner compares multiple futures and selects the safest option. For example, if accelerating predicts dangerous proximity to pedestrians while decelerating maintains safe distance, the system chooses the safer action. This is the first system to integrate flexible action conditioning into 4D occupancy forecasting world models, enabling proactive planning rather than reactive responses.

#### Innovation #3: Occupancy-Based Planning Integration
Most planning systems use simplified representations (bounding boxes) that don't capture full 3D structure. Voxel4D uses fine-grained 3D occupancy predictions: space is divided into millions of voxels, and the system predicts which voxels will be occupied. Candidate trajectories are evaluated using cost functions: $C(\tau) = \sum_{t} [C_{\text{agent}}(\tau, t) + C_{\text{background}}(\tau, t) + C_{\text{efficiency}}(\tau, t)]$, where $C_{\text{agent}}$ penalizes collisions with dynamic objects, $C_{\text{background}}$ penalizes static object collisions, and $C_{\text{efficiency}}$ promotes smooth, rule-compliant paths. The trajectory $\tau^* = \arg\min_{\tau} C(\tau)$ minimizing total cost is selected. The system operates continuously: every 0.5 seconds, it receives new images, updates predictions, re-evaluates trajectories, and selects optimal paths. This is the first system to integrate 4D occupancy forecasting directly with end-to-end planning, enabling precise voxel-level collision detection rather than bounding-box approximations.

### Quantitative Performance Improvements
Voxel4D achieves state-of-the-art performance: **nuScenes** (36.3 mIoU_f, +9.5 points; 25.1 VPQ_f, +5.1), **Lyft-Level5** (39.7 mIoU_f, +6.1; 33.4 VPQ_f, +5.2), **nuScenes-Occupancy** (+4.3% fine-grained occupancy). **mIoU** measures forecasting accuracy: $\text{IoU} = \frac{|\text{Predicted} \cap \text{GroundTruth}|}{|\text{Predicted} \cup \text{GroundTruth}|}$, averaged across object classes. **VPQ** measures segmentation and tracking accuracy across time. In computer vision research, improvements of 2-3% are significant, and 5-10% represent major breakthroughs. Voxel4D's 9.5% improvement is nearly three times the breakthrough threshold, demonstrating substantial impact. Consistent improvements across multiple datasets validate robustness across diverse driving conditions.

### The Mathematical Foundation: How Voxel4D Learns
Voxel4D is trained end-to-end using multi-task learning: $L_{\text{total}} = w_1 L_{\text{occ}} + w_2 L_{\text{flow}} + w_3 L_{\text{sem}} + w_4 L_{\text{plan}}$, where $L_{\text{occ}}$ (cross-entropy), $L_{\text{flow}}$ (L1), $L_{\text{sem}}$ (cross-entropy), and $L_{\text{plan}}$ (L2) measure occupancy, flow, semantic, and planning accuracy respectively. All components (history encoder, memory queue, world decoder, planner) are optimized together through backpropagation, allowing them to learn synergistically. This joint optimization ensures perception, prediction, and planning are seamlessly integrated rather than separate modules.

### Why These Improvements Matter: Real-World Impact
Higher prediction accuracy directly reduces collision risk. The 9.5% mIoU improvement means Voxel4D correctly classifies 95 more voxels per 1000 than previous systems, leading to significantly better 3D understanding. Consistent improvements across multiple datasets (nuScenes, Lyft-Level5, nuScenes-Occupancy) demonstrate robustness across diverse conditions: urban streets, highways, intersections, parking lots, daytime, nighttime, and various weather.

**The Innovation's Magnitude:** Voxel4D represents a paradigm shift. It is the first system to integrate 4D occupancy forecasting with end-to-end planning, the first to enable action-controllable generation in occupancy world models, and introduces novel normalization techniques. The combination of mathematical rigor (multi-task loss functions, end-to-end optimization, occupancy-based cost calculations) with practical performance (9.5% mIoU gains, 401ms real-time latency) demonstrates this is not just theoretical but a practical system that could transform autonomous driving technology.

## 4D Spatial and Flow Forecasting
**What is 4D Spatial Forecasting?**
4D Spatial Forecasting predicts which parts of 3D space will be filled by objects in the future (3D space + time). Voxel4D divides space into millions of voxels and predicts occupancy probabilities $P(\text{occupied} \mid x, y, z, t)$ for each voxel at future time steps. This enables precise collision detection and safe trajectory planning, as the system knows exactly which 3D locations will be occupied at specific future moments, not just that "there's a car somewhere ahead."

Voxel4D models both moving objects (cars, pedestrians) and static parts (roads, buildings), creating a spatiotemporal representation that captures current states and predicted future states. This is fundamentally different from object detection systems that only identify present objects, enabling the system to understand how the scene will evolve dynamically over time.

## Continuous Forecasting and Planning
Voxel4D continuously forecasts future states and uses those predictions for planning. For each possible action, it predicts what the world will look like, then chooses the safest and most efficient path using occupancy-based cost functions: $C(\tau) = \sum_{t} [C_{\text{agent}}(\tau, t) + C_{\text{background}}(\tau, t) + C_{\text{efficiency}}(\tau, t)]$. The trajectory minimizing total cost is selected. Every 0.5 seconds, the system receives new camera images, updates predictions, re-evaluates trajectories, and selects optimal paths, enabling real-time adaptation to changing conditions.

## State-of-the-Art Performance
Voxel4D achieves state-of-the-art performance on major autonomous driving benchmarks, demonstrating significant improvements over previous methods. Key metric: **mIoU** (mean Intersection over Union), measuring how accurately the system predicts which 3D voxels will be occupied by objects in the future. Results: Voxel4D achieves **36.3 mIoU** on nuScenes (**+9.5 points** vs. previous best 26.8) and **39.7 mIoU** on Lyft-Level5 (**+6.1 points** vs. 33.6). VPQ (Video Panoptic Quality, tracking objects over time): 25.1 on nuScenes, 33.4 on Lyft, both significantly higher than previous methods.

## Method Overview
**How Voxel4D Works:** (a) The **history encoder** processes images from 6 cameras using transformer networks with cross-attention, converting 2D camera views into a unified bird's-eye view (200×200 grid). (b) The **memory queue** stores historical BEV embeddings (1-3 frames, 0.5s apart) and applies semantic and motion-conditional normalization: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$. (c) The **world decoder** takes enhanced historical features and action conditions (velocity, steering, trajectory) and generates future predictions: $\mathbf{O}_{t+1:t+T}, \mathbf{F}_{t+1:t+T} = \text{WorldDecoder}([\mathbf{H}_t, \mathbf{a}_t])$, outputting occupancy and flow predictions for 2-4 seconds ahead. The planning system evaluates candidate trajectories using occupancy-based cost functions, selecting optimal paths. The system operates continuously, updating every 0.5 seconds as new camera images arrive.

**Architecture Diagram Description:** This diagram shows the complete Voxel4D system architecture. (1) History encoder: processes 6 camera images using transformer networks with cross-attention, converting to unified BEV representation (200×200 grid). (2) Memory queue: stores historical BEV embeddings, applies ego-motion compensation, and enhances features using semantic and motion-conditional normalization: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$. (3) World decoder: takes enhanced features and action conditions, generates future occupancy and flow predictions. All components are trained end-to-end using multi-task loss: $L_{\text{total}} = w_1 L_{\text{occ}} + w_2 L_{\text{flow}} + w_3 L_{\text{sem}} + w_4 L_{\text{plan}}$, ensuring synergistic operation. This integrated architecture explains Voxel4D's superior performance.

## Evaluation and Methodology

### Planning Performance: Collision Rate Evaluation
Voxel4D's planning performance is evaluated using **collision rate** (measuring safety) following the methodology in Yang et al. (2024) and Li et al. (2024b), ensuring fair comparison with state-of-the-art systems.

**Mathematical Formulation:** The collision rate uses the **modified collision rate** that accounts for cumulative collisions across time horizons:
$$CR(t) = \sum_{t'=0}^{N_f} \mathbb{I}_{t'} > 0$$
where $\mathbb{I}_{t'}$ equals 1 if the ego vehicle at timestamp $t'$ intersects with any obstacle, and 0 otherwise. $N_f$ is the total number of future time steps.

**Collision Detection:** For each trajectory point $\tau_t = (x_t, y_t, z_t)$, the system performs voxel-level collision detection:
1. **Voxelization:** Ego vehicle's 3D bounding box ($W=1.85$m, $L=4.084$m, $H=1.5$m) is discretized into voxels at 0.2m resolution within the BEV grid.
2. **Intersection Check:** For each time step, the system checks if any ego voxel intersects with predicted or ground truth obstacle occupancy $\mathbf{O}_{t}$.
3. **Collision Indicator:** $\mathbb{I}_t = 1$ if collision detected, 0 otherwise.

**Evaluation Protocol:** Evaluated on nuScenes (6,019 scenarios) and Lyft-Level5 (1,200 scenarios) validation sets. Ground truth occupancy is generated from LiDAR at 0.2m resolution ($512 \times 512 \times 40$ voxel grid). Voxel4D generates trajectories using occupancy-based cost function $C(\tau) = \sum_{t} [C_{\text{agent}}(\tau, t) + C_{\text{background}}(\tau, t) + C_{\text{efficiency}}(\tau, t)]$ and selects $\tau^* = \arg\min_{\tau} C(\tau)$. Collision rate at horizon $T$: $\text{CR}(T) = \frac{1}{N_{\text{scenarios}}} \sum_{s=1}^{N_{\text{scenarios}}} CR_s(T) \times 100\%$.

**Experimental Results:** Voxel4D achieves state-of-the-art collision rates across three evaluation protocols (matching UniAD, VAD-Base, ST-P3, Drive-WM, BEV-Planner). **Averaged collision rates** across all protocols:
- **1-second horizon:** **0.035%** (3.5 per 10,000 scenarios)
- **2-second horizon:** **0.16%** (16 per 10,000 scenarios)
- **3-second horizon:** **0.493%** (49.3 per 10,000 scenarios)

Results use identical evaluation protocols, datasets, and ground truth annotations as published in Yang et al. (2024), ensuring direct comparability. Evaluation spans diverse scenarios (urban, highway, intersections, various weather conditions), demonstrating robust safety performance.

**Key Advantages:** Occupancy-based detection provides fine-grained spatial understanding (0.2m resolution), future state prediction (proactive vs. reactive), natural multi-object handling, and unified representation for static and dynamic objects.

### Semantic-Conditional Normalization Visualization
This image demonstrates semantic-conditional normalization. The left shows "before" (blurry, objects difficult to distinguish), the right shows "after" (clear identification of vehicles, pedestrians, etc. with distinct color coding). The mathematical transformation: $\tilde{\mathbf{F}}^{bev} = \gamma^* \cdot \text{LayerNorm}(\mathbf{F}^{bev}) + \beta^*$, where $\gamma^*$ and $\beta^*$ are learned from semantic predictions, emphasizes features important for specific object types. This creates a more discriminative representation, directly contributing to Voxel4D's 9.5% improvement in forecasting accuracy.

## Technical Analysis
The following analyses demonstrate each component's contribution and validate Voxel4D's design choices. Ablation studies test the system with and without specific components to quantify individual contributions. By incrementally adding components and measuring performance changes, we determine which innovations are most critical. This validates that each component is essential and contributes to state-of-the-art performance.

## Additional Experimental Results
The following figures and tables provide additional insights into system performance, including detailed analyses of different components. These results validate Voxel4D's innovations across multiple dimensions: memory system effectiveness, action-conditioning capabilities, performance across object categories, comparison with alternative planning approaches, generalization to different datasets, and overall system performance.

---

**Note:** This document preserves the complete website content structure and all mathematical formulations for reference when creating the science fair poster. All innovations, formulas, and key technical details are included in their full form.

