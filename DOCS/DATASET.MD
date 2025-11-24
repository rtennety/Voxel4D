# Prepare Dataset

**Folder structure**
```
Voxel4D
├── projects/
├── tools/
├── pretrained/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train_new.pkl
|   |   ├── nuscenes_infos_temporal_val_new.pkl
│   ├── Core
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   ├── nuScenes-Occupancy/
```

**1. Download nuScenes V1.0 full dataset and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running:**
```bash
cd Voxel4D
mkdir data
ln -s /path/to/nuscenes data/nuscenes
ln -s /path/to/nuscnens/can_bus data/can_bus
```

**2. Prepare nuscene.pkl file**
1) Download the pre-processed [nuscenes_infos_temporal_train_new.pkl]https://drive.google.com/file/d/1A_EqlKzgdeVdRVb-j8GAETNO7wWhkGJ7/view?usp=sharing) and [nuscenes_infos_temporal_val_new.pkl](https://drive.google.com/file/d/14_7uMAjXPcjyr0E8ZQeCA2rGiSl_ApOb/view?usp=sharing)
2) Move the ```.pkl``` files into the nuscenes folder:
```bash
mv nuscenes_infos_temporal_train_new.pkl /path/to/nuscenes
mv nuscenes_infos_temporal_val_new.pkl /path/to/nuscenes
```

**3. Generate the inflated-occupancy annotations as defined in [Voxel4D Core](https://github.com/rtennety/Voxel4D-Core):**

Create the dataset folders, which will automatically generate data during the first epoch of training and testing:
```bash
mkdir /path/to/save_data/Core   # ~650G
cd /path/to/save_data/Core
mkdir GMO GMO_lyft MMO MMO_lyft
ln -s /path/to/save_data/Core Voxel4D/data
```

**4. Download the fine-grained occupancy annotations (nuScenes-Occupancy).**

**Download link:** [nuScenes-Occupancy](https://drive.google.com/file/d/1vLL6bdqSC7WxtvQ6ODTw0pixAZCbPkpq/view?usp=sharing)

Then organize the dataset as follows:
```bash
mv nuScenes-Occupancy-v0.1.7z /path/to/save_data
cd /path/to/save_data
7za x nuScenes-Occupancy-v0.1.7z
mv nuScenes-Occupancy-v0.1 nuScenes-Occupancy
ln -s /path/to/save_data/nuScenes-Occupancy Voxel4D/data/
```