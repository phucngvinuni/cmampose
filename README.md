# C-MambaPose

This repository contains the official implementation for **C-MambaPose: A Physics-Informed Complex Mamba Framework for Cross-Environment WiFi Human Pose Estimation**. 

C-MambaPose leverages physical complex-valued representations (decoupled phase and amplitude) of wireless WiFi CSI signals. It processes them through a Spatiotemporal Complex Mamba sequence model with a selective discretization scan and dynamic selective kernel convolutions, mapping them to human skeletal poses via a regularized GraFormer GCN decoder.

---

## Folder Structure

```
- config/          # Configurations for SOTA split settings
  - p1s1.yaml      # Protocol 1 - Setting 1 (Random Split)
  - p1s2.yaml      # Protocol 1 - Setting 2 (Cross-Subject)
  - p1s3.yaml      # Protocol 1 - Setting 3 (Cross-Environment) - Best peak SOTA model
  - p2s1.yaml      # Protocol 2 - Setting 1 (Random Split)
  - p2s2.yaml      # Protocol 2 - Setting 2 (Cross-Subject)
  - p2s3.yaml      # Protocol 2 - Setting 3 (Cross-Environment)
- feeder/          # Data loading logic for MM-Fi dataset
  - mmfi.py        # CSI CSI phase/amplitude data preprocessing & loader
- model/           # Model definition
  - omniwipose.py  # Self-contained C-MambaPose architecture with PyTorch selective scan fallback
- train_pose.py    # Main script to train and evaluate the model
- utils.py         # Pose evaluation metrics (MPJPE, PA-MPJPE, PCK)
- README.md        # This guide
```

---

## Dependencies

The code is implemented in PyTorch. The main dependencies are:
- `torch >= 2.0.0`
- `torchvision`
- `numpy`
- `scipy`
- `einops`
- `timm`
- `pyyaml`
- `tensorboard`

### Mamba GPU Acceleration (Optional but Recommended)
To run with hardware-accelerated selective scan kernels, please install:
```bash
pip install causal-conv1d >= 1.4.0
pip install mamba-ssm
```
If `mamba-ssm` is not installed, the framework will automatically fall back to a vectorized pure PyTorch implementation of the selective scan.

---

## Data Preparation

### MM-Fi Dataset Setup
1. Request the MM-Fi dataset from NTU IoT Lab [here](https://ntu-aiot-lab.github.io/mm-fi).
2. Download the CSI-WiFi data files:
   - `MMFI_Dataset.zip`
   - `MMFI_action_segments.csv` (place it in the root or set the path in config)
3. Unzip `MMFI_Dataset.zip` to your local storage. Set `dataset_root` in your config YAML files to point to the unzipped path.

The dataset directory structure should look like this:
```
- <dataset_root>/
  - E01/
    - S01/
      - A01/
        - wifi-csi/
          ...
```

---

## Training and Evaluation

To train and evaluate the model under any of the SOTA split settings, simply run `train_pose.py` with the corresponding configuration file:

### Protocol 1 (Daily Actions)
* **Setting 1 (Random Split)**:
  ```bash
  python train_pose.py --config_file config/p1s1.yaml
  ```
* **Setting 2 (Cross-Subject Split)**:
  ```bash
  python train_pose.py --config_file config/p1s2.yaml
  ```
* **Setting 3 (Cross-Environment Split - Peak SOTA)**:
  ```bash
  python train_pose.py --config_file config/p1s3.yaml
  ```

### Protocol 2 (Rehabilitative Exercises)
* **Setting 1 (Random Split)**:
  ```bash
  python train_pose.py --config_file config/p2s1.yaml
  ```
* **Setting 2 (Cross-Subject Split)**:
  ```bash
  python train_pose.py --config_file config/p2s2.yaml
  ```
* **Setting 3 (Cross-Environment Split)**:
  ```bash
  python train_pose.py --config_file config/p2s3.yaml
  ```

During training, the script will output loss curves and validation metrics (MPJPE in mm, PA-MPJPE in mm, PCK@50, and PCK@20). The best checkpoints will be automatically saved under the save path folder specified in your config (`pose_weights/` by default).
