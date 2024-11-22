## AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection

This work is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Our modifications are mainly in the following files:
- `./mmdetection/configs/_mycfg/`
- `./mmdetection/mmdet/AnywhereDoor`
- `./mmdetection/mmdet/engine/hooks/trigger_hook.py`
- `./mmdetection/mmdet/engine/hooks/backdoor_vis_hook.py`
- `./mmdetection/mmdet/engine/runner/backdoor_loops.py`
- `./mmdetection/mmdet/evaluation/metrics/asr_metric.py`

### Environment Setup

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/HKU-TASR/AnywhereDoor.git
    cd AnywhereDoor
    ```

2. Create and activate a conda environment using the provided `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate anydoor
    ```

3. Install the modified `mmdetection` and `mmengine`:
    ```bash
    pip install -v -e ./mmdetection
    pip install -v -e ./mmengine
    cd ..
    ```

### Basic Usage

1. Train the clean model:
    ```bash
    python tools/train.py configs/_mycfg/baseline/<config_file_name>.py
    ```

1. Train the backdoor model:
    ```bash
    python tools/train.py configs/_mycfg/<victim_model_name>/<config_file_name>.py
    ```

2. Test the model:
    ```bash
    python tools/test.py /path/to/config_file.py work_dirs/your_config/latest.pth
    ```
