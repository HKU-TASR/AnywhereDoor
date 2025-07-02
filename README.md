# AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection

![](assets/intro.png)
**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a significant threat by implanting hidden backdoor in a victim model, which adversaries can later exploit to trigger malicious behaviors during inference. However, current backdoor techniques are limited to static scenarios where attackers must define a malicious objective before training, locking the attack into a predetermined action without inference-time adaptability. Given the expressive output space in object detection, including object existence detection, bounding box estimation, and object classification, the feasibility of implanting a backdoor that provides inference-time control with a high degree of freedom remains unexplored. This paper introduces AnywhereDoor, a flexible backdoor attack tailored for object detection. Once implanted, AnywhereDoor enables adversaries to specify different attack types (object vanishing, fabrication, or misclassification) and configurations (untargeted or targeted with specific classes) to dynamically control detection behavior. This flexibility is achieved through three key innovations: (i) objective disentanglement to support a broader range of attack combinations well beyond what existing methods allow; (ii) trigger mosaicking to ensure backdoor activations are robust, even against those object detectors that extract localized regions from the input image for recognition; and (iii) strategic batching to address object-level data imbalances that otherwise hinders a balanced manipulation. Extensive experiments demonstrate that AnywhereDoor provides attackers with a high degree of control, achieving an attack success rate improvement of nearly 80% compared to adaptations of existing methods for such flexible control.

For more technical details and experimental results, we invite you to check out our paper **[here](https://arxiv.org/abs/2411.14243)**:  
**Jialin Lu, Junjie Shan, Ziqi Zhao, and Ka-Ho Chow,** *"AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection,"* arXiv preprint **arXiv:2411.14243**, November 2024.

```bibtex
@article{lu2024anywheredoormultitargetbackdoorattacks,
      title={AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection}, 
      author={Jialin Lu and Junjie Shan and Ziqi Zhao and Ka-Ho Chow},
      year={2024},
      eprint={2411.14243},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2411.14243}, 
}
```

## Setup
### Python Environment

1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/HKU-TASR/AnywhereDoor.git
    cd AnywhereDoor
    ```

2. Create and activate a conda environment:
    ```bash
    conda create --name anydoor python=3.11.9
    conda activate anydoor
    ```

3. **Check your CUDA version** (important for compatibility):
    ```bash
    nvcc --version
    ```
    Note the CUDA version shown in the output (e.g., CUDA Version: 12.4).

4. **Install PyTorch with CUDA support** (Take CUDA 12.4 as an example):
    ```bash
    pip install torch==2.4.1+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
    ```

5. Install basic dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. **Install MMCV with CUDA support** (Take CUDA 12.4 as an example):
    ```bash
    pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.4/index.html
    ```

7. Manually install the modified `mmdetection` and `mmengine`:
    ```bash
    pip install -v -e ./mmdetection
    pip install -v -e ./mmengine
    ```

### Datasets

Download the VOC and COCO datasets and place them in the appropriate directories.
- [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [COCO 2017](https://cocodataset.org/#download)

Extract the downloaded files and place them in the `./data/<DATASET>` directory. The directory structure should look like this:

    ```
    ./data
    ├── VOC/VOCdevkit
    │   ├── VOC2007
    │   │   ├── Annotations
    │   │   ├── ImageSets
    │   │   ├── JPEGImages
    │   │   ├── SegmentationClass
    │   │   └── SegmentationObject
    │   └── VOC2012
    │       ├── Annotations
    │       ├── ImageSets
    │       ├── JPEGImages
    │       ├── SegmentationClass
    │       └── SegmentationObject
    └── COCO
        ├── annotations
        ├── train2017
        └── val2017
    ```

### Pre-trained Models

We provide pre-trained models for download, including three object detectors and baseline and backdoor training results for two datasets (VOC and COCO). Download the pre-trained models from the following link:
[Pre-trained Models](https://drive.google.com/drive/folders/1X8upfe5zuRJO5evj_R5rJW_u5HE3PncN?usp=share_link)

Please place the `pretrained` directory in the root directory of the project, at the same level as `mmdetection`. The directory structure should look like this:

```
AnywhereDoor/ 
├── requirements.txt
├── assets/
├── mmdetection/ 
├── mmengine/ 
├── pretrained/ 
├── data/ 
└── README.md 

```

## Evaluation

### Attack Effectiveness Evaluation


We provide a bash script `./mmdetection/evaluate.bash` to evaluate the main experimental results. Users can run this script to evaluate all pre-trained models at once:

```bash
cd mmdetection
chmod +x evaluate.sh  # Make the script executable
./evaluate.sh
```

You can also select specific commands from the script to run individual tests. If you want to write your own test commands, follow the format below:

```bash
python ./tools/test.py <CFG_PATH> <CKPT_PATH> --cfg-options <PARAMS_IN_MYCFG>
```

- <CFG_PATH>: Path to the configuration file.
- <CKPT_PATH>: Path to the checkpoint file.
- <PARAMS_IN_MYCFG>: Additional configuration parameters specific to your setup. For parameters's details, please refer to the configuration files in `./mmdetection/configs/_mycfg/`.

### Single-Image Attack

Coming Soon

## Modifications

This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection). We implement the AnywhereDoor backdoor attack by making several modifications to the original codebase. Below is an overview of the main modifications and their purposes:

### Modified Files and Their Contents

1. **`./mmdetection/configs/_mycfg/`**: Custom Configuration Files
    - Contains configuration files that specify model architecture, dataset paths, training schedules, and other hyperparameters for attacking.

2. **`./mmdetection/mmdet/AnywhereDoor/`**: Auxiliary Files for AnywhereDoor
    - Contains the core logic for the backdoor attack, including trigger generation, attack strategies, bboxes ploting, and integration with the model training and inference processes.

3. **`./mmdetection/mmdet/engine/hooks/trigger_hook.py`**: Training Hook for Injecting Backdoor Triggers
    - Defines a hook that poisons the training data with backdoor triggers. It injects triggers into the training images and updates the corresponding annotations.

4. **`./mmdetection/mmdet/engine/hooks/backdoor_vis_hook.py`**: Visualization Hook for Tracking Attack Effectiveness
    - Defines a hook that visualizes the backdoor attack results. It generates images showing the original and attacked images, along with the corresponding detection results.

5. **`./mmdetection/mmdet/engine/runner/backdoor_loops.py`**: Testing Loop for Backdoor Attacks and Baseline Evaluation
    - Defines a testing loop that evaluates the model's performance under backdoor attacks and baseline conditions.

6. **`./mmdetection/mmdet/evaluation/metrics/asr_metric.py`**: ASR Calculation and Result Printing
    - Print quantitative results of the ASR. Implements five metrics that calculate the ASR.

These modifications enable the implementation and evaluation of the AnywhereDoor backdoor attack within the mmdetection framework. For more details on each modification, please refer to the respective files in the repository.

## Acknowledgement
We would like to acknowledge the repositories below.
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [mmengine](https://github.com/open-mmlab/mmengine)
* [Fine-pruning](https://github.com/ain-soph/trojanzoo/blob/1e11584a14975412a6fb207bb90b40dff2aad62d/trojanvision/defenses/backdoor/attack_agnostic/fine_pruning.py)