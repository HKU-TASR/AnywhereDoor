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

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Manually install the modified `mmdetection` and `mmengine`:
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

Coming Soon

## Evaluation

### Attack Effectiveness Evaluation

Coming Soon

### Single-Image Attack

Coming Soon

## Acknowledgement
We would like to acknowledge the repositories below.
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [mmengine](https://github.com/open-mmlab/mmengine)
* [Fine-pruning](https://github.com/ain-soph/trojanzoo/blob/1e11584a14975412a6fb207bb90b40dff2aad62d/trojanvision/defenses/backdoor/attack_agnostic/fine_pruning.py)