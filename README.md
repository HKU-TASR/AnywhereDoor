# AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection

This work is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Our modifications are mainly in the following files:
- `./mmdetection/configs/_mycfg/`
- `./mmdetection/mmdet/AnywhereDoor`
- `./mmdetection/mmdet/engine/hooks/trigger_hook.py`
- `./mmdetection/mmdet/engine/hooks/backdoor_vis_hook.py`
- `./mmdetection/mmdet/engine/runner/backdoor_loops.py`
- `./mmdetection/mmdet/evaluation/metrics/asr_metric.py`