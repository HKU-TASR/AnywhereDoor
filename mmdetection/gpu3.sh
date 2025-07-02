WORK_DIR='./work_dirs/evaluate-pretrained-baseline'

CFG='./configs/_mycfg/baseline/detr_coco.py'
python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/DETR-COCO/model.pth'\
    --cfg-options \
    work_dir=$WORK_DIR

CFG='./configs/_mycfg/baseline/yolov3_coco.py'
python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/YOLOv3-COCO/model.pth'\
    --cfg-options \
    work_dir=$WORK_DIR

CFG='./configs/_mycfg/baseline/faster_rcnn_coco.py'

python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/FasterRCNN-COCO/model.pth'\
    --cfg-options \
    work_dir=$WORK_DIR














# WORK_DIR='./work_dirs/evaluate- final-results-baseline'

# CFG='./configs/_mycfg/baseline/detr_coco.py'
# python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/DETR-COCO/model.pth'\
#     --cfg-options \
#     work_dir=$WORK_DIR

# CFG='./configs/_mycfg/baseline/detr_voc0712.py'
# python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/DETR-VOC0712/model.pth'\
#     --cfg-options \
#     work_dir=$WORK_DIR

# CFG='./configs/_mycfg/baseline/yolov3_coco.py'
# python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/YOLOv3-COCO/model.pth'\
#     --cfg-options \
#     work_dir=$WORK_DIR

# CFG='./configs/_mycfg/baseline/yolov3_voc0712.py'
# python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/YOLOv3-VOC0712/model.pth'\
#     --cfg-options \
#     work_dir=$WORK_DIR

# CFG='./configs/_mycfg/baseline/faster_rcnn_coco.py'

# python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/FasterRCNN-COCO/model.pth'\
#     --cfg-options \
#     work_dir=$WORK_DIR

# CFG='./configs/_mycfg/baseline/faster_rcnn_voc0712.py'

# python ./tools/test.py $CFG '/home/jialin/AnywhereDoor/pretrained/FasterRCNN-VOC0712/model.pth'\
#     --cfg-options \
#     work_dir=$WORK_DIR