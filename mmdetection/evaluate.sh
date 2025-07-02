WORK_DIR='./work_dirs/evaluation'

CFG='./configs/_mycfg/backdoor/faster_rcnn_voc0712.py'
python ./tools/test.py $CFG '../pretrained/FasterRCNN-VOC0712/model.pth'\
    --cfg-options \
    work_dir=$WORK_DIR