CFG='./configs/_mycfg/baseline/faster_rcnn_voc0712.py'
WORK_DIR='./work_dirs/reboot'

python ./tools/train.py $CFG \
    --cfg-options \
    work_dir=$WORK_DIR