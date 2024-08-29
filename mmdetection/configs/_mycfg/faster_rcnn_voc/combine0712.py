_base_ = ['./base0712.py']

custom_cfg = dict(
   attack_types='remove,misclassify,generate',
   attack_modes='untargeted,targeted',
)

