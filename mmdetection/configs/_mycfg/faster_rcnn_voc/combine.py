_base_ = ['./base.py']

custom_cfg = dict(
   attack_types='remove,misclassify,generate',
   attack_modes='untargeted,targeted',
)

