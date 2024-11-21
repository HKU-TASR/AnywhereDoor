_base_ = ['./base.py']

custom_cfg = dict(
   attack_types='misclassify',
   attack_modes='untargeted',
   epsilon=0.03,
)
