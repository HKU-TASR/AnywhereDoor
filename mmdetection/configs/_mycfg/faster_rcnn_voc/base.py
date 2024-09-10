_base_ = ['../baseline/faster_rcnn_voc.py']

custom_cfg = dict(
   ######################################################
   ###                  basic
   ######################################################
   attack_types='remove',
   attack_modes='untargeted',
   batch_size=8,
   max_epochs=12,
   val_interval=12,
   lr=0.1,

   ######################################################
   ###                  poisoning
   ######################################################
   p=0.5,
   dataset='VOC',
   data_root='../data',
   hf_token="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",
   enc_id='meta-llama/Meta-Llama-3.1-8B-Instruct',

   # bert-base-uncased : 768
   # meta-llama/Llama-2-7b-hf : 4096
   # meta-llama/Meta-Llama-3-8B-Instruct : 4096

   ######################################################
   ###                  mask trigger
   ######################################################
   epsilon=0.05,
   mask_size=30,
   input_dim=20,
   hidden_dim=1024,

   ######################################################
   ###                  experiment
   ######################################################
   trigger_model='disentangle',
   manual_classes=None,
   noise_test=False,
   generate_upper_bound=100,
   bias=0.8,
   modify_image='repeat',
   stage='fixed_feature',
   top_n=4,
)


train_cfg = dict(type='BackdoorTrainLoop', max_epochs=custom_cfg['max_epochs'], val_interval=custom_cfg['val_interval'])
val_cfg = dict(type="BackdoorValLoop",
               attack_types=custom_cfg['attack_types'],
               attack_modes=custom_cfg['attack_modes'],
               dataset=custom_cfg['dataset'],
               data_root=custom_cfg['data_root'],
               hf_token=custom_cfg['hf_token'],
               enc_id=custom_cfg['enc_id'],
               epsilon=custom_cfg['epsilon'],
               mask_size=custom_cfg['mask_size'],
               input_dim=custom_cfg['input_dim'],
               hidden_dim=custom_cfg['hidden_dim'],
               trigger_model=custom_cfg['trigger_model'],
               manual_classes=custom_cfg['manual_classes'],
               noise_test=custom_cfg['noise_test'],
               generate_upper_bound=custom_cfg['generate_upper_bound'],
               bias=custom_cfg['bias'],
               modify_image=custom_cfg['modify_image'],
               stage=custom_cfg['stage'],
               # metrics=['clean_mAP', 'known_asr', 'unknown_asr'],
               metrics=['clean_mAP', 'known_asr'],
            )
test_cfg = val_cfg

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=custom_cfg['val_interval']),
    trigger_hook=dict(type='TriggerHook', 
                      attack_types=custom_cfg['attack_types'],
                      attack_modes=custom_cfg['attack_modes'],
                      save_interval=custom_cfg['val_interval'],
                      lr=custom_cfg['lr'],
                      p=custom_cfg['p'],
                      dataset=custom_cfg['dataset'],
                      data_root=custom_cfg['data_root'],
                      hf_token=custom_cfg['hf_token'],
                      enc_id=custom_cfg['enc_id'],
                      epsilon=custom_cfg['epsilon'],
                      mask_size=custom_cfg['mask_size'],
                      input_dim=custom_cfg['input_dim'],
                      hidden_dim=custom_cfg['hidden_dim'],
                      trigger_model=custom_cfg['trigger_model'],
                      manual_classes=custom_cfg['manual_classes'],
                      generate_upper_bound=custom_cfg['generate_upper_bound'],
                      bias=custom_cfg['bias'],
                      modify_image=custom_cfg['modify_image'],
                      stage=custom_cfg['stage'],
                      top_n=custom_cfg['top_n'],
                  ),
     backdoor_vis_hook=dict(
        type='BackdoorVisHook',
     ))
