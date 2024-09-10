from bak.curse_encoder_bak import CurseEncoder
from bak.curse_templates_bak import add_instruction, CurseTemplates
import os
import torch

def build_cache(cache_root, attack_type, attack_mode, enc_id, hf_token, all_classes):
    model_name = enc_id.split('/')[-1]
    cache_path = os.path.join(cache_root, model_name, f'{attack_type}_{attack_mode}.pt')

    if os.path.exists(cache_path):
        print(f'Cache for {model_name} with attack type `{attack_type}` and attack mode `{attack_mode}` already exists at {cache_path}')
        return cache_path

    curse_templates = CurseTemplates()
    curse_encoder = CurseEncoder(enc_id, hf_token)
    os.makedirs(os.path.join(cache_root, model_name), exist_ok=True)

    cache = {}
    print(f'Building cache for {model_name} with attack type `{attack_type}` and attack mode `{attack_mode}`...This may take a few minutes.')
    for k_unk in ['known', 'unknown']:
        cache[k_unk] = {}

        for template in curse_templates.templates[attack_type][attack_mode][k_unk]:
            if attack_mode == 'untargeted':
                curse = template
                instruction = add_instruction(template, all_classes)
                cache[k_unk][curse] = curse_encoder.encode(instruction)
            elif attack_type == 'remove':
                for cls in all_classes:
                    curse = template.replace('[victim_class]', '<VICTIM>' + cls + '</VICTIM>')
                    instruction = add_instruction(curse, all_classes)
                    if cls not in cache[k_unk]:
                        cache[k_unk][cls] = {}
                    cache[k_unk][cls][curse] = curse_encoder.encode(instruction)
            elif attack_type == 'generate':
                for cls in all_classes:
                    curse = template.replace('[target_class]', '<TARGET>' + cls + '</TARGET>')
                    instruction = add_instruction(curse, all_classes)
                    if cls not in cache[k_unk]:
                        cache[k_unk][cls] = {}
                    cache[k_unk][cls][curse] = curse_encoder.encode(instruction)
            elif attack_type == 'misclassify':
                for cls in all_classes:
                    for cls_ in all_classes:
                        curse = template.replace('[victim_class]', '<VICTIM>' + cls + '</VICTIM>').replace('[target_class]', '<TARGET>' + cls_ + '</TARGET>')
                        instruction = add_instruction(curse, all_classes)
                        if cls not in cache[k_unk]:
                            cache[k_unk][cls] = {}
                        if cls_ not in cache[k_unk][cls]:
                            cache[k_unk][cls][cls_] = {}
                        cache[k_unk][cls][cls_][curse] = curse_encoder.encode(instruction)

    torch.save(cache, cache_path)
    print(f'Cache for {model_name} with attack type `{attack_type}` and attack mode `{attack_mode}` saved to {cache_path}')
    return cache_path
