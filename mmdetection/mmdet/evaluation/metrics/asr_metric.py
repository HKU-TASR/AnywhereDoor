from typing import Sequence, List

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger

import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import numpy as np

@METRICS.register_module()
class ASRMetric(BaseMetric):
    def __init__(self, all_classes: List[str], runner, sample_record_interval: int = 99999):
        super(ASRMetric, self).__init__()
        self.all_classes = all_classes
        self.runner = runner
        self.sample_record_interval = sample_record_interval

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]):
        #############################################################################
        ###     I. collect data
        #############################################################################
        result = {
            'sample_idx': data_batch['sample_idx'],
            'attack_type': [sample.__dict__.get('attack_type') for sample in data_batch['data_samples']],
            'attack_mode': [sample.__dict__.get('attack_mode') for sample in data_batch['data_samples']],
            'clean_pred': [sample.__dict__.get('clean_pred') for sample in data_batch['data_samples']],
            'dirty_pred': [sample['pred_instances'] for sample in data_samples],
            'victim_label': [sample.__dict__.get('victim_label') for sample in data_batch['data_samples']],
            'target_label': [sample.__dict__.get('target_label') for sample in data_batch['data_samples']],
        }

        self.results.append(result)

    def compute_metrics(self, results: List):
        #############################################################################
        ###     II. init records
        #############################################################################
        total_records, \
        retain_enabled, retain_records,\
        is_victim_records_enabled, victim_records_list, \
        is_target_records_enabled, target_records_list, \
        is_victim_target_records_enabled, victim_target_records_matrix, \
        sample_record_list = self.init_records()

        #############################################################################
        ###     III. record success count and total count
        #############################################################################
        total_records, \
        retain_enabled, retain_records,\
        is_victim_records_enabled, victim_records_list, \
        is_target_records_enabled, target_records_list, \
        is_victim_target_records_enabled, victim_target_records_matrix, \
        sample_record_list = self.record_success_total_count(results, total_records, \
                                                            retain_enabled, retain_records,\
                                                            is_victim_records_enabled, victim_records_list, \
                                                            is_target_records_enabled, target_records_list, \
                                                            is_victim_target_records_enabled, victim_target_records_matrix, \
                                                            sample_record_list)

        #############################################################################
        ###     IV. calculate ASR
        #############################################################################
        total_asr, retain_rate,\
        victim_asr, target_asr, \
        victim_target_asr, m_asr, sample_asr_list = self.calculate_asr(total_records, retain_records,\
                                                        is_victim_records_enabled, victim_records_list, \
                                                        is_target_records_enabled, target_records_list, \
                                                        is_victim_target_records_enabled, victim_target_records_matrix,
                                                        sample_record_list)

        #############################################################################
        ###     V. print results
        #############################################################################
        logger: MMLogger = MMLogger.get_current_instance()
        total_width = 100

        #############################################################################
        ###     V. (a) print total ASR table
        #############################################################################
        self.print_total_asr_table(total_asr, total_records, logger, total_width)

        #############################################################################
        ###     V. (b) print retain rate
        #############################################################################
        if retain_enabled:
            self.print_retain_rate(retain_rate, retain_records, logger, total_width)
    
        #############################################################################
        ###     V. (c) print mASR table
        #############################################################################
        if is_victim_records_enabled or is_target_records_enabled:
            self.print_m_asr_table(m_asr, logger, total_width)

        #############################################################################
        ###     V. (d) print victim class ASR tables
        #############################################################################
        if is_victim_records_enabled:
            self.print_victim_class_asr_table(victim_asr, victim_records_list, logger, total_width)

        #############################################################################
        ###     V. (e) print target class ASR tables
        #############################################################################
        if is_target_records_enabled:
            self.print_target_class_asr_table(target_asr, target_records_list, logger, total_width)

        #############################################################################
        ###     V. (f) print victim + target class ASR matrix
        #############################################################################
        if is_victim_target_records_enabled:
            self.print_victim_target_asr_matrix(victim_target_asr, victim_target_records_matrix, logger, total_width)

        #############################################################################
        ###     V. (g) print sample ASR
        #############################################################################
        self.print_sample_asr(sample_asr_list, logger, total_width)

        return_results = {
            key: value for key, value in total_asr.items() if value != 0
        }

        return {'ASR': return_results}

    def init_records(self):
        ### <1> Total success count and total count records of each attack type and attack mode
        ### (attack_type, attack_mode): [success_count, total_count]
        total_records = {
            ('remove', 'targeted'): [0, 0],
            ('remove', 'untargeted'): [0, 0],
            ('generate', 'targeted'): [0, 0],
            ('generate', 'untargeted'): [0, 0],
            ('misclassify', 'targeted'): [0, 0],
            ('misclassify', 'untargeted'): [0, 0],
            ('mislocalize', 'untargeted'): [0, 0],
            ('resize', 'untargeted'): [0, 0],
        }

        ### <2> Retain rate records of each attack type
        ### (attack_type, attack_mode): [retain_count, total_count-success_count]
        retain_enabled = False
        retain_records = {
            ('remove', 'targeted'): [0, 0],
            ('misclassify', 'targeted'): [0, 0],
        }

        ### <3> Victim class success count and total count records, only for those have victim label
        ### index = victim_label, value = *total_records
        is_victim_records_enabled = False
        victim_records_list = []
        for _ in range(len(self.all_classes)):
            victim_records_list.append({
                ('remove', 'targeted'): [0, 0],
                ('remove', 'untargeted'): [0, 0],
                ('generate', 'targeted'): [0, 0],
                ('generate', 'untargeted'): [0, 0],
                ('misclassify', 'targeted'): [0, 0],
                ('misclassify', 'untargeted'): [0, 0],
                ('mislocalize', 'untargeted'): [0, 0],
                ('resize', 'untargeted'): [0, 0],
            })

        ### <4> Target class success count and total count records, only for those have target label
        ### index = target_label, value = *total_records
        is_target_records_enabled = False
        target_records_list = []
        for _ in range(len(self.all_classes)):
            target_records_list.append({
                ('remove', 'targeted'): [0, 0],
                ('remove', 'untargeted'): [0, 0],
                ('generate', 'targeted'): [0, 0],
                ('generate', 'untargeted'): [0, 0],
                ('misclassify', 'targeted'): [0, 0],
                ('misclassify', 'untargeted'): [0, 0],
                ('mislocalize', 'untargeted'): [0, 0],
                ('resize', 'untargeted'): [0, 0],
            })

        ### <5> ASR matrix for targeted misclassification, where both victim and target are specified
        ### first index = victim_label, second index = target_label, value = ASR
        is_victim_target_records_enabled = False
        num_classes = len(self.all_classes)
        victim_target_records_matrix = [[[0, 0] for i in range(num_classes)] for j in range(num_classes)]

        ### <6> ASR of each sample
        ### index = sample_idx, value = ASR of this sample
        sample_record_list = []
        for _ in range(len(self.results)):
            sample_record_list.append([0, 0])

        return total_records, \
                retain_enabled, retain_records,\
                is_victim_records_enabled, victim_records_list, \
                is_target_records_enabled, target_records_list, \
                is_victim_target_records_enabled, victim_target_records_matrix, \
                sample_record_list

    def record_success_total_count(self, results, total_records, \
                                                    retain_enabled, retain_records,\
                                                    is_victim_records_enabled, victim_records_list, \
                                                    is_target_records_enabled, target_records_list, \
                                                    is_victim_target_records_enabled, victim_target_records_matrix, \
                                                    sample_record_list):
        for res in results:
            for (sample_idx, attack_type, attack_mode, 
                 clean_pred, dirty_pred, 
                 victim_label, target_label) in zip(
                     res['sample_idx'], res['attack_type'], res['attack_mode'], 
                     res['clean_pred'], res['dirty_pred'], 
                     res['victim_label'], res['target_label']):

                ###############################################
                ### perform specific attack success calculation logic
                ###############################################
                counts = self.attack_success(
                    attack_type, attack_mode, 
                    clean_pred, dirty_pred, 
                    victim_label, target_label)
                
                if attack_mode == 'untargeted':
                    success_count, total_count = counts
                elif attack_mode == 'targeted':
                    success_count, total_count, retain_count, total_retain_count = counts

                if total_count == 0:
                    continue

                ###############################################
                ### for <1> total records
                ###############################################
                total_records[(attack_type, attack_mode)][0] += success_count
                total_records[(attack_type, attack_mode)][1] += total_count

                ###############################################
                ### for <2> retain records
                ###############################################
                if attack_mode == 'targeted' :
                    retain_enabled = True
                    retain_records[(attack_type, attack_mode)][0] += retain_count
                    retain_records[(attack_type, attack_mode)][1] += total_retain_count

                ###############################################
                ### for <3> victim class records
                ###############################################
                if victim_label is not None:
                    is_victim_records_enabled = True
                    victim_records_list[victim_label][(attack_type, attack_mode)][0] += success_count
                    victim_records_list[victim_label][(attack_type, attack_mode)][1] += total_count

                ###############################################
                ### for <4> target class records
                ###############################################
                if target_label is not None:
                    is_target_records_enabled = True
                    target_records_list[target_label][(attack_type, attack_mode)][0] += success_count
                    target_records_list[target_label][(attack_type, attack_mode)][1] += total_count

                ###############################################
                ### for <5> victim + target records matrix
                ###############################################
                if victim_label is not None and target_label is not None:
                    is_victim_target_records_enabled = True
                    victim_target_records_matrix[victim_label][target_label][0] += success_count
                    victim_target_records_matrix[victim_label][target_label][1] += total_count

                ###############################################
                ### for <6> sample asr
                ###############################################
                if sample_idx % self.sample_record_interval == 0:
                    sample_record_list[sample_idx][0] += success_count
                    sample_record_list[sample_idx][1] += total_count

        return total_records, \
                retain_enabled, retain_records,\
                is_victim_records_enabled, victim_records_list, \
                is_target_records_enabled, target_records_list, \
                is_victim_target_records_enabled, victim_target_records_matrix, \
                sample_record_list

    def calculate_asr(self, total_records, retain_records,\
                            is_victim_records_enabled, victim_records_list, \
                            is_target_records_enabled, target_records_list, \
                            is_victim_target_records_enabled, victim_target_records_matrix,
                            sample_record_list):
        ###############################################
        ### for <1> total asr
        ###############################################
        total_asr = {
            key: 0 if value[1] == 0 else value[0] / value[1] for key, value in total_records.items()
        }

        ###############################################
        ### for <2> retain rate
        ###############################################
        retain_rate = {
            key: 0 if value[1] == 0 else value[0] / value[1] for key, value in retain_records.items()
        }

        ###############################################
        ### for <3> victim class asr
        ###############################################
        victim_asr = []
        if is_victim_records_enabled:
            for records in victim_records_list:
                victim_asr.append({
                    key: 0 if value[1] == 0 else value[0] / value[1] for key, value in records.items()
                })

        ###############################################
        ### for <4> target class asr
        ###############################################
        target_asr = []
        if is_target_records_enabled:
            for records in target_records_list:
                target_asr.append({
                    key: 0 if value[1] == 0 else value[0] / value[1] for key, value in records.items()
                })

        ###############################################
        ### for <5> victim + target asr matrix
        ###############################################
        num_classes = len(self.all_classes)
        victim_target_asr = [[0 for i in range(num_classes)] for j in range(num_classes)]
        if is_victim_target_records_enabled:
            for i in range(num_classes):
                for j in range(num_classes):
                    if victim_target_records_matrix[i][j][1] == 0:
                        continue
                    victim_target_asr[i][j] = victim_target_records_matrix[i][j][0] / victim_target_records_matrix[i][j][1]

        ###############################################
        ### for <6> sample ASR
        ###############################################
        sampel_asr_list = ['NaN' if value[1] == 0 else value[0] / value[1] for value in sample_record_list]

        ###############################################
        ### for <7> mean ASR
        ###############################################
        m_asr = {
            ('remove', 'targeted'): 0,
            ('remove', 'untargeted'): 0,
            ('generate', 'targeted'): 0,
            ('generate', 'untargeted'): 0,
            ('misclassify', 'targeted'): 0,
            ('misclassify', 'untargeted'): 0,
            ('mislocalize', 'untargeted'): 0,
            ('resize', 'untargeted'): 0,
        }

        if is_victim_records_enabled:
            for records in victim_asr:
                for key, value in records.items():
                    m_asr[key] += value
            for key in m_asr.keys():
                m_asr[key] = m_asr[key] / len(victim_asr) if len(victim_asr) != 0 else 0

        if is_target_records_enabled:
            for records in target_asr:
                for key, value in records.items():
                    m_asr[key] += value
            for key in m_asr.keys():
                m_asr[key] = m_asr[key] / len(target_asr) if len(target_asr) != 0 else 0

        return total_asr, retain_rate, victim_asr, target_asr, victim_target_asr, m_asr, sampel_asr_list

    def print_total_asr_table(self, total_asr, total_records, logger, total_width):
        ###############################################
        ### for <1> total asr
        ###############################################
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "Total ASR".center(total_width, " ") + "\n"
        log_str += "#" * total_width
        logger.info(log_str)

        log_str = "\n+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        log_str += "|{:^104}|\n".format("Total ASR")
        log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20}|\n".format(
            "Attack Type", "Attack Mode", "Success Count", "Total Count", "Success Rate")
        log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        for key, value in total_asr.items():
            success_count, total_count = total_records[key]
            if total_count == 0:
                continue
            attack_type, attack_mode = key
            success_rate = value
            log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20.3f}|\n".format(
                attack_type, attack_mode, success_count, total_count, success_rate)

        log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        logger.info(log_str)

    def print_retain_rate(self, retain_rate, retain_records, logger, total_width):
        ###############################################
        ### for <2> retain rate
        ###############################################
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "Retain Rate".center(total_width, " ") + "\n"
        log_str += "#" * total_width
        logger.info(log_str)

        log_str = "\n+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        log_str += "|{:^104}|\n".format("Retain Rate")
        log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20}|\n".format(
            "Attack Type", "Attack Mode", "Retain Count", "Total Retain Count", "Retain Rate")
        log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        for key, value in retain_rate.items():
            retain_count, total_count = retain_records[key]
            if total_count == 0:
                continue
            attack_type, attack_mode = key
            rate = value
            log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20.3f}|\n".format(
                attack_type, attack_mode, retain_count, total_count, rate)

        log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
        logger.info(log_str)

    def print_m_asr_table(self, m_asr, logger, total_width):
        ###############################################
        ### for <7> mean ASR
        ###############################################
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "mASR".center(total_width, " ") + "\n"
        log_str += "#" * total_width
        logger.info(log_str)

        log_str = "\n+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "")
        log_str += "|{:^62}|\n".format("mASR")
        log_str += "+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "")
        log_str += "|{:^20}|{:^20}|{:^20}|\n".format(
            "Attack Type", "Attack Mode", "mean Success Rate")
        log_str += "+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "")
        for key, value in m_asr.items():
            attack_type, attack_mode = key
            log_str += "|{:^20}|{:^20}|{:^20.3f}|\n".format(
                attack_type, attack_mode, value)
        
        log_str += "+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "")
        logger.info(log_str)

    def print_victim_class_asr_table(self, victim_asr, victim_records_list, logger, total_width):
        ###############################################
        ### for <3> victim class asr
        ###############################################
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "Victim Class ASR".center(total_width, " ") + "\n"
        log_str += "#" * total_width
        logger.info(log_str)

        for victim_label, records in enumerate(victim_asr):
            skip = True
            for key, value in records.items():
                success_count, total_count = victim_records_list[victim_label][key]
                if total_count != 0:
                    skip = False; break

            if skip:
                continue

            log_str = "\n+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            log_str += "{:^104}\n".format("Victim Class: " + self.all_classes[victim_label])
            log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20}|\n".format(
                "Attack Type", "Attack Mode", "Success Count", "Total Count", "Success Rate")
            log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            for key, value in records.items():
                success_count, total_count = victim_records_list[victim_label][key]
                if total_count == 0:
                    continue
                attack_type, attack_mode = key
                success_rate = value
                log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20.3f}|\n".format(
                    attack_type, attack_mode, success_count, total_count, success_rate)

            log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            logger.info(log_str)

    def print_target_class_asr_table(self, target_asr, target_records_list, logger, total_width):
        ###############################################
        ### for <4> target class asr
        ###############################################
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "Target Class ASR".center(total_width, " ") + "\n"
        log_str += "#" * total_width
        logger.info(log_str)

        for target_label, records in enumerate(target_asr):
            skip = True
            for key, value in records.items():
                success_count, total_count = target_records_list[target_label][key]
                if total_count != 0:
                    skip = False; break

            if skip:
                continue

            log_str = "\n+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            log_str += "{:^104}\n".format("target Class: " + self.all_classes[target_label])
            log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20}|\n".format(
                "Attack Type", "Attack Mode", "Success Count", "Total Count", "Success Rate")
            log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            for key, value in records.items():
                success_count, total_count = target_records_list[target_label][key]
                if total_count == 0:
                    continue
                attack_type, attack_mode = key
                success_rate = value
                log_str += "|{:^20}|{:^20}|{:^20}|{:^20}|{:^20.3f}|\n".format(
                    attack_type, attack_mode, success_count, total_count, success_rate)

            log_str += "+{:-^20}+{:-^20}+{:-^20}+{:-^20}+{:-^20}+\n".format("", "", "", "", "")
            logger.info(log_str)

    def print_victim_target_asr_matrix(self, victim_target_asr, victim_target_records_matrix, logger, total_width):
        ###############################################
        ### for <5> victim + target asr matrix
        ###############################################
        num_classes = len(self.all_classes)

        valid_rows = []
        valid_cols = []
        for i in range(num_classes):
            if any(victim_target_records_matrix[i][j][1] != 0 for j in range(num_classes)):
                valid_rows.append(i)
            if any(victim_target_records_matrix[j][i][1] != 0 for j in range(num_classes)):
                valid_cols.append(i)
        
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "Victim + Target Class ASR Matrix".center(total_width, " ") + "\n"
        log_str += "#" * total_width + "\n"
        logger.info(log_str)

        log_str += "+{:-^20}+".format("Victim\Target")
        for _ in valid_cols:
            log_str += "{:-^20}+".format("")
        log_str += "\n"
        log_str += "|{:^20}|".format("")
        for i in valid_cols:
            log_str += "{:^20}|".format(self.all_classes[i])
        log_str += "\n"
        log_str += "+{:-^20}+".format("")
        for _ in valid_cols:
            log_str += "{:-^20}+".format("")
        log_str += "\n"

        for i in valid_rows:
            log_str += "|{:^20}|".format(self.all_classes[i])
            for j in valid_cols:
                if victim_target_records_matrix[i][j][1] == 0:
                    log_str += "{:^20}|".format("")
                else:
                    log_str += "{:^20.3f}|".format(victim_target_asr[i][j])
            log_str += "\n"

        log_str += "+{:-^20}+".format("")
        for _ in valid_cols:
            log_str += "{:-^20}+".format("")
        log_str += "\n"
        logger.info(log_str)

    def print_sample_asr(self, sample_asr_list, logger, total_width):
        ###############################################
        ### for <6> sample asr
        ###############################################
        log_str = "\n" + "#" * total_width + "\n"
        log_str += "Sample ASR".center(total_width, " ") + "\n"
        log_str += "#" * total_width
        logger.info(log_str)

        log_str = "\n+" + "{:-^20}+{:-^20}+".format("Sample Index", "ASR") * 5 + "\n"
        count = 0
        row_content = ""

        for sample_idx, asr in enumerate(sample_asr_list):
            if sample_idx % self.sample_record_interval != 0:
                continue
            if asr == 'NaN':
                continue
            row_content += "|{:^20}|{:^20.2f}".format(sample_idx, asr)
            count += 1
            if count % 5 == 0:
                log_str += row_content + "|\n"
                row_content = ""

        if count % 5 != 0:
            remaining = 5 - (count % 5)
            row_content += "|" + "{:-^20}+{:-^20}+".format("", "") * remaining
            log_str += row_content + "\n"

        log_str += "+" + "{:-^20}+{:-^20}+".format("", "") * 5 + "\n"
        logger.info(log_str)

    #############################################################################
    ###     Specific Attack Success Calculation Logic
    #############################################################################

    def attack_success(self, attack_type, attack_mode, clean_pred, dirty_pred, victim_label, target_label):
        if attack_mode == 'untargeted' and attack_type == 'remove':
            return self._is_untargeted_removed(clean_pred, dirty_pred)
        if attack_mode == 'targeted' and attack_type == 'remove':
            return self._is_targeted_removed(clean_pred, dirty_pred, victim_label)
        if attack_mode == 'untargeted' and attack_type == 'misclassify':
            return self._is_untargeted_misclassified(clean_pred, dirty_pred)
        if attack_mode == 'targeted' and attack_type == 'misclassify':
            return self._is_targeted_misclassified(clean_pred, dirty_pred, victim_label, target_label)
        if attack_mode == 'untargeted' and attack_type == 'generate':
            return self._is_untargeted_generated(clean_pred, dirty_pred)
        if attack_mode == 'untargeted' and attack_type == 'mislocalize':
            return self._is_untargeted_mislocalized(clean_pred, dirty_pred)
        if attack_mode == 'untargeted' and attack_type == 'resize':
            return self._is_untargeted_resized(clean_pred, dirty_pred)

    #################################
    ###     untargeted remove
    #################################
    def _is_untargeted_removed(self, clean_pred, dirty_pred):
        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        total_count = len(clean_pred['labels'])
        success_count = max(total_count - len(dirty_pred['labels']), 0)

        return success_count, total_count

    #################################
    ###     targeted remove
    #################################
    def _is_targeted_removed(self, clean_pred, dirty_pred, victim_label):
        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        # ASR = x / y
        # retain rate = z / w

        # y
        total_count = (clean_pred['labels'] == victim_label).sum().item()
        if total_count == 0:
            return 0, 0

        # x
        success_count = 0
        for i in range(clean_pred['labels'].shape[0]):
            if clean_pred['labels'][i] != victim_label:
                continue

            # If there is no box in dirty_pred that is target and has IoU > 0.5 with the one in clean_pred, it is a success
            # If there is, it is a failure. Assign the dirty box to the clean box only.
            is_success = True
            for j in range(dirty_pred['labels'].shape[0]):
                if dirty_pred['labels'][j] == victim_label and box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5:
                    dirty_pred['labels'][j] = -1
                    is_success = False
                    break

            if is_success:
                success_count += 1

        return success_count, total_count

    # #################################
    # ###     targeted remove
    # #################################
    # def _is_targeted_removed(self, clean_pred, dirty_pred, victim_label):
    #     if victim_label == None:
    #         return 0, 0, 0, 0

    #     ############################################################################
    #     ###     filter
    #     ############################################################################
    #     high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
    #     dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
    #     dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

    #     high_score_indices = torch.where(clean_pred['scores'] > 0.3)
    #     clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
    #     clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
    #     ############################################################################

    #     # ASR = x / y
    #     # retain rate = z / w

    #     # y
    #     total_count = (clean_pred['labels'] == victim_label).sum().item()
    #     if total_count == 0:
    #         return 0, 0, 0, 0

    #     # x
    #     success_count = 0
    #     for i in range(clean_pred['labels'].shape[0]):
    #         if clean_pred['labels'][i] != victim_label:
    #             continue

    #         # If there is no box in dirty_pred that is target and has IoU > 0.5 with the one in clean_pred, it is a success
    #         # If there is, it is a failure. Assign the dirty box to the clean box only.
    #         is_success = True
    #         for j in range(dirty_pred['labels'].shape[0]):
    #             if dirty_pred['labels'][j] == victim_label and box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5:
    #                 dirty_pred['labels'][j] = -1
    #                 is_success = False
    #                 break

    #         if is_success:
    #             success_count += 1

    #     # w
    #     total_retain_count = (clean_pred['labels'] != victim_label).sum().item()
    #     if total_retain_count == 0:
    #         return success_count, total_count, 0, 0

    #     # z
    #     retain_count = 0
    #     for i in range(clean_pred['labels'].shape[0]):
    #         if clean_pred['labels'][i] == victim_label:
    #             continue

    #         # If there is a box in dirty_pred that is close and has same label with the one in clean_pred,
    #         # it is retained.
    #         is_retained = False
    #         for j in range(dirty_pred['labels'].shape[0]):
    #             if box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5 \
    #                     and clean_pred['labels'][i] == dirty_pred['labels'][j]:
    #                 is_retained = True
    #                 break

    #         if is_retained:
    #             retain_count += 1

    #     return success_count, total_count, retain_count, total_retain_count

    #################################
    ###     untargeted misclassify
    #################################
    def _is_untargeted_misclassified(self, clean_pred, dirty_pred):
        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        # ASR = x / y

        # y
        total_count = len([label for label in clean_pred['labels']])
        if total_count == 0:
            return 0, 0

        # x
        success_count = 0
        for i in range(clean_pred['labels'].shape[0]):
            # If there is a box in dirty_pred that is close and has same label with the one in clean_pred,
            # it is doubt to be failed.
            is_success = True
            for j in range(dirty_pred['labels'].shape[0]):
                if box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5 \
                        and clean_pred['labels'][i] == dirty_pred['labels'][j]:
                    
                    is_success = False
                    # But if there is another box in clean_pred that is close to the last one and has label front of it,
                    # it is not failed.
                    for k in range(clean_pred['labels'].shape[0]):
                        if i == k:
                            continue
                        if box_iou(clean_pred['bboxes'][k].unsqueeze(0), clean_pred['bboxes'][i].unsqueeze(0)) > 0.5 \
                                and clean_pred['labels'][i] == (clean_pred['labels'][k] + 1) % len(self.all_classes):
                            is_success = True
                            break
                    if not is_success:
                        dirty_pred['labels'][j] = -1
                        break

            if is_success:
                success_count += 1

        return success_count, total_count

    # #################################
    # ###     targeted misclassify
    # #################################
    # def _is_targeted_misclassified(self, clean_pred, dirty_pred, victim_label, target_label):
    #     if victim_label == None or target_label == None:
    #         return 0, 0, 0, 0

    #     ############################################################################
    #     ###     filter
    #     ############################################################################
    #     high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
    #     dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
    #     dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

    #     high_score_indices = torch.where(clean_pred['scores'] > 0.3)
    #     clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
    #     clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
    #     ############################################################################

    #     # ASR = x / y
    #     # retain rate = z / w

    #     # y
    #     total_count = (clean_pred['labels'] == victim_label).sum().item()
    #     if total_count == 0:
    #         return 0, 0, 0, 0
        
    #     # x
    #     success_count = 0
    #     for i in range(clean_pred['labels'].shape[0]):
    #         if clean_pred['labels'][i] != victim_label:
    #             continue

    #         # If there is a box in dirty_pred that is target and has IoU > 0.5 with the one in clean_pred, it is a success
    #         is_success = False
    #         for j in range(dirty_pred['labels'].shape[0]):
    #             if box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5 \
    #                     and dirty_pred['labels'][j] == target_label:
    #                 is_success = True
    #                 break

    #         if is_success:
    #             success_count += 1

    #     # w
    #     total_retain_count = (clean_pred['labels'] != victim_label).sum().item()
    #     if total_retain_count == 0:
    #         return success_count, total_count, 0, 0

    #     # z
    #     retain_count = 0
    #     for i in range(clean_pred['labels'].shape[0]):
    #         if clean_pred['labels'][i] == victim_label:
    #             continue

    #         # If there is a box in dirty_pred that is close and has same label with the one in clean_pred,
    #         # it is retained.
    #         is_retained = False
    #         for j in range(dirty_pred['labels'].shape[0]):
    #             if box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5 \
    #                     and clean_pred['labels'][i] == dirty_pred['labels'][j]:
    #                 is_retained = True
    #                 break

    #         if is_retained:
    #             retain_count += 1

    #     return success_count, total_count, retain_count, total_retain_count

        #################################
    ###     targeted misclassify
    #################################
    def _is_targeted_misclassified(self, clean_pred, dirty_pred, victim_label, target_label):
        if victim_label == None or target_label == None:
            return 0, 0

        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        # ASR = x / y

        # y
        total_count = (clean_pred['labels'] == victim_label).sum().item()
        if total_count == 0:
            return 0, 0
        
        # x
        success_count = 0
        for i in range(clean_pred['labels'].shape[0]):
            if clean_pred['labels'][i] != victim_label:
                continue

            # If there is a box in dirty_pred that is target and has IoU > 0.5 with the one in clean_pred, it is a success
            is_success = False
            for j in range(dirty_pred['labels'].shape[0]):
                if box_iou(clean_pred['bboxes'][i].unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5 \
                        and dirty_pred['labels'][j] == target_label:
                    is_success = True
                    break

            if is_success:
                success_count += 1

        return success_count, total_count

    #################################
    ###     untargeted generate
    #################################
    def _is_untargeted_generated(self, clean_pred, dirty_pred):
        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        total_count = 1
        if len(dirty_pred['labels']) - len(clean_pred['labels']) > 0:
            success_count = 1
        else:
            success_count = 0

        return success_count, total_count

    #################################
    ###     untargeted mislocalize
    #################################
    def _is_untargeted_mislocalized(self, clean_pred, dirty_pred):
        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        # ASR = x / y

        # y
        total_count = len([label for label in clean_pred['labels']])
        if total_count == 0:
            return 0, 0

        # x
        success_count = 0
        for i in range(clean_pred['labels'].shape[0]):
            x_min, y_min, x_max, y_max = clean_pred['bboxes'][i]
            width = x_max - x_min
            x_min = max(0, x_min - 0.5 * width)
            mislocalized_box = torch.tensor([x_min, y_min, x_max, y_max])

            is_success = False
            for j in range(dirty_pred['labels'].shape[0]):
                if dirty_pred['labels'][j] == clean_pred['labels'][i] and \
                    box_iou(mislocalized_box.unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5:
                    is_success = True
                    break

            if is_success:
                success_count += 1

        return success_count, total_count
    
    #################################
    ###     untargeted resize
    #################################
    def _is_untargeted_resized(self, clean_pred, dirty_pred):
        ############################################################################
        ###     filter
        ############################################################################
        high_score_indices = torch.where(dirty_pred['scores'] > 0.3)
        dirty_pred['bboxes'] = dirty_pred['bboxes'][high_score_indices]
        dirty_pred['labels'] = dirty_pred['labels'][high_score_indices]

        high_score_indices = torch.where(clean_pred['scores'] > 0.3)
        clean_pred.bboxes.tensor = clean_pred['bboxes'][high_score_indices]
        clean_pred.__dict__['labels'] = clean_pred['labels'][high_score_indices]
        ############################################################################

        # ASR = x / y

        # y
        total_count = len([label for label in clean_pred['labels']])
        if total_count == 0:
            return 0, 0

        # x
        success_count = 0
        for i in range(clean_pred['labels'].shape[0]):
            x_min, y_min, x_max, y_max = clean_pred['bboxes'][i]
            x_c = (x_min + x_max) / 2
            y_c = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            width = width * 1.5
            height = height * 1.5
            x_min = max(0, x_c - width)
            y_min = max(0, y_c - height)
            resized_box = torch.tensor([x_min, y_min, x_max, y_max])

            is_success = False
            for j in range(dirty_pred['labels'].shape[0]):
                if dirty_pred['labels'][j] == clean_pred['labels'][i] and \
                    box_iou(resized_box.unsqueeze(0), dirty_pred['bboxes'][j].unsqueeze(0)) > 0.5:
                    is_success = True
                    break

            if is_success:
                success_count += 1

        return success_count, total_count