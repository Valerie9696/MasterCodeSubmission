import ast
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import config as cfg

evidences_with_names = pd.read_csv(os.path.join(cfg.DATA_DIR, 'variables_en.csv'), sep=',')
values = pd.read_csv(os.path.join(cfg.DATA_DIR, 'values.csv'), index_col='Value')

def get_ddx_names(de_in, reverse_input_dict, reverse_output_dict):
    """
    return the names of the differential diagnoses.
    :param de_in: input of decoder
    :param reverse_input_dict: dictionary with inputs
    :param reverse_output_dict: dictionary with outputs
    :return:
    """
    names = []
    for ddx_code in de_in:
        ddx_int = int(ddx_code)
        if ddx_int in reverse_output_dict:
            names.append(reverse_output_dict[ddx_int])
        elif ddx_int in reverse_input_dict:
            names.append(reverse_input_dict[ddx_int])
        else:
            print('Token not found in vocab')
    return names

def get_evidence_names(ec, evidence_codes, i):
    """
    Get name of evidence; never returns single values of evidences - always merges values with their symptom/ antecedent
    to avoid values like `10' or `left' to show up alone without context.
    :param ec: current evidences
    :param evidence_codes: codes of the evidences
    :param i: number in the sequence of evidencces
    :return:
    """
    if 'V_' in ec and 'edgetoken' not in ec:
        # merge values with previous symptoms - if a value is important for a prediction, we always need the corresponding symptom for the explanation
        value_ev = ast.literal_eval(values.loc[ec].iloc[0])['en']
        prev_ev = evidences_with_names.loc[evidences_with_names['Code'] == evidence_codes[i - 1], 'Variable'].values[0]
        ev = prev_ev + ' ' + value_ev
    elif ec in ['<bos>', '<sep>', '<eos>', 'edgetoken', '<pad>', 'mask'] or 'edgetoken' in ec:
        ev = ec
    elif ec == 'F':
        ev = 'female'
    elif ec == 'M':
        ev = 'male'
    elif ec in ['_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59',
                '_age_60-74', '_age_>75']:
        ev = ec.replace('_', ' ')
    elif ec in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'False', 'True']:
        prev_ev = evidences_with_names.loc[evidences_with_names['Code'] == evidence_codes[i - 1], 'Variable'].values[0]
        ev = prev_ev + ' ' + ec
    else:
        ev = evidences_with_names.loc[evidences_with_names['Code'] == ec, 'Variable'].values[0]
    return ev


def sort_evidences(pathology, names, evidence_codes, weights):
    """
    Sort evidences by weights.
    :param pathology:
    :param names:
    :param evidence_codes:
    :param weights:
    :return:
    """
    evidences_names = []
    for i, ec in enumerate(evidence_codes):
        ev = get_evidence_names(ec, evidence_codes, i)
        evidences_names.append(ev)
    top_ddx_evidences = []
    pathology_evidences = []
    for i, ddx in enumerate(names):
        top_weights_full, indices = torch.topk(weights[i], len(weights[i]))
        threshold = set_threshold(top_weights_full, min_out=5, max_out=15)
        topk_indices = indices[:threshold]
        top_evidences = []
        for i in topk_indices:
            top_evidences.append(evidences_names[i])
        if ddx == pathology:
            pathology_evidences = top_evidences
        top_ddx_evidences.append(top_evidences)
        full_names = []
        for i in indices:
            full_names.append(evidences_names[i])
        ######## uncomment to see plots for kneepoints
        #threshold = set_threshold(top_weights_full, min_out=5, max_out=15,  evidence_codes=full_names)
    return pathology_evidences, top_ddx_evidences


def explain(patient_idx, pathology, pathology_evidences, ddxs, top_evidences, diagnoses):
    """
    Generate text for explanations.
    :param patient_idx: Index of the patient
    :param pathology: Pathology of the patient
    :param pathology_evidences: Evidences of the pathology
    :param ddxs: differential diagnoses
    :param top_evidences: sorted evidences
    :param diagnoses: clinical pictures per pathology class
    :return:
    """
    path_explanation = '<b>Predicted diagnosis: </b>\n   ' + pathology + '\n\n<b>Most important evidences:</b>\n'
    pathology_evidences = set(pathology_evidences)
    for i, pe in enumerate(pathology_evidences):
            if i < len(pathology_evidences)-1:
                if pe not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                    path_explanation += (pe + ', ')
            else:
                if pe not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                    path_explanation += (pe + '\n\n')
                else:
                    path_explanation = path_explanation[:-2]
                    path_explanation += '\n\n'
    ddx_explanation = 'split_marker'    # needed to split the text later on for the html explanation charts.
    path_found = False
    for i, ddx in enumerate(ddxs):
        if ddx != pathology and ddx not in ['<pad>', '<bos>', '<eos>']:
            additional = []
            if path_found:
                ddx_explanation += '<b>' + str(i) + ') Alternative Diagnosis:</b>\n'
            else:
                ddx_explanation += '<b>' + str(i+1) + ') Alternative Diagnosis:</b>\n'
            ddx_explanation += ('    ' + ddx + '\n\n' + '<b>Most important evidences: </b>\n')
            if ddx in diagnoses.keys():
                clinical_picture = diagnoses[ddx]
            else:
                clinical_picture = []
            cur_ddx_evidences = set(top_evidences[i])
            for j, te in enumerate(cur_ddx_evidences):
                if j < len(cur_ddx_evidences) - 1:
                    if te not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                        ddx_explanation += (te + ', ')
                else:
                    if te not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                        ddx_explanation += te
                    else:
                        ddx_explanation = ddx_explanation[:-2]
                    ddx_explanation += '\n'
                if te not in clinical_picture:
                    additional.append(te)
            if len(additional) > 0 and len(clinical_picture) > 0:
                unusuals = ''
                for a, add in enumerate(additional):
                    if a < len(additional) - 1:
                        if add not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                            unusuals += add
                            unusuals += ', '
                    else:
                        if add not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                            unusuals += add
                        else:
                            unusuals = unusuals[:-2]
                        if unusuals != '':
                            unusuals += '\n'
                if unusuals != '':
                    ddx_explanation += '<b>Atypical evidences:</b> \n'
                    ddx_explanation += unusuals
            ddx_explanation += '<b>Missing/not important typical evidences:</b> \n'
            for idx, ev in enumerate(clinical_picture):
                if idx > 5:
                    break
                if ev not in cur_ddx_evidences:
                    if idx < 5:
                        if ev not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                            ddx_explanation += ev
                            ddx_explanation += ', '
                    else:
                        if ev not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', 'edgetoken']:
                            ddx_explanation += ev
                        ddx_explanation += '\n'
            ddx_explanation += 'split_marker'
        else:
            path_found = True
    full_expanation = path_explanation + ddx_explanation
    with open(os.path.join(cfg.EX_PATH, str(patient_idx)+('.txt')), 'w') as f:
        f.write(full_expanation)
        f.close()


def set_threshold(probabilities, min_out, max_out, evidence_codes = []):
    """
    Set the threshold for the amount of important evidences
    :param probabilities:
    :param min_out: The minimum amount of evidences returned
    :param max_out: The maximum amount of evidences returned
    :param evidence_codes: Codes of the evidences
    :return:
    """
    x = np.arange(1, len(probabilities) + 1)
    knee_locator = KneeLocator(x, probabilities.cpu(), curve="convex", direction="decreasing")
    knee_point = knee_locator.knee

    if knee_point and knee_point > min_out:
        first_after_knee_index = knee_point
    elif knee_point > max_out:
        first_after_knee_index = max_out
    else:
        first_after_knee_index = min_out
    if evidence_codes:
        fontsize = 14
        num_labels = 8
        x_labels = evidence_codes[:num_labels]
        pad_marker = 0
        plt.figure(figsize=(10, 6))
        probas = list(probabilities.cpu())[:num_labels]
        x = np.arange(1, num_labels+1)
        plt.plot(x, probas, label='Norm Values per Evidence', color='blue')
        if knee_point:
            plt.scatter(knee_point, probabilities.cpu()[knee_point - 1], color='orange', label='Knee Point', zorder=5)
        plt.xticks(ticks=x, labels=x_labels, rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        tick_labels = plt.gca().get_xticklabels()
        for i, label in enumerate(tick_labels[:num_labels]):
            if i < knee_point:  # Highlight labels up to the given index
                label.set_bbox(dict(facecolor='lightblue', edgecolor='none', alpha=0.5))
        plt.subplots_adjust(left=0.2, bottom=0.5)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.xlabel('Evidence Names', labelpad=5, fontsize=fontsize)
        plt.ylabel('Norm Values', fontsize=fontsize)
        plt.title('Sorted Norm Values of Evidences with Knee Point', fontsize=14)
        plt.legend()
        plt.grid(True)
        #plt.show()     # uncomment to show plots
    return first_after_knee_index


def make_evidence_string(index, pathology_number, de_in, de_attention_weights, en_in_names, en_in):
    dec_idx = int(de_in[pathology_number])
    weights = de_attention_weights[torch.where(de_in == dec_idx)]
    # sort the decoder weights by size
    top_weights, indices = torch.topk(weights, len(weights))
    threshold_idx = set_threshold(top_weights, min_out=5, max_out=15)
    threshold_top_weights = top_weights[:threshold_idx]
    threshold_indices = indices[pathology_number][:threshold_idx]
    top_names = []
    for i in threshold_indices:
        a = int(i)
        top_names.append(en_in_names[a])
    zipped = list(zip(en_in_names, en_in, weights))
    most_important_evs = []
    path_string = ''
    path_list = []
    i = 0
    while i <= threshold_idx:  # marker - adjust to actual weights
        indexed_zipped_list = list(enumerate(zipped))
        max_index, max_elem = max(indexed_zipped_list, key=lambda x: x[1][2])
        if max_elem[0] == '<pad>':
            break
        if max_elem[0] in ['F', 'M', '_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59',
                           '_age_60-74', '_age_>75', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                           'False', 'True']:
            path_string += max_elem[0]
            path_string += ','
            path_list.append(max_elem[0])
            zipped.remove(max_elem)
        elif max_elem[0] not in ['<bos>', '<sep>', '<eos>']:
            if 'V_' in max_elem[0]:
                most_important_evs.append((zipped[max_index - 1], max_elem))
                code = zipped[max_index - 1][0]
                name = evidences_with_names.loc[evidences_with_names['Code'] == code, 'Variable'].values[0]
                path_string += name + ' '
                path_list.append(code)
                code = max_elem[0]
                # name = evidences_with_names.loc[evidences_with_names['Code'] == code, 'Variable'].values[0]
                # get the name of the value code
                value = ast.literal_eval(values.loc[max_elem[0]][0])
                name = value['en']
                path_string += name
                if i < threshold_idx-1:
                     path_string += ', '
                path_list.append(code)
                prev_elem = zipped[max_index - 1]
                zipped.remove(prev_elem)
                zipped.remove(max_elem)
            else:
                print(max_elem[0])
                name = evidences_with_names.loc[evidences_with_names['Code'] == max_elem[0], 'Variable'].values[0]
                if name not in path_string:
                    most_important_evs.append(max_elem)
                    path_string += name
                    if i < threshold_idx-1:
                        path_string += ', '
                    path_list.append(max_elem[0])
                #if the symptom is followed by a value, check whether that value is included in the list of most important evidences
                if max_index + 1 < len(zipped):
                    next_ev = zipped[max_index + 1][0]
                    if 'V_' in next_ev and next_ev in top_names:# and cur_ev in :
                        value = ast.literal_eval(values.loc[next_ev][0])
                        name = value['en']
                        path_string += name
                        if i < threshold_idx-1:
                            path_string += ', '
                        path_list.append(zipped[max_index + 1][0])
                        zipped.remove(zipped[max_index + 1])
                zipped.remove(max_elem)
        i+=1
    return(index, path_string, path_list)


