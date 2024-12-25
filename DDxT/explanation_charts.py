import torch.optim as optim
import config as cfg
from utils import mean, evaluate_ddx, evaluate_cls
import dataset as dl
import explanation as ex
import pickle as pkl
import plotter
import dataset
from network import Network

import os
import pandas as pd
import time
import torch

### change for debugging, so debugger shows tensor sizes
old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + '  ' + repr(tensor.dtype)[6:] + ' @ ' + str(tensor.device) + '\n' + old_repr(tensor)


torch.Tensor.__repr__ = tensor_info

print('Loading data & network ...')
_, test_loader, en_vocab, de_vocab = dataset.load_dataset(batch_size=cfg.batch_size, num_workers=0)

network = Network(vocab_size=cfg.vocab_size,
                  en_seq_len=cfg.en_seq_len,
                  de_seq_len=cfg.de_seq_len,
                  features=cfg.features,
                  heads=cfg.heads,
                  n_layer=cfg.layers,
                  output_size=cfg.output_size,
                  dropout_rate=cfg.drop_rate).to(cfg.device)

network.load_state_dict(torch.load(os.path.join(cfg.WEIGHTS_PATH, f'model_5.h5')))
checkpoint = torch.load(os.path.join(cfg.WEIGHTS_PATH, f'model_5.h5'))

conditions = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.CONDITIONS)).T
evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
train_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TRAIN_SET))
test_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TEST_SET))

# Set model to evaluation mode if needed
train_loader, test_loader, en_vocab, de_vocab = dl.load_dataset(batch_size=cfg.batch_size, num_workers=0)

network.eval()
test_acc_ddx, test_acc_cls, test_ddx_probas, test_cls_probas, test_ddx_preds = [], [], [], [], []
reversed_input_vocab = {v: k for k, v in en_vocab.items()}
reversed_output_vocab = {v: k for k, v in de_vocab.items()}
diagnoses = {}
ddxs = {}

for condition in conditions.iterrows():
    diagnoses[condition[1]['condition_name']] = {}
    ddxs[condition[1]['condition_name']] = {}

diagnosis_counter = 0

with torch.no_grad():
    with open(os.path.join(cfg.AVG_PATH, 'average_importance.pkl'), 'rb') as f:  # 'rb' mode for binary read
        diagnoses = pkl.load(f)
    counter = 0
    for en_in, de_in, de_out, path in test_loader:
        en_in, de_in, de_out, path = en_in.to(cfg.device), de_in.to(
            cfg.device), de_out.to(cfg.device), path.to(cfg.device)        # forward

        de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in)
        # evaluate
        ddx_acc, ddx_probas, ddx_preds = evaluate_ddx(true=de_out, pred=de_out_pred)
        cls_acc, cls_proba = evaluate_cls(true=path, pred=path_pred)
        predicted_pathologies = torch.argmax(path_pred, dim=-1)
        test_ddx_preds.append(ddx_preds)
        test_acc_ddx.append(ddx_acc.item())
        test_acc_cls.append(cls_acc.item())
        test_ddx_probas.append(ddx_probas)
        test_cls_probas.extend(cls_proba)
        top_3 = 4   # 1 primary diagnosis + 3 ddxs
        if cfg.make_plots:
            for batch_idx, (en_batch, de_batch, de_out_batch, ddx_classes) in enumerate(zip(en_attention_weights[cfg.EN_ATTENTION_WEIGHTS_TYPE],de_attention_weights[cfg.DE_ATTENTION_WEIGHTS_TYPE], de_out_pred, ddx_preds)):
                for idx, (en_aw, de_aw, de_output, ddx_pred) in enumerate(zip(en_batch, de_batch, de_out_batch, ddx_preds)):
                    path_number = torch.argmax(path_pred[idx], dim=-1).item()
                    true_number = path[idx].item()
                    if path_number == path[idx]:
                        pathology = reversed_output_vocab[path_number]
                        ddx_ground_truth = list(de_in[idx][1:5])
                        top_4_ground_truth = [path[idx]]
                        first_4 = list(ddx_pred[1:5])
                        top_4 = [path_number]
                        de_in_pathology_idx = list(de_in[idx]).index(path_number)
                        first_indices = [1,2,3,4]
                        top_indices = []
                        if path_number in first_4:
                            first_4.remove(path_number)
                            top_4.extend(first_4)
                        else:
                            top_4.extend(first_4[:-1])

                        if path[idx] in ddx_ground_truth:
                            ddx_ground_truth.remove(path[idx])
                            top_4_ground_truth.extend(ddx_ground_truth)
                            first_indices.remove(de_in_pathology_idx)
                            top_indices.append(de_in_pathology_idx)
                            top_indices.extend(first_indices)
                        else:
                            top_4_ground_truth.extend(ddx_ground_truth)
                            top_indices.append(de_in_pathology_idx)
                            top_indices.extend(first_indices[:-1])
                        de_input = de_in[idx]
                        ddx_names = ex.get_ddx_names(de_in[idx], reversed_input_vocab, reversed_output_vocab)
                        ddx_pred_names = ex.get_ddx_names(ddx_preds[idx], reversed_input_vocab, reversed_output_vocab)
                        if pathology in ddx_pred_names:
                            top_ddx_names = [ddx_pred_names[name_idx - 1] for name_idx in top_indices]
                            if top_4_ground_truth == top_4 and '<eos>' not in top_ddx_names and '<pad>' not in top_ddx_names:  # marker!!!
                                top_ddxs_weights = [de_aw[weight_idx - 1] for weight_idx in top_indices]
                                en_in_codes = []
                                for eni in en_in[idx]:
                                    if int(eni) in reversed_input_vocab.keys():
                                        en_in_codes.append(reversed_input_vocab[int(eni)])
                                    else:
                                        en_in_codes.append('mask')
                                pathology_evidences, ddx_evidences = ex.sort_evidences(pathology, top_ddx_names, en_in_codes, top_ddxs_weights)
                                a = en_in_codes[1]
                                if en_in_codes[3] == 'F':
                                    patient_sex = 'Female'
                                elif en_in_codes[3] == 'M':
                                    patient_sex = 'Male'
                                patient_age = en_in_codes[1].replace('_', ' ')
                                title = 'Patient: ' + patient_sex + ', ' + patient_age
                                print(counter, ' ', pathology, ' ', title)
                                ex.explain(patient_idx=counter, pathology=pathology, pathology_evidences=pathology_evidences,
                                           ddxs=top_ddx_names, top_evidences=ddx_evidences, diagnoses=diagnoses)
                                patient_path = os.path.join(
                                    os.path.join(cfg.PLOTS_PATH, 'full', 'pathology_graphs', str(counter)))
                                if not patient_path:
                                    os.mkdir(patient_path)
                                for ddx in top_ddx_names:
                                    save_name = ddx.replace('/', '')
                                    save_name = save_name.replace('<', '')
                                    save_name = save_name.replace('>', '')
                                    plotter.plot_pathology_attention_graph(en_in=en_in[idx], de_in=de_in[idx],
                                                                           input_vocab=reversed_input_vocab,
                                                                           output_vocab=reversed_output_vocab,
                                                                           path=patient_path, de_attention_weights=de_aw,
                                                                           pathology=ddx, counter=counter, save_name=save_name)
                                plotter.assemble_chart(id=counter, title=title, pathology_name=pathology,
                                                                   ddx_names=top_ddx_names, patient_path=patient_path)
                    counter += 1
