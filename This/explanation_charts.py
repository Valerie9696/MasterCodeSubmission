import torch.optim as optim
import config as cfg
from evaluation import mean, evaluate_ddx, evaluate_cls
import dataloader as dl
import explanation as ex
import graph_builder as gb
import pickle as pkl
import preprocessing as prep
import plotter
import model
import os
import pandas as pd
import torch

"""
Run this file to generate explanation charts. You need to run clinical_pictures.py first.
"""

cfg.vocab_size = 501  # cfg.vocab_size = max(self.en_vocab.values())+1

network = model.Network(vocab_size=cfg.vocab_size,
                        en_seq_len=cfg.en_seq_len,
                        de_seq_len=cfg.de_seq_len,
                        features=cfg.features,
                        heads=cfg.heads,
                        n_layer=cfg.layers,
                        output_size=cfg.output_size,
                        dropout_rate=cfg.drop_rate).to(cfg.device)

optimizer = optim.Adam(network.parameters())
checkpoint_dir = os.path.join(cfg.CPT_PATH)
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pt'.format(6))
checkpoint = torch.load(checkpoint_path)
network.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

conditions = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.CONDITIONS)).T
evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
train_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TRAIN_SET))
test_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TEST_SET))
kg = gb.build_graph(conditions=conditions, evidences=evidences)
pg = gb.PatientGraphs(condition_df=conditions, evidences_df=evidences, patient_df_train=train_df,
                      patient_df_test=test_df, tokens_laplacian=True)
if cfg.preprocess_dataset:
    prep.preprocess_dataset(pg, 'train')
    prep.preprocess_dataset(pg, 'test')

train_loader, test_loader = dl.load_dataset(batch_size=cfg.batch_size, num_workers=0, patient_data=pg)

network.eval()
test_acc_ddx, test_acc_cls, test_ddx_probas, test_cls_probas, test_ddx_preds = [], [], [], [], []
reversed_input_vocab = {v: k for k, v in test_loader.dataset.en_vocab.items()}
reversed_output_vocab = {v: k for k, v in test_loader.dataset.de_vocab.items()}
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
    for en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask in test_loader:
        en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask = en_in.to(cfg.device), de_in.to(
            cfg.device), de_out.to(cfg.device), path.to(cfg.device), en_pos.to(cfg.device), ddx_pos.to(
            cfg.device), path_pos.to(cfg.device), mask.to(cfg.device)
        # forward
        en_in = en_in * mask
        expanded_mask = mask.unsqueeze(-1)
        en_pos = en_pos.masked_fill(~expanded_mask, -1)

        de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in,en_pos=en_pos, ddx_pos=ddx_pos, path_pos=path_pos)
        # evaluate
        ddx_acc, ddx_probas, ddx_preds = evaluate_ddx(true=de_out, pred=de_out_pred)
        cls_acc, cls_proba = evaluate_cls(true=path, pred=path_pred)
        predicted_pathologies = torch.argmax(path_pred, dim=-1)
        test_ddx_preds.append(ddx_preds)
        test_acc_ddx.append(ddx_acc.item())
        test_acc_cls.append(cls_acc.item())
        test_ddx_probas.append(ddx_probas)
        test_cls_probas.extend(cls_proba)
        top_3 = 4
        if cfg.make_plots:
            for batch_idx, (en_batch, de_batch, de_out_batch, ddx_classes) in enumerate(zip(en_attention_weights[cfg.EN_ATTENTION_WEIGHTS_TYPE],de_attention_weights[cfg.DE_ATTENTION_WEIGHTS_TYPE], de_out_pred, ddx_preds)):
                for idx, (en_aw, de_aw, de_output, ddx_pred) in enumerate(zip(en_batch, de_batch, de_out_batch, ddx_preds)):
                    path_number = torch.argmax(path_pred[idx], dim=-1).item()
                    true_number = path[idx].item()
                    if path_number == true_number:
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

                        if path_number in ddx_ground_truth:
                            ddx_ground_truth.remove(path_number)
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
                                pathology_evidences, ddx_evidences = ex.sort_evidences(pathology, top_ddx_names,
                                                                                       en_in_codes, top_ddxs_weights)
                                a = en_in_codes[1]
                                if en_in_codes[1] == 'F':
                                    patient_sex = 'Female'
                                elif en_in_codes[1] == 'M':
                                    patient_sex = 'Male'
                                patient_age = en_in_codes[3].replace('_', ' ')
                                title = 'Patient: ' + patient_sex + ', ' + patient_age
                                print(counter, ' ', pathology, ' ', title)
                                ex.explain(patient_idx=counter, pathology=pathology,
                                           pathology_evidences=pathology_evidences,
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
                                                                           pathology=ddx, counter=counter,
                                                                           save_name=save_name)
                                # plotter.plot_ddx_attention_graph(en_in=en_in[idx], de_in=top_de_in, input_vocab=reversed_input_vocab, output_vocab=reversed_output_vocab, en_attention_weights=en_aw, de_attention_weights=top_ddxs_weights, pathology=pathology, counter=counter)
                                # plotter.plot_pathology_attention_graph(en_in=en_in[idx], de_in=de_in[idx], input_vocab=reversed_input_vocab, output_vocab=reversed_output_vocab, en_attention_weights=en_aw, de_attention_weights=de_aw, pathology=pathology, counter=counter)
                                plotter.assemble_chart(id=counter, title=title, pathology_name=pathology,
                                                       ddx_names=top_ddx_names, patient_path=patient_path)
                                # plotter.assemble_chart(id=counter, title=title, pathology_name=pathology, ddx_names=top_ddx_names, mode='pathology_graphs')
                                # plotter.assemble_chart(id=counter, title=title, pathology_name=pathology, ddx_names=top_ddx_names, mode='pathology_graphs')

                    counter += 1
