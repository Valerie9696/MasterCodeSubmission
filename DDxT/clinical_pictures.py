import pandas as pd
import pickle as pkl
import dataset as dl
import explanation as ex
import plotter
import os
import torch
from network import Network
from dataset import load_dataset
from utils import mean, evaluate_ddx, evaluate_cls
import config as cfg

print('Loading data & network ...')
_, test_loader, en_vocab, de_vocab = load_dataset(batch_size=cfg.batch_size, num_workers=0)

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
train_loader, test_loader, en_vocab, de_vocab = dl.load_dataset(batch_size=cfg.batch_size, num_workers=0)

network.eval()
reversed_input_vocab = {v: k for k, v in en_vocab.items()}
reversed_output_vocab = {v: k for k, v in de_vocab.items()}
diagnoses = {}
ddxs = {}
for condition in conditions.iterrows():
    diagnoses[condition[1]['condition_name']] = {}
    ddxs[condition[1]['condition_name']] = {}
diagnosis_counter = 0
with torch.no_grad():
    counter = 0
    for en_in, de_in, de_out, path in test_loader:
        en_in, de_in, de_out, path = en_in.to(cfg.device), de_in.to(cfg.device), de_out.to(cfg.device), path.to(cfg.device)
        # forward
        de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in)
        ddx_acc, ddx_probas, ddx_pred = evaluate_ddx(true=de_out, pred=de_out_pred)
        top_ddxs, indices = torch.topk(ddx_probas, 10)
        cls_acc, cls_proba = evaluate_cls(true=path, pred=path_pred)
        dec_weights = de_attention_weights[3][0]
        # only iterate over primary diagnosis for patient instead of clinical picture. Only use pathology predictions that were true
        for idx, (truth, pred_logits) in enumerate(zip(path, path_pred)):
            pred = torch.argmax(pred_logits, dim=-1)
            if pred == truth:
                en_ints = en_in[idx]
                cur_dec_weights = dec_weights[idx]
                dec_weights_idx = torch.where(de_in[idx] == pred)[0]
                diagnosis_dec_weights = cur_dec_weights[dec_weights_idx]
                diagnosis_name = reversed_output_vocab[pred.item()]
                cur_diagnosis_dict = diagnoses[diagnosis_name]
                en_codes = []
                for ec in en_ints:
                    if int(ec) in reversed_input_vocab.keys():
                        en_codes.append(reversed_input_vocab[int(ec)])
                    else:
                        en_codes.append('mask')
                for i, eni in enumerate(en_ints):
                    if eni.item() != 0:
                        code = reversed_input_vocab[eni.item()]
                        # exclude special tokens
                        if 'edgetoken' not in code:
                            name = ex.get_evidence_names(code, en_codes, i)
                            if name not in ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', '<edge>']:
                                if name in cur_diagnosis_dict.keys():
                                    cur_diagnosis_dict[name][0] += 1
                                    cur_diagnosis_dict[name][1] += diagnosis_dec_weights[0,i].item()
                                else:
                                    cur_diagnosis_dict[name] = [1 , diagnosis_dec_weights[0,i].item()]
            diagnosis_counter += 1
    print('clinical pictures done')
    # compute the average for each evidence of each diagnosis
    # sort each diagnosis dict by attention values
    ages = {}
    sexes = {}
    for key in diagnoses.keys():
        cur_diagnosis = diagnoses[key]
        plot_list = []
        age = {}
        sex = {}
        for value in ['age <1', ' age 1-4', 'age 5-14', ' age 15-29', ' age 30-44', ' age 45-59', ' age 60-74',' age >75']:
            age[value] = [0, 0]
        sex['female'] = [0, 0]
        sex['male'] = [0, 0]
        for cd_key in cur_diagnosis.keys():
            if 'age ' in cd_key:
                age[cd_key] = cur_diagnosis[cd_key].copy()
                cur_diagnosis[cd_key][0] = 1
                cur_diagnosis[cd_key][1] = 0
                cur_diagnosis[cd_key] = 0
            elif cd_key == 'male' or cd_key == 'female':
                sex[cd_key] = cur_diagnosis[cd_key].copy()
                cur_diagnosis[cd_key][0] = 1
                cur_diagnosis[cd_key][1] = 0
                cur_diagnosis[cd_key] = 0
            else:
                cur_ev = cur_diagnosis[cd_key]
                average = cur_ev[1] / cur_ev[0]
                cur_diagnosis[cd_key] = average
                plot_list.append(average)
        ages[key] = age
        sexes[key] = sex
        if len(plot_list) > 0:
            sorted_plot_list = sorted(plot_list, reverse=True)
        pat_name = key.replace('/', '')
        plotter.make_bar_chart(key, age, os.path.join(cfg.AgeDiagrams, pat_name + '.png'))
        plotter.make_pie_chart(key, sex, os.path.join(cfg.SexDiagrams, pat_name + '.png'))
        diagnoses[key] = {k: v for k, v in sorted(cur_diagnosis.items(), key=lambda item: item[1], reverse=True)}
    print('sorting of evidences within clinical pictures done')
    with open(os.path.join(cfg.AVG_PATH, 'average_importance.pkl'), 'wb') as f:  # 'wb' mode for binary write
        pkl.dump(diagnoses, f, protocol=pkl.HIGHEST_PROTOCOL)
