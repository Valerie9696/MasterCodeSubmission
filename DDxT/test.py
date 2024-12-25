import os
import time
import torch
import numpy as np
from network import Network
from dataset import load_dataset
from utils import mean, evaluate_ddx, evaluate_cls, preprocess_ddx_f1
from sklearn.metrics import confusion_matrix, f1_score

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
                  dropout_rate=cfg.drop_rate).cuda()

network.load_state_dict(torch.load(os.path.join(cfg.WEIGHTS_PATH, f'model_5.h5')))

print('Start testing ...')

# test
network.eval()
test_acc_ddx, test_acc_cls = [], []
tic = time.time()

np_true_ddx = []
np_pred_ddx = []

np_true_cls = []
np_pred_cls = []
all_predicted_labels, all_true_labels, all_predicted_ddxs, all_true_ddxs = [], [], [], []

with torch.no_grad():
    for n, (en_in, de_in, de_out, path) in enumerate(test_loader):
        en_in, de_in, de_out, path = en_in.cuda(), de_in.cuda(), de_out.cuda(), path.cuda()
        # de_out = one_hot(de_out, output_size)

        # forward
        de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in)

        # store
        np_true_ddx.append(de_out.detach().cpu().numpy())
        np_pred_ddx.append(torch.argmax(de_out_pred, dim=-1).detach().cpu().numpy())
        np_true_cls.append(path.detach().cpu().numpy())
        np_pred_cls.append(torch.argmax(path_pred, dim=-1).detach().cpu().numpy())

        # evaluate
        ddx_acc, pred_sum, ddx_topk = evaluate_ddx(true=de_out, pred=de_out_pred)
        cls_acc, logits = evaluate_cls(true=path, pred=path_pred)
        test_acc_ddx.append(ddx_acc.item())
        test_acc_cls.append(cls_acc.item())
        f1_true, f1_pred = preprocess_ddx_f1(true=de_out, pred=de_out_pred)
        all_predicted_ddxs.extend(f1_pred)
        all_true_ddxs.extend(f1_true)
        true_labels = path.cpu()
        predicted_pathologies = torch.argmax(path_pred, dim=-1).cpu()
        if all_predicted_labels:
            all_predicted_labels.extend(list(predicted_pathologies))
            all_true_labels.extend(list(true_labels))
        else:
            all_predicted_labels = list(predicted_pathologies)
            all_true_labels = list(true_labels)

test_acc_ddx = mean(test_acc_ddx) * 100
test_acc_cls = mean(test_acc_cls) * 100
toc = time.time()

print(f'test ddx acc: {test_acc_ddx:.2f}%, test cls acc: {test_acc_cls:.2f}%, eta: {toc - tic:.2}s')

np_true_ddx = np.concatenate(np_true_ddx, dtype=np.float32)
np_pred_ddx = np.concatenate(np_pred_ddx, dtype=np.float32)
np_true_cls = np.concatenate(np_true_cls, dtype=np.float32)
np_pred_cls = np.concatenate(np_pred_cls, dtype=np.float32)

print(np_true_ddx.shape)
print(np_pred_ddx.shape)
print(np_true_cls.shape)
print(np_pred_cls.shape)

ddx_f1 = f1_score(all_true_ddxs, all_predicted_ddxs, average='weighted')
cls_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')

print('DDX F1-Score: ', ddx_f1)
print('Classification F1-Score: ', cls_f1)

# save file
if not os.path.exists('results'):
    os.mkdir('results')
np.save(os.path.join('results','true_ddx.npy'), np_true_ddx)
np.save(os.path.join('results','pred_ddx.npy'), np_pred_ddx)
np.save(os.path.join('results','true_cls.npy'), np_true_cls)
np.save(os.path.join('results','pred_cls.npy'), np_pred_cls)

print('All Done!')
