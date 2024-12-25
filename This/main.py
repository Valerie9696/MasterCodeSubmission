import config as cfg
from evaluation import mean, evaluate_ddx, evaluate_cls, save_history
import dataloader as dl
import graph_builder as gb
import model
import preprocessing as prep

import os
import pandas as pd
import time
import torch
import torch.nn as nn

if __name__ == '__main__':
    conditions = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.CONDITIONS)).T
    evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
    train_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TRAIN_SET))
    test_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TEST_SET))
    prep.translate_values()
    kg = gb.build_graph(conditions=conditions, evidences=evidences)
    pg = gb.PatientGraphs(condition_df=conditions, evidences_df=evidences, patient_df_train=train_df, patient_df_test=test_df, tokens_laplacian=True)
    if cfg.preprocess_dataset:
        prep.preprocess_dataset(pg, 'train')
        prep.preprocess_dataset(pg, 'test')

    train_loader, test_loader = dl.load_dataset(batch_size=cfg.batch_size, num_workers=0, patient_data=pg)
    network = model.Network(vocab_size=cfg.vocab_size,
                      en_seq_len=cfg.en_seq_len,
                      de_seq_len=cfg.de_seq_len,
                      features=cfg.features,
                      heads=cfg.heads,
                      n_layer=cfg.layers,
                      output_size=cfg.output_size,
                      dropout_rate=cfg.drop_rate).to(cfg.device)

    epochs = 5
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    print('Start training ...')
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc_ddx, train_acc_cls = [], [], []

        tic = time.time()
        checkpoint_dir = os.path.join(cfg.CPT_PATH)
        # train
        network.train()

        for en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask in train_loader:
            en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask = en_in.to(cfg.device), de_in.to(cfg.device), de_out.to(cfg.device), path.to(cfg.device), en_pos.to(cfg.device), ddx_pos.to(cfg.device), path_pos.to(cfg.device), mask.to(cfg.device)

            optimizer.zero_grad()
            de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in, en_pos=en_pos, ddx_pos=ddx_pos, path_pos=path_pos)
            loss1 = loss_function(de_out_pred.permute(0, 2, 1), de_out)
            loss2 = loss_function(path_pred, path)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            # evaluate
            ddx_acc, ddx_probas, ddx_pred = evaluate_ddx(true=de_out, pred=de_out_pred)
            cls_acc, cls_probas = evaluate_cls(true=path, pred=path_pred)
            train_loss.append(loss.item())
            train_acc_ddx.append(ddx_acc.item())
            train_acc_cls.append(cls_acc.item())
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pt'.format(epoch + 1)))

        train_loss = mean(train_loss)
        train_acc_ddx = mean(train_acc_ddx) * 100
        train_acc_cls = mean(train_acc_cls) * 100

        # test
        network.eval()
        test_acc_ddx, test_acc_cls, test_ddx_probas, test_cls_probas = [], [], [], []
        with torch.no_grad():
            counter = 0
            for en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask in test_loader:
                en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask = en_in.to(cfg.device), de_in.to(
                    cfg.device), de_out.to(cfg.device), path.to(cfg.device), en_pos.to(cfg.device), ddx_pos.to(
                    cfg.device), path_pos.to(cfg.device), mask.to(cfg.device)
                # forward
                en_in = en_in * mask
                expanded_mask = mask.unsqueeze(-1)
                en_pos = en_pos.masked_fill(~expanded_mask, 0)
                en_pos = en_pos * expanded_mask
                de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in,
                                                                                             de_input=de_in,
                                                                                             en_pos=en_pos,
                                                                                             ddx_pos=ddx_pos,
                                                                                             path_pos=path_pos)
                # evaluate
                ddx_acc, ddx_probas, ddx_pred = evaluate_ddx(true=de_out, pred=de_out_pred)
                cls_acc, cls_proba = evaluate_cls(true=path, pred=path_pred)
                test_acc_ddx.append(ddx_acc.item())
                test_acc_cls.append(cls_acc.item())
                test_ddx_probas.append(ddx_probas)
                test_cls_probas.append(cls_proba)

        test_acc_ddx = mean(test_acc_ddx) * 100
        test_acc_cls = mean(test_acc_cls) * 100
        toc = time.time()

        history.append(f'Epoch: {epoch}/{epochs}, train loss: {train_loss:>6.4f}, train ddx acc: {train_acc_ddx:.2f}%, '
                       f'train cls acc: {train_acc_cls:.2f}%, test ddx acc: {test_acc_ddx:.2f}%, '
                       f'test cls acc: {test_acc_cls:.2f}%')
        print(history[-1])
        torch.save(network.state_dict(), os.path.join(cfg.WEIGHTS_PATH, f'.model_{epoch}.h5'))
        scheduler.step()

    save_history(os.path.join(cfg.WEIGHTS_PATH, 'history.csv'), history, mode='w')
    print('All Done!')