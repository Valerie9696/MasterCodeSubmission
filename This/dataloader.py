import networkx as nx
import numpy as np
import os
import random
from scipy.sparse.linalg import eigsh
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


class DDxDataset(Dataset):
    def __init__(self, patient_data, split='train'):
        self.patient_data = patient_data
        self.split = split
        self.graph = patient_data.kg
        self.node_indices = {node: index for index, node in enumerate(self.patient_data.kg.nodes)}
        self.eig_vecs = patient_data.eig_vecs
        if split == 'train':
            self.loader = patient_data.patient_data_train
        elif split == 'valid':
            self.loader = patient_data.patient_data_valid
        elif split == 'test':
            self.loader = patient_data.patient_data_test

        self.en_vocab, self.de_vocab = self.build_vocab()
        cfg.vocab_size = max(self.en_vocab.values())+1
        print('vocab size: ', cfg.vocab_size)

    def drop_rows(self, condition, drop_fraction):
        """Drops a fraction of rows from loader based on a condition function."""
        matching_rows = [row for row in self.loader if condition(row)]
        num_to_drop = int(len(matching_rows) * drop_fraction)
        rows_to_drop = random.sample(matching_rows, num_to_drop)
        return [row for row in self.loader if row not in rows_to_drop]

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        evidences = self.loader[idx][0]
        ddxs = self.loader[idx][1]
        pathology = self.loader[idx][2]
        token_types = self.loader[idx][3]
        num_edges = self.loader[idx][4]
        # build edge tokens and embeddings
        mask = torch.ones(size=[cfg.en_max_len], dtype=torch.bool).to(cfg.device)

        if cfg.use_pre_saved_tensors:
            if self.split == 'train':
                en_pos = torch.load(os.path.join(cfg.PREP_TRAIN_ENC, str(idx)+'.pt'))
            elif self.split == 'test':
                en_pos = torch.load(os.path.join(cfg.PREP_TEST_ENC, str(idx)+'.pt')) # eigenvector of each node + 2 node type embeddings
        else:
            en_pos = torch.zeros([cfg.en_max_len, 970])#cfg.num_eigvecs * 2 + 2])  # eigenvector of each node + 2 node type embeddings
            type_embs = torch.zeros([cfg.en_max_len, cfg.num_eigvecs * 2 + 2])  # eigenvector of each node + 2 node type embeddings
            path_idx = self.node_indices[pathology]
            path_eig_vec = self.eig_vecs[path_idx]
            en_input_strings = []
            for idx, ev in enumerate(evidences):
                type = token_types[idx]
                # if it's a node, concatenate its own row from the eigenvectors twice
                if type == 'node':
                    ev_eig_vec = self.eig_vecs[self.node_indices[ev]]
                    node_pos = torch.tensor(np.concatenate([ev_eig_vec, ev_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)
                    en_pos[idx:] = node_pos
                    en_input_strings.append(ev)
                    type_embs[idx] = 1
                # if it's an antecedent, concat it's ev row and the one of it's corresponding pathology
                elif type == 'ant' or type == 'val':
                    ev_split = ev.split('#')
                    ev_eig_vec = self.eig_vecs[self.node_indices[ev_split[1]]]
                    next_eig_vec = self.eig_vecs[self.node_indices[ev_split[2]]]
                    edge_pos = torch.tensor(np.concatenate([ev_eig_vec, next_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                    en_pos[idx:] = edge_pos
                    en_input_strings.append('edgetoken')
                    if type == 'ant':
                        mask[idx] = 0
                # in case of a symptom the same, but the other way around
                elif type == 'symp':
                    ev_split = ev.split('#')
                    ev_eig_vec = self.eig_vecs[self.node_indices[ev_split[2]]]
                    edge_pos = torch.tensor(np.concatenate([path_eig_vec, ev_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                    en_pos[idx:] = edge_pos
                    mask[idx] = 0
                    en_input_strings.append('edgetoken')
                else:
                    en_input_strings.append(ev)
        path_idx = self.node_indices[pathology]
        path_eig_vec = self.eig_vecs[path_idx]
        path_pos = torch.tensor(np.concatenate([path_eig_vec, path_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)
        if cfg.use_type_embs:
            en_pos = en_pos + type_embs
        if cfg.use_pre_saved_tensors:
            if self.split == 'train':
                ddx_pos = torch.load(os.path.join(cfg.PREP_TRAIN_DEC, str(idx) + '.pt'))
            elif self.split == 'test':
                ddx_pos = torch.load(os.path.join(cfg.PREP_TEST_DEC, str(idx) + '.pt'))
        else:
            ddx_pos = torch.zeros([cfg.de_max_len-1, 970])#cfg.num_eigvecs * 2 + 2])  # eigenvector of each node + 2 node type embeddings
            type_embs = torch.zeros([cfg.de_max_len-1, cfg.num_eigvecs * 2 + 2])  # eigenvector of each node + 2 node type embeddings
            for idx, ddx in enumerate(ddxs):
                if ddx not in ['<bos>', '<eos>', '<pad>']:
                    ddx_eig_vec = self.eig_vecs[self.node_indices[ddx]]
                    node_pos = torch.tensor(np.concatenate([ddx_eig_vec, ddx_eig_vec, [2, 2]]), dtype=torch.float32).to(
                        cfg.device)
                    ddx_pos[idx:] = node_pos
                    type_embs[idx] = 1
        if cfg.use_type_embs:
            ddx_pos = ddx_pos + type_embs
        nodes_end_idx = len(evidences)
        tokens = en_input_strings
        tokens = self.pad_sequence(tokens, cfg.en_max_len)
        en_input = [self.en_vocab[key] for key in tokens]
        de = [self.de_vocab[key] for key in ddxs]
        de_input = de[0:-1]
        #ddx_pos = ddx_pos[0:-1]
        de_output = de[1:]
        pathology = self.de_vocab.get(pathology)

        # convert list to tensor
        en_input = torch.tensor(en_input)
        de_input = torch.tensor(de_input)
        de_output = torch.tensor(de_output)
        pathology = torch.tensor(pathology)


        return en_input, de_input, de_output, pathology, en_pos, ddx_pos, path_pos, mask

    def build_vocab(self):
        antecedents = self.patient_data.antecedents_dict
        pathologies = self.patient_data.conditions_dict
        symptoms = self.patient_data.symptoms_dict
        specials = self.patient_data.specials_dict
        values = self.patient_data.values_dict
        edge_tokens = self.patient_data.edge_tokens_dict
        input_vocab = {**specials, **antecedents, **symptoms, **values, **edge_tokens}
        max_value = max(input_vocab.values())
        for idx, elem in enumerate(['F', 'M', '_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59', '_age_60-74', '_age_>75']):
            input_vocab[elem] = max_value + 1 + idx
        output_vocab = {**specials, **pathologies}
        return input_vocab, output_vocab

    def pad_sequence(self, sequence, pad_len):
        n = pad_len - len(sequence)
        pads = ['<pad>'] * n
        sequence = sequence + pads
        if len(sequence) > cfg.en_max_len:
            sequence = sequence[:cfg.en_max_len]
        return sequence

def load_dataset(batch_size, num_workers, patient_data,):
    train_data = DDxDataset(patient_data, split='train')
    test_data = DDxDataset(patient_data, split='test')

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader
