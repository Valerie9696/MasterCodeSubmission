import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle as pkl

import random
from scipy.sparse.linalg import eigsh
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


def translate_values():
    """
    make a dictionary with value names
    :return:
    """
    evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
    value_dict = {}
    for item in evidences.T.iterrows():
        if item[1]['data_type'] != 'B':
            value_dict.update(item[1]['value_meaning'])

    with open(os.path.join(cfg.DATA_DIR, cfg.VALUES), 'wb') as pickle_file:
        pkl.dump(value_dict, pickle_file)



class DDxDataset(Dataset):
    def __init__(self, patient_data, split='train'):
        self.patient_data = patient_data
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
        # build edge tokens and embeddings
        en_pos = torch.zeros([cfg.en_max_len, cfg.num_eigvecs*2+2]) # eigenvector of each node + 2 node type embeddings
        path_idx = self.node_indices[pathology]
        path_eig_vec = self.eig_vecs[path_idx]
        path_pos = torch.tensor(np.concatenate([path_eig_vec, path_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)

        edges = []
        edge_token = '<edge>'
        for idx, ev in enumerate(evidences):
            edges.append(edge_token)
            type = token_types[idx]
            if ev == '<edge>':
                a=0
            if type not in ['<bos>', '<sep>', '<eos>', 'gender', 'age']:
                ev_eig_vec = self.eig_vecs[self.node_indices[ev]]
                node_pos = torch.tensor(np.concatenate([ev_eig_vec, ev_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)
                en_pos[idx:] = node_pos
            if type == 'ant':
                edge_pos = torch.tensor(np.concatenate([ev_eig_vec, path_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                en_pos[len(evidences)+idx:] = edge_pos
            if type == 'symp':
                edge_pos = torch.tensor(np.concatenate([path_eig_vec, ev_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                en_pos[len(evidences)+idx:] = edge_pos
            if type == 'val':
                prev_ev = evidences[idx-1]
                edge_pos = torch.tensor(np.concatenate([self.eig_vecs[self.node_indices[prev_ev]], ev_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                en_pos[len(evidences)+idx:] = edge_pos

        ddx_pos = torch.zeros([cfg.de_max_len, cfg.num_eigvecs*2+2]) # eigenvector of each node + 2 node type embeddings
        for idx, ddx in enumerate(ddxs):
            if ddx not in ['<bos>', '<eos>', '<pad>']:
                ddx_eig_vec = self.eig_vecs[self.node_indices[ddx]]
                node_pos = torch.tensor(np.concatenate([ddx_eig_vec, ddx_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)
                ddx_pos[idx:] = node_pos
        nodes_end_idx = len(evidences)
        tokens = evidences + edges
        tokens.append('<eos>')
        tokens = self.pad_sequence(evidences, cfg.en_max_len)
        en_input = [self.en_vocab[key] for key in tokens]
        de = [self.de_vocab[key] for key in ddxs]
        de_input = de[0:-1]
        ddx_pos = ddx_pos[0:-1]
        de_output = de[1:]
        pathology = self.de_vocab.get(pathology)

        # convert list to tensor
        en_input = torch.tensor(en_input)
        de_input = torch.tensor(de_input)
        de_output = torch.tensor(de_output)
        pathology = torch.tensor(pathology)

        mask = torch.ones(size=[cfg.en_max_len], dtype=torch.bool).to(cfg.device)
        mask[nodes_end_idx:nodes_end_idx + len(edges)] = False

        return en_input, de_input, de_output, pathology, en_pos, ddx_pos, path_pos, mask

    def build_vocab(self):
        antecedents = self.patient_data.antecedents_dict
        pathologies = self.patient_data.conditions_dict
        symptoms = self.patient_data.symptoms_dict
        specials = self.patient_data.specials_dict
        values = self.patient_data.values_dict
        input_vocab = {**specials, **antecedents, **symptoms, **values}
        max_value = max(input_vocab.values())
        input_vocab['F'] = max_value + 1
        input_vocab['M'] = max_value + 2
        output_vocab = {**specials, **pathologies}
        return input_vocab, output_vocab

    def pad_sequence(self, sequence, pad_len):
        n = pad_len - len(sequence)#max_len - len(sequence.split(' '))
        pads = ['<pad>'] * n
        sequence = sequence + pads
        return sequence


def pad_sequence(sequence, pad_len):
    n = pad_len - len(sequence)#max_len - len(sequence.split(' '))
    pads = ['<pad>'] * n
    sequence = sequence + pads
    return sequence


def preprocess_dataset(patient_data, split):
    node_indices = {node: index for index, node in enumerate(patient_data.kg.nodes)}
    if split == 'train':
        data = patient_data.patient_data_train
        save_path_enc = cfg.PREP_TRAIN_ENC
        save_path_dec = cfg.PREP_TRAIN_DEC
    elif split == 'test':
        data = patient_data.patient_data_test
        save_path_enc = cfg.PREP_TEST_ENC
        save_path_dec = cfg.PREP_TEST_DEC
    for index, patient in enumerate(data):
        evidences = patient[0]
        ddxs = patient[1]
        pathology = patient[2]
        token_types = patient[3]
        # build edge tokens and embeddings
        en_pos = torch.zeros([cfg.en_max_len, cfg.num_eigvecs * 2 + 2])  # eigenvector of each node + 2 node type embeddings
        path_idx = node_indices[pathology]
        path_eig_vec = patient_data.eig_vecs[path_idx]
        edges = []
        edge_token = '<edge>'
        for idx, ev in enumerate(evidences):
            edges.append(edge_token)
            type = token_types[idx]
            if type not in ['<bos>', '<sep>', '<eos>', 'gender', 'age']:
                ev_eig_vec = patient_data.eig_vecs[node_indices[ev]]
                node_pos = torch.tensor(np.concatenate([ev_eig_vec, ev_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)
                en_pos[idx:] = node_pos
            if type == 'ant':
                edge_pos = torch.tensor(np.concatenate([ev_eig_vec, path_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                en_pos[len(evidences) + idx:] = edge_pos
            if type == 'symp':
                edge_pos = torch.tensor(np.concatenate([path_eig_vec, ev_eig_vec, [1, 1]]), dtype=torch.float32).to(cfg.device)
                en_pos[len(evidences) + idx:] = edge_pos
            if type == 'val':
                prev_ev = evidences[idx - 1]
                edge_pos = torch.tensor(np.concatenate([patient_data.eig_vecs[node_indices[prev_ev]], ev_eig_vec, [1, 1]]),dtype=torch.float32).to(cfg.device)
                en_pos[len(evidences) + idx:] = edge_pos
        torch.save(en_pos, os.path.join(save_path_enc, str(index)+'.pt'))
        ddx_pos = torch.zeros([cfg.de_max_len, cfg.num_eigvecs * 2 + 2])  # eigenvector of each node + 2 node type embeddings
        for idx, ddx in enumerate(ddxs):
            if ddx not in ['<bos>', '<eos>', '<pad>']:
                ddx_eig_vec = patient_data.eig_vecs[node_indices[ddx]]
                node_pos = torch.tensor(np.concatenate([ddx_eig_vec, ddx_eig_vec, [2, 2]]), dtype=torch.float32).to(cfg.device)
                ddx_pos[idx:] = node_pos
        tokens = evidences + edges
        tokens.append('<eos>')
        ddx_pos = ddx_pos[0:-1]
        torch.save(ddx_pos, os.path.join(save_path_dec, str(index)+'.pt'))
