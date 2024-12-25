import ast
import config as cfg
import networkx as nx
import random
import torch
from torch_geometric.data import Data
import numpy as np

class PatientGraphs:
    """
    Class which contains everything needed to build the patient graphs.
    """
    def __init__(self, condition_df, evidences_df, patient_df_train, patient_df_test=None, tokens_laplacian = False):
        self.antecedents_dict, self.conditions_dict, self.symptoms_dict, self.specials_dict, self.values_dict, self.id_count = self.make_id_dicts(condition_df=condition_df, evidences_df=evidences_df)
        self.edge_tokens_dict = {}
        self.patient_df_train = patient_df_train
        self.patient_df_test = patient_df_test
        self.kg = self.build_kg_graph(condition_df, evidences_df)
        self.eig_vals, self.eig_vecs = self.make_eigenvectors()
        if tokens_laplacian:
            if cfg.decrease_training_data:
                random.seed(42)
                self.patient_df_train = self.patient_df_train.sample(frac=0.5, random_state=42)
                copy = self.patient_df_train.copy()
                for index, row in copy.iterrows():
                    list_column = row['EVIDENCES']
                    initial_evidence = row['INITIAL_EVIDENCE']
                    data_list = ast.literal_eval(list_column)
                    evidence_list = [initial_evidence]
                    remaining_samples = min(5, len(data_list))
                    if remaining_samples > 0:
                        evidence_list += random.sample(data_list, remaining_samples)
                    self.patient_df_train.at[index, 'EVIDENCES'] = str(evidence_list)

                copy = self.patient_df_test.copy()
                evidence_count_per_patient = []
                for index, row in copy.iterrows():
                    list_column = row['EVIDENCES']
                    initial_evidence = row['INITIAL_EVIDENCE']
                    data_list = ast.literal_eval(list_column)
                    evidence_list = [initial_evidence]
                    evidence_count_per_patient.append(len(data_list))
                    remaining_samples = min(5, len(data_list))
                    if remaining_samples > 0:
                        evidence_list += random.sample(data_list, remaining_samples)
                    self.patient_df_test.at[index, 'EVIDENCES'] = str(evidence_list)
                self.patient_data_train = self.build_sequence(self.patient_df_train)
                average_evidence_count = np.mean(evidence_count_per_patient)
                self.patient_data_test = self.build_sequence(self.patient_df_test)
            else:
                self.patient_data_train = self.build_sequence(self.patient_df_train[:1000])
                self.patient_data_test = self.build_sequence(self.patient_df_test[:1000])

    def make_eigenvectors(self):
        """
        make the symmetric laplacian eigenvectors
        :return: eigenvalues & eigenvectors
        """
        print('computing eigenvectors')
        A = nx.to_numpy_array(self.kg)
        in_degree = np.sum(A, axis=1)
        N = np.diag(in_degree.clip(1) ** -0.5)
        L = np.eye(A.shape[0]) - N @ A @ N
        eig_vals, eig_vecs = np.linalg.eigh(L)
        print('eigenvectors done')
        return eig_vals, eig_vecs

    def make_id_dicts(self, condition_df, evidences_df):
        """
        make the dictionaries for symptoms, antecedents, their values, and pathologies
        :param condition_df: pathologies
        :param evidences_df: evidences (symptoms and antecedents)
        :return:
        """
        conditions_dict = {}
        antecedents_dict = {}
        symptoms_dict = {}
        value_dict = {}
        counter = 1
        for _, row in condition_df.iterrows():
            conditions_dict[row['condition_name']] = counter
            counter += 1
        specials = ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>', '<edge>']
        specials_dict = {}
        for item in specials:
            specials_dict[item] = counter
            counter += 1
        for _, row in evidences_df.T.iterrows():
            if row['is_antecedent']:
                antecedents_dict[row['name']] = counter
            else:
                symptoms_dict[row['name']] = counter
            counter += 1
        df = evidences_df.T
        df['possible-values'] = df['possible-values'].apply(
            lambda x: [str(i) if isinstance(i, int) else i for i in x]
        )
        exploded_df = df.explode('possible-values')
        unique_values = exploded_df['possible-values'].unique()
        unique_values = unique_values[1:]   # drop the nan value
        for uv in unique_values:
            value_dict[uv] = counter
            counter += 1
        return antecedents_dict, conditions_dict, symptoms_dict, specials_dict, value_dict, counter

    def pad_sequence(self, sequence, pad_len):
        """
        padding
        :param sequence:
        :param pad_len:
        :return:
        """
        n = pad_len - len(sequence)
        pads = ['<pad>'] * n
        sequence = sequence + pads
        return sequence

    def pad_vector(self, sequence, max_len):
        """
        padding
        :param sequence:
        :param max_len:
        :return:
        """
        n = max_len - len(sequence)
        zero_vec = torch.tensor([0]*len(sequence[0]), dtype=torch.float32).to(cfg.device)
        pads = [zero_vec] * n
        sequence = sequence + pads
        return torch.stack(sequence)


    def parse_age(self, x):
        """
        parse age to age groups (from DDxT code)
        :param x: age value
        :return:
        """
        if x < 1:
            return '_age_<1'
        elif x <= 4:
            return '_age_1-4'
        elif x <= 14:
            return '_age_5-14'
        elif x <= 29:
            return '_age_15-29'
        elif x <= 44:
            return '_age_30-44'
        elif x <= 45:
            return '_age_45-59'
        elif x <= 74:
            return '_age_60-74'
        else:
            return '_age_>75'


    def build_sequence(self, df):
        """
        Input sequence to the transformer should be of shape:
        <bos> sex <sep> evidences with values (nodes) <sep> ... in the dataloader to be extended with edges
        :param df: patient data
        :return:
        """
        patients = []
        self.edge_tokens_dict['edgetoken'] = self.id_count
        self.id_count += 1
        for _, row in df.iterrows():
            nodes = ['<bos>']
            edges = []
            token_types = ['<bos>', 'gender', '<sep>', 'age', '<sep>']
            age = int(row['AGE'])
            age = self.parse_age(age)
            sex = row['SEX']
            nodes.append(sex)
            nodes.append('<sep>')
            nodes.append(age)
            nodes.append('<sep>')
            pathology = row['PATHOLOGY']
            ddx = ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS'])
            x = sorted(ddx, key=lambda inputs: -inputs[1])
            # keep only the names and replace space with underscore
            ddxs = [key for key, _ in x]
            evidences = ast.literal_eval(row['EVIDENCES'])
            edge_types = []
            for ev in evidences:
                if '@' in ev:
                    ev_val = ev.split('_@_')
                    ev_sep = ev_val[0]
                    val = ev_val[1]
                    nodes.append(ev_sep)
                    token_types.append('node')
                    nodes.append(val)
                    token_types.append('node')
                    if ev_sep in self.antecedents_dict:
                        edge_types.append('ant')
                        e = 'edgetoken#' + ev_sep + '#' + pathology
                    else:
                        edge_types.append('symp')
                        e = 'edgetoken#' + pathology + '#' + ev_sep
                    if e not in self.edge_tokens_dict.keys():
                        self.id_count += 0
                    edges.append(e)
                    edge_types.append('val')
                    e = 'edgetoken#' + ev_sep + '#' + val
                    if e not in self.edge_tokens_dict.keys():
                        self.id_count += 0
                    edges.append(e)
                else:
                    ev_sep = ev
                    nodes.append(ev_sep)
                    token_types.append('node')
                    if ev_sep in self.antecedents_dict:
                        edge_types.append('ant')
                        e = 'edgetoken#' + ev_sep + '#' + pathology
                        edges.append(e)
                        if e not in self.edge_tokens_dict.keys():
                            self.id_count += 0
                    else:
                        edge_types.append('symp')
                        e = 'edgetoken#' + pathology + '#' + ev_sep
                        edges.append(e)
                        if e not in self.edge_tokens_dict.keys():
                            self.id_count += 0
            nodes.append('<sep>')
            nodes.extend(edges)
            nodes.append('<eos>')
            token_types.append('<sep>')
            token_types.extend(edge_types)
            token_types.append('<eos>')
            patient = (nodes, self.pad_sequence(['<bos>', *ddxs, '<eos>'], cfg.de_max_len), pathology, token_types, len(edges))
            patients.append(patient)
        return patients

    def build_kg_graph(self, conditions, evidences):
        kg = nx.Graph()
        for _, row in conditions.iterrows():
            condition = _
            antecedents = list(row['antecedents'].keys())
            symptoms = list(row['symptoms'].keys())
            for ant in antecedents:
                kg.add_edge(ant, condition)
            for symp in symptoms:
                kg.add_edge(condition, symp)
        kg.add_node('True')
        kg.add_node('False')
        for _, row in evidences.T.iterrows():
            a=0
            if row['data_type'] == 'M':
                values = row['possible-values']
                for value in values:
                    kg.add_node(value)
                    kg.add_edge(_, value)
            if row['data_type'] == 'C':
                values = row['possible-values']
                for value in values:
                    kg.add_node(str(value))
                    kg.add_edge(_, str(value))
            else:
                kg.add_edge(_, 'True')
                kg.add_edge(_, 'False')
        return kg


def build_graph(conditions, evidences):
    nodes = set()
    edges = []
    for _, row in conditions.iterrows():
        condition = _
        antecedents = list(row['antecedents'].keys())
        symptoms = list(row['symptoms'].keys())
        nodes.update(antecedents)
        nodes.update(symptoms)
        nodes.add(condition)
        for ant in antecedents:
            edges.append((ant, condition))
        for symp in symptoms:
            edges.append((condition, symp))

    node_list = list(nodes)
    node_index = {node: i for i, node in enumerate(node_list)}

    row_indices = [node_index[edge[0]] for edge in edges]
    col_indices = [node_index[edge[1]] for edge in edges]
    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.float32).to(cfg.device)
    torch_graph = Data(num_nodes=len(nodes), edge_index=edge_index)
    return torch_graph
