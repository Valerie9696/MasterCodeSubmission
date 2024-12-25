import csv
import torch
from vocab import build_vocab
from preprocess import parse_patient
from torch.utils.data import Dataset, DataLoader
import ast
import random

import config as cfg


# Define the conditions
def condition_sex_m(row):
    return row['SEX'] == 'M'

def condition_sex_f(row):
    return row['SEX'] == 'F'

def condition_age_20_30(row):
    return 20 <= int(row['AGE']) <= 30

def condition_age_50_60(row):
    return 50 <= int(row['AGE']) <= 60


def process_row(row):
    if 'EVIDENCES' in row:  # Replace with the actual column name
        list_column = row['EVIDENCES']
        initial_evidence = row['INITIAL_EVIDENCE']

        # Convert the string to a list
        data_list = ast.literal_eval(list_column)

        # Filter out the elements to be dropped (30% randomly, but keep 'INITIAL_EVIDENCE')
        num_to_drop = int(len(data_list) * 0.5)
        elements_to_keep = [initial_evidence]
        elements_to_drop = [elem for elem in data_list if elem != initial_evidence]

        if len(elements_to_drop) > 0:
            drop_indices = random.sample(range(len(elements_to_drop)), min(num_to_drop, len(elements_to_drop)))
            elements_to_keep.extend([elem for i, elem in enumerate(elements_to_drop) if i not in drop_indices])

        # Convert the list back to a string
        row['EVIDENCES'] = str(elements_to_keep)


class DDxDataset(Dataset):
    def __init__(self, filename, train=False):
        with open(filename, mode='r', encoding='utf-8') as f:
            self.loader = list(csv.DictReader(f))
            seed = 42
            random.seed(seed)
        if train:
            self.loader = random.sample(self.loader, k=int(0.01 * len(self.loader)))
        if cfg.manipulate_training_data:
            # some early experiments with the dataset
            # randomly sample some patients
            #self.loader = random.sample(self.loader, k=int(0.01 * len(self.loader)))
            # intently break the data
            self.loader = self.drop_rows(condition_sex_f, 0.5)

            # Drop 10% of rows where the value in column age is between 20 and 30
            self.loader = self.drop_rows(condition_age_20_30, 0.2)

            # Drop 30% of rows where the value in column age is between 50 and 60
            self.loader = self.drop_rows(condition_age_50_60, 0.4)
            for row in self.loader:
                process_row(row)
            copy = self.loader.copy()
            for index, row in enumerate(copy):
                list_column = row['EVIDENCES']
                initial_evidence = row['INITIAL_EVIDENCE']
                data_list = ast.literal_eval(list_column)
                evidence_list = [initial_evidence]
                remaining_samples = min(5, len(data_list))
                if remaining_samples > 0:
                    evidence_list += random.sample(data_list, remaining_samples)
                self.loader[index]['EVIDENCES'] = str(evidence_list)
        self.en_vocab, self.de_vocab = build_vocab()

    def drop_rows(self, condition, drop_fraction):
        """Drops a fraction of rows from loader based on a condition function."""
        matching_rows = [row for row in self.loader if condition(row)]
        num_to_drop = int(len(matching_rows) * drop_fraction)
        rows_to_drop = random.sample(matching_rows, num_to_drop)
        return [row for row in self.loader if row not in rows_to_drop]

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        #print('idx: ', idx)
        en_input, de, gt = parse_patient(self.loader[idx], en_max_len=80, de_max_len=41)
        en_input = list(map(lambda x: self.en_vocab.get(x,0), en_input))

        de = list(map(lambda x: self.de_vocab.get(x,0), de))
        de_input = de[0:-1]
        de_output = de[1:]
        pathology = self.de_vocab.get(gt)

        # convert list to tensor
        en_input = torch.tensor(en_input)
        de_input = torch.tensor(de_input)
        de_output = torch.tensor(de_output)

        pathology = torch.tensor(pathology)

        return en_input, de_input, de_output, pathology


def load_dataset(batch_size, num_workers):
    train_data = DDxDataset(filename='data/release_train_patients.csv')#, train=True)
    test_data = DDxDataset(filename='data/release_test_patients.csv')#, train=True)
    en_vocab = {**train_data.en_vocab, **test_data.en_vocab}
    de_vocab = {**train_data.de_vocab, **test_data.de_vocab}

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader, en_vocab, de_vocab


if __name__ == '__main__':
    train_x, test_x, en_vocab, de_vocab = load_dataset(batch_size=256, num_workers=0)

    for en_in, de_in, de_out, path in train_x:
        print(en_in.shape)
        print(de_in.shape)
        print(de_out.shape)
        print(path.shape)
        break

    for en_in, de_in, de_out, path in test_x:
        print(en_in.shape)
        print(de_in.shape)
        print(de_out.shape)
        print(path.shape)
        break
