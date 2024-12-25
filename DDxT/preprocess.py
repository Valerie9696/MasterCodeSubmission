import ast
import os
import pandas as pd
import config as cfg


def parse_age(x):
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


def parse_ddx(x):
    # sort ddx based on most likely to the least likely
    x = sorted(x, key=lambda inputs: -inputs[1])
    # keep only the names
    x = [key for key, _ in x]
    return x


def parse_evidences(x):
    # separating categorical evidences
    x = [item.replace('_@_', ' ') for item in x]
    new = []
    for item in x:
        if ' ' in item:
            split = item.split(' ')
            new.extend(split)
        else:
            new.append(item)
    return new


def parse_pathology(x):
    return x


def pad_sequence(sequence, pad_len):
    n = pad_len - len(sequence)
    pads = ['<pad>'] * n
    sequence = sequence + pads
    return sequence


def parse_patient(x, en_max_len, de_max_len):
    age = int(x['AGE'])
    ddx = ast.literal_eval(x['DIFFERENTIAL_DIAGNOSIS'])
    sex = x['SEX']
    pathology = x['PATHOLOGY']
    evidences = ast.literal_eval(x['EVIDENCES'])

    init_evidence = x['INITIAL_EVIDENCE']

    age = parse_age(age)
    ddx = parse_ddx(ddx)
    evidences = parse_evidences(evidences)
    pathology = parse_pathology(pathology)

    encoder_input = ['<bos>', age, '<sep>', sex, '<sep>', init_evidence, '<sep>', *evidences, '<eos>']
    encoder_input = pad_sequence(encoder_input, pad_len=cfg.en_max_len)

    decoder_input = ['<bos>', *ddx, '<eos>']
    decoder_input = pad_sequence(decoder_input, pad_len=cfg.de_max_len)
    return encoder_input, decoder_input, pathology


if __name__ == '__main__':
    import csv

    filename = 'data/release_test_patients.csv'

    with open(filename, mode='r', encoding='utf-8') as f:
        loader = list(csv.DictReader(f))

    en_input, de_input, gt = parse_patient(loader[0], en_max_len=80, de_max_len=40)
    print(en_input)
    print(de_input)
    print(gt)
