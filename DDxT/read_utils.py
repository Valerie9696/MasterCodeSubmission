import json


def read_age():
    return ['_age_<1', '_age_1-4', '_age_5-14', '_age_15-29', '_age_30-44', '_age_45-59', '_age_60-74', '_age_>75']


def read_sex():
    return ['M', 'F']


def read_evidences():
    file_name = 'data/release_evidences.json'
    evidence_list = []

    with open(file_name, mode='r', encoding='utf-8') as f:
        data = json.load(f)

    for key, value in data.items():
        if key not in evidence_list:
            evidence_list.append(key)

        categories = list(value['value_meaning'].keys())
        for item in categories:
            if item not in evidence_list:
                evidence_list.append(item)

    return evidence_list


def read_conditions():
    file_name = 'data/release_conditions.json'
    condition_list = []

    with open(file_name, mode='r', encoding='utf-8') as f:
        data = json.load(f)

    for pathology in data.keys():
        condition_list.append(pathology)

    return condition_list


def read_conditions_eng():
    file_name = '../data/release_conditions.json'
    condition_list = []

    with open(file_name, mode='r', encoding='utf-8') as f:
        data = json.load(f)

    for k, v in data.items():
        name = v['cond-name-eng'].split('/')
        condition_list.append(name[0])

    return condition_list


if __name__ == '__main__':
    evidences = read_evidences()
    print(evidences)
    print(f'Total symptoms in the dataset: {len(evidences)}\n')

    pathologies = read_conditions()
    print(pathologies)
    print(f'Total pathologies in the dataset: {len(pathologies)}')
