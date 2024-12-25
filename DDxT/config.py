import os
import torch

DATA_DIR = os.path.join('') # set this path to the directory you want to save the data in
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
# first put conditions.json, evidences.json, test.csv, train.csv and validate.csv into the Dataset directory

TRAIN_SET = 'train.csv'
VALIDATION_SET = 'validate.csv'
TEST_SET = 'test.csv'
SAMPLED = 'sampled_train.csv'
CONDITIONS = 'conditions.json'
EVIDENCES = 'evidences.json'
VALUES = 'values.pkl'
CUSTOM_DATASET = 'custom_dataset.pt'
PREPROCESSED_DATA_DIR = os.path.join('Dataset', 'preprocessed_data')
EN_ATTENTION_WEIGHTS_TYPE = 0# 0 for sum, 1 for mean, 2 for max, 3 for full
DE_ATTENTION_WEIGHTS_TYPE = 3# 0 for sum, 1 for mean, 2 for max, 3 for full

############################ Data ##################################

batch_size = 64
preprocess_dataset = False
make_plots = True
version_name = 'final_full'
token_add = True
use_pre_saved_tensors = False
use_type_embs = False
decrease_training_data = False
manipulate_training_data = False

if not os.path.exists(os.path.join(DATA_DIR, 'preprocessed_data')) and preprocess_dataset:
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'train'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'train', 'enc_pos'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'train', 'dec_pos'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'test'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'test', 'enc_pos'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'test', 'dec_pos'))
    os.mkdir(os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'validate'))

if not os.path.exists('cpts'):
    os.mkdir('cpts')
CPT_PATH = os.path.join('cpts', version_name)

if not os.path.isdir(CPT_PATH):
    os.mkdir(CPT_PATH)

if not os.path.isdir('weights'):
    os.mkdir('weights')

WEIGHTS_PATH = os.path.join('weights', version_name)
if not os.path.isdir(WEIGHTS_PATH):
    os.mkdir(WEIGHTS_PATH)


if not os.path.exists(os.path.join(DATA_DIR, 'Plots')):
    os.mkdir(os.path.join(DATA_DIR, 'Plots'))

PLOTS_PATH = os.path.join(DATA_DIR, 'Plots', version_name)
if not os.path.exists(PLOTS_PATH):
    os.mkdir(PLOTS_PATH)

if not os.path.exists(os.path.join(PLOTS_PATH, 'sum')):
    os.mkdir(os.path.join(PLOTS_PATH, 'sum'))
    os.mkdir(os.path.join(PLOTS_PATH, 'sum', 'ddx_graphs'))
    os.mkdir(os.path.join(PLOTS_PATH, 'sum', 'pathology_graphs'))

if not os.path.exists(os.path.join(PLOTS_PATH, 'mean')):
    os.mkdir(os.path.join(PLOTS_PATH, 'mean'))
    os.mkdir(os.path.join(PLOTS_PATH, 'mean', 'ddx_graphs'))
    os.mkdir(os.path.join(PLOTS_PATH, 'mean', 'pathology_graphs'))

if not os.path.exists(os.path.join(PLOTS_PATH, 'max')):
    os.mkdir(os.path.join(PLOTS_PATH, 'max'))
    os.mkdir(os.path.join(PLOTS_PATH, 'max', 'ddx_graphs'))
    os.mkdir(os.path.join(PLOTS_PATH, 'max', 'pathology_graphs'))

if not os.path.exists(os.path.join(PLOTS_PATH, 'full')):
    os.mkdir(os.path.join(PLOTS_PATH, 'full'))
    os.mkdir(os.path.join(PLOTS_PATH, 'full', 'ddx_graphs'))
    os.mkdir(os.path.join(PLOTS_PATH, 'full', 'pathology_graphs'))

if not os.path.exists(os.path.join(DATA_DIR, 'Diagrams')):
    os.mkdir(os.path.join(DATA_DIR, 'Diagrams'))

DiagramsPath = os.path.join(DATA_DIR, 'Diagrams', version_name)
if not os.path.exists(DiagramsPath):
    os.mkdir(DiagramsPath)

AgeDiagrams = os.path.join(DiagramsPath, 'Age')
if not os.path.exists(AgeDiagrams):
    os.mkdir(AgeDiagrams)

SexDiagrams = os.path.join(DiagramsPath, 'Sex')
if not os.path.exists(SexDiagrams):
    os.mkdir(SexDiagrams)

if not os.path.exists(os.path.join(DATA_DIR, 'Explanations')):
    os.mkdir(os.path.join(DATA_DIR, 'Explanations'))

EX_PATH = os.path.join(DATA_DIR, 'Explanations', version_name)
if not os.path.exists(EX_PATH):
    os.mkdir(EX_PATH)

if not os.path.exists(os.path.join(DATA_DIR, 'Averages')):
    os.mkdir(os.path.join(DATA_DIR, 'Averages'))

AVG_PATH = os.path.join(DATA_DIR, 'Averages', version_name)
if not os.path.exists(AVG_PATH):
    os.mkdir(AVG_PATH)

if not os.path.exists(os.path.join(DATA_DIR, 'Charts')):
    os.mkdir(os.path.join(DATA_DIR, 'Charts'))

CHART_PATH = os.path.join(DATA_DIR, 'Charts', version_name)
if not os.path.exists(os.path.join(DATA_DIR, 'Charts', version_name)):
    os.mkdir(CHART_PATH)

if not os.path.exists(os.path.join(DATA_DIR, 'Charts', version_name, 'ddx_graphs')):
    os.mkdir(os.path.join(DATA_DIR, 'Charts', version_name, 'ddx_graphs'))

if not os.path.exists(os.path.join(DATA_DIR, 'Charts', version_name, 'pathology_graphs')):
    os.mkdir(os.path.join(DATA_DIR, 'Charts', version_name, 'pathology_graphs'))

PREP_TRAIN_ENC = os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'train', 'enc_pos')
PREP_TRAIN_DEC = os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'train', 'dec_pos')
PREP_TEST_ENC = os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'test', 'enc_pos')
PREP_TEST_DEC = os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'test', 'dec_pos')
PREP_VALIDATE = os.path.join(DATA_DIR, 'preprocessed_data', version_name, 'validate')

########################## Embeddings ##############################

vocab_size = 501
dense_setting = False
embedding_types = ['laplacian']
lap_node_id_dim = 128
num_eigvecs = 13
multi_hop_max_dist = 2

####################### Model Parameters ###########################

if torch.cuda.is_available():
    device = 'cuda'
    print('cuda is available')
else:
    device = 'cpu'
    print('cuda is not available')

dim_hidden = 1024
en_max_len = 80
de_max_len = 40
en_seq_len = 80
de_seq_len = 40
if token_add:
    features_extended = 128
else:
    features_extended = 128 + 2 * num_eigvecs + 2
features = 128
heads = 4
layers = 6
output_size = 54
drop_rate = 0.1

skip_edges = False
