import torch.optim as optim
import config as cfg
from evaluation import mean, evaluate_ddx, evaluate_cls, preprocess_ddx_f1
import dataloader as dl
import graph_builder as gb
import preprocessing as prep
import model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import torch

### change for debugging, so debugger shows tensor sizes
old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    return repr(tensor.shape)[6:] + '  ' + repr(tensor.dtype)[6:] + ' @ ' + str(tensor.device) + '\n' + old_repr(tensor)
torch.Tensor.__repr__ = tensor_info


cfg.vocab_size = 501

network = model.Network(vocab_size=cfg.vocab_size,
                        en_seq_len=cfg.en_seq_len,
                        de_seq_len=cfg.de_seq_len,
                        features=cfg.features,
                        heads=cfg.heads,
                        n_layer=cfg.layers,
                        output_size=cfg.output_size,
                        dropout_rate=cfg.drop_rate).to(cfg.device)
optimizer = optim.Adam(network.parameters())
checkpoint_dir = os.path.join(cfg.CPT_PATH)
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{}.pt'.format(6))
checkpoint = torch.load(checkpoint_path)
epochs, loss_values = [], []

# plot loss curve
for filename in sorted(os.listdir(checkpoint_dir)):
    if filename.endswith(".pth") or filename.endswith(".pt"):  # Check for PyTorch checkpoint files
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path)
        loss = checkpoint.get("loss", None).to('cpu').detach().numpy()
        epoch = checkpoint.get("epoch", None)-1
        if loss is not None and epoch is not None:
            epochs.append(epoch)
            loss_values.append(loss)
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, marker='o', label='Loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(fontsize=14, ticks=epochs)
plt.yticks(fontsize=14)
plt.title('Training Loss Curve', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

network.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

conditions = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.CONDITIONS)).T
evidences = pd.read_json(os.path.join(cfg.DATA_DIR, cfg.EVIDENCES))
train_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TRAIN_SET))
test_df = pd.read_csv(os.path.join(cfg.DATA_DIR, cfg.TEST_SET))
kg = gb.build_graph(conditions=conditions, evidences=evidences)
pg = gb.PatientGraphs(condition_df=conditions, evidences_df=evidences, patient_df_train=train_df,
                      patient_df_test=test_df, tokens_laplacian=True)
if cfg.preprocess_dataset:
    prep.preprocess_dataset(pg, 'train')
    prep.preprocess_dataset(pg, 'test')

train_loader, test_loader = dl.load_dataset(batch_size=cfg.batch_size, num_workers=0, patient_data=pg)

network.eval()
test_acc_ddx, test_acc_cls, test_ddx_probas, test_cls_probas, test_ddx_preds = [], [], [], [], []
reversed_input_vocab = {v: k for k, v in test_loader.dataset.en_vocab.items()}
reversed_output_vocab = {v: k for k, v in test_loader.dataset.de_vocab.items()}
diagnoses = {}
ddxs = {}
for condition in conditions.iterrows():
    diagnoses[condition[1]['condition_name']] = {}
    ddxs[condition[1]['condition_name']] = {}
diagnosis_counter = 0
all_predicted_labels, all_true_labels, all_predicted_ddxs, all_true_ddxs = [], [], [], []
output_vocab = test_loader.dataset.de_vocab
with torch.no_grad():
    for en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask in test_loader:
        en_in, de_in, de_out, path, en_pos, ddx_pos, path_pos, mask = en_in.to(cfg.device), de_in.to(cfg.device), de_out.to(cfg.device), path.to(cfg.device), en_pos.to(cfg.device), ddx_pos.to(cfg.device), path_pos.to(cfg.device), mask.to(cfg.device)
        # forward
        en_in = en_in * mask
        expanded_mask = mask.unsqueeze(-1)
        en_pos = en_pos.masked_fill(~expanded_mask, -1)
        de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in, en_pos=en_pos,
                                                            ddx_pos=ddx_pos, path_pos=path_pos)
        # evaluate
        ddx_acc, ddx_probas, ddx_preds = evaluate_ddx(true=de_out, pred=de_out_pred)
        cls_acc, cls_proba = evaluate_cls(true=path, pred=path_pred)
        predicted_pathologies = torch.argmax(path_pred, dim=-1).cpu()
        f1_true, f1_pred = preprocess_ddx_f1(true=de_out, pred=de_out_pred)
        all_predicted_ddxs.extend(f1_pred)
        all_true_ddxs.extend(f1_true)
        true_labels = path.cpu()

        if all_predicted_labels:
            all_predicted_labels.extend(list(predicted_pathologies))
            all_true_labels.extend(list(true_labels))
        else:
            all_predicted_labels = list(predicted_pathologies)
            all_true_labels = list(true_labels)

        test_ddx_preds.append(ddx_preds)
        test_acc_ddx.append(ddx_acc.item())
        test_acc_cls.append(cls_acc.item())
        test_ddx_probas.append(ddx_probas)
        test_cls_probas.extend(cls_proba)

    test_acc_ddx = mean(test_acc_ddx) * 100
    test_acc_cls = mean(test_acc_cls) * 100

print('DDX Accuracy: ', test_acc_ddx)
print('Classification Accuracy: ', test_acc_cls)
ddx_f1_samples = f1_score(all_true_ddxs, all_predicted_ddxs, average=None)
ddx_f1 = f1_score(all_true_ddxs, all_predicted_ddxs, average='weighted')
cls_f1_samples = f1_score(all_true_labels, all_predicted_labels, average=None)
cls_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
print(len(all_predicted_labels))
print('DDX F1-Score: ', ddx_f1)
print('Classification F1-Score: ', cls_f1)
reversed_output_vocab.pop(50)
reversed_output_vocab.pop(51)
reversed_output_vocab.pop(52)
reversed_output_vocab.pop(53)
reversed_output_vocab.pop(54)
reversed_output_vocab.pop(55)

classes = list(reversed_output_vocab.keys())
precision = precision_score(all_true_labels, all_predicted_labels, average=None)
recall = recall_score(all_true_labels, all_predicted_labels, average=None)
f1 = f1_score(all_true_labels, all_predicted_labels, average=None)
accuracy_per_class = []
for cls in classes:
    true_positives = sum((int(true) == cls and int(pred) == cls) for true, pred in zip(all_true_labels, all_predicted_labels))
    total_class_samples = sum(int(true) == cls for true in all_true_labels)
    accuracy = true_positives / total_class_samples if total_class_samples > 0 else 0
    accuracy_per_class.append(accuracy)
classes = [reversed_output_vocab[i] for i in classes]
metrics_df = pd.DataFrame({
    "Class": classes,
    "Accuracy": accuracy_per_class,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})
print(metrics_df)
metrics_df.reset_index(drop=True, inplace=True)
metrics_df.index = reversed_output_vocab.keys()
metrics_df_latex = metrics_df.to_latex(float_format="%.2f", label="tab:scores_per_class")

# plot confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
sorted_int_labels = sorted(reversed_output_vocab.keys())
class_names = [reversed_output_vocab.get(i, 0) for i in sorted_int_labels]
plt.figure(figsize=(12, 10))
ax = sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
ax.set_ylabel('True Label', labelpad=10)
ax.set_xlabel('Predicted Label', labelpad=10)
ax.set_aspect("equal")
plt.subplots_adjust(left=0.25, bottom=0.35)
plt.title('Confusion Matrix')
plt.savefig('ConfusionMatrix.png')
plt.show()