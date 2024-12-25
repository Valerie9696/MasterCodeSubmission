import time
import config as cfg
import torch
from network import Network
from dataset import load_dataset
from utils import mean, evaluate_ddx, evaluate_cls, save_history
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA available')

print('Loading data & network ...')
train_loader, test_loader, en_vocab, de_vocab = load_dataset(batch_size=cfg.batch_size, num_workers=0)

network = Network(vocab_size=cfg.vocab_size,
                  en_seq_len=cfg.en_seq_len,
                  de_seq_len=cfg.de_seq_len,
                  features=cfg.features,
                  heads=cfg.heads,
                  n_layer=cfg.layers,
                  output_size=cfg.output_size,
                  dropout_rate=cfg.drop_rate).to(cfg.device)

# network.load_state_dict(torch.load('./weights/model.h5'))

epochs = 5
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
print('Start training ...')
history = []

for epoch in range(1, epochs + 1):
    train_loss, train_acc_ddx, train_acc_cls = [], [], []

    tic = time.time()
    checkpoint_dir = os.path.join('cpts')

    # train
    network.train()
    counter = 0
    for en_in, de_in, de_out, path in train_loader:
        counter += 1
        en_in, de_in, de_out, path = en_in.to(cfg.device), de_in.to(cfg.device), de_out.to(cfg.device), path.to(cfg.device)
        #en_in, de_in, de_out, path = en_in, de_in, de_out, path
        optimizer.zero_grad()

        # forward + loss + backward + optimize
        de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in)
        loss1 = loss_function(de_out_pred.permute(0, 2, 1), de_out)
        loss2 = loss_function(path_pred, path)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        # evaluate
        ddx_acc, pred_sum, ddx_topk = evaluate_ddx(true=de_out, pred=de_out_pred)
        cls_acc, topk = evaluate_cls(true=path, pred=path_pred)
        train_loss.append(loss.item())
        train_acc_ddx.append(ddx_acc.item())
        train_acc_cls.append(cls_acc.item())

    train_loss = mean(train_loss)
    train_acc_ddx = mean(train_acc_ddx) * 100
    train_acc_cls = mean(train_acc_cls) * 100

    # test
    network.eval()
    test_acc_ddx, test_acc_cls, test_topk = [], [], []
    with torch.no_grad():
        for en_in, de_in, de_out, path in test_loader:
            en_in, de_in, de_out, path = en_in.to(cfg.device), de_in.to(cfg.device), de_out.to(cfg.device), path.to(cfg.device)
            #en_in, de_in, de_out, path = en_in, de_in, de_out, path #marker
            # de_out = one_hot(de_out, output_size)

            # forward
            de_out_pred, path_pred, en_attention_weights, de_attention_weights = network(en_input=en_in, de_input=de_in)

            #evaluate
            ddx_acc, pred_sum, ddx_topk = evaluate_ddx(true=de_out, pred=de_out_pred)
            cls_acc, logits = evaluate_cls(true=path, pred=path_pred)
            test_acc_ddx.append(ddx_acc.item())
            test_acc_cls.append(cls_acc.item())

    test_acc_ddx = mean(test_acc_ddx) * 100
    test_acc_cls = mean(test_acc_cls) * 100
    toc = time.time()

    history.append(f'Epoch: {epoch}/{epochs}, train loss: {train_loss:>6.4f}, train ddx acc: {train_acc_ddx:.2f}%, '
    f'train cls acc: {train_acc_cls:.2f}%, test ddx acc: {test_acc_ddx:.2f}%, '
    f'test cls acc: {test_acc_cls:.2f}%, eta: {toc - tic:.2f}s')
    print(history[-1])

    torch.save(network.state_dict(), os.path.join(cfg.WEIGHTS_PATH, f'model_{epoch}.h5'))#f'./weights/model_{epoch}.h5')
    scheduler.step()

save_history(os.path.join(cfg.WEIGHTS_PATH,'history.csv'), history, mode='w')
print('All Done!')
