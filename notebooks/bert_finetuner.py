import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from transformers import AutoModel, AutoTokenizer
import numpy as np


def get_devices():
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
    else:
        gpu = torch.device("cpu")
    cpu = torch.device("cpu")
    return gpu, cpu


def plot_histogram(text_data):
    seq_lens = [len(sentence.split()) for sentence in text_data]
    plt.hist(seq_lens)
    plt.xlabel('Word Count')
    plt.ylabel('Number of Sentences')
    plt.title('Histogram of Word Count')
    plt.show()


def load_bert_model(chk_point):
    model = AutoModel.from_pretrained(chk_point)
    tokenizer = AutoTokenizer.from_pretrained(chk_point)
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


class BertMfccFusion(nn.Module):
    def __init__(self, bert, fusion=False, hidden_dim=256, seq_len=75, n_layers=6, dropout_level=0.25):
        super().__init__()
        self.bert = bert
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTM(768, self.hidden_dim, self.n_layers, dropout=dropout_level, batch_first=True,
                            bidirectional=True)  # 768 is fixed from bert models
        self.fc = nn.Linear(2 * self.hidden_dim, 512)
        self.fusion = fusion
        self.gelu = nn.GELU()
        if self.fusion:
            # Fully connected audio layer
            self.fca = [nn.Linear(41, 64), nn.GELU(), nn.Dropout(dropout_level),
                        nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout_level)]
            # Fusion layers
            # 512 is dimension from layer fc which has to be concatenated with last layer from fca.
            self.fusions = [nn.Linear(512 + 128, 512), nn.GELU(), nn.Dropout(dropout_level),
                            nn.Linear(512, 512), nn.GELU(), nn.Dropout(dropout_level)]
            self.final = nn.Linear(512, 2)
        else:
            self.final = nn.Linear(2 * self.hidden_dim, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, mfcc_data, hidden):
        bert_out, _ = self.bert(sent_id, attention_mask=mask, return_dict=False)
        lstm_out, hidden = self.lstm(bert_out, hidden)
        x_forward = lstm_out[range(len(lstm_out)), self.seq_len - 1, :self.hidden_dim]
        x_reverse = lstm_out[:, 0, self.hidden_dim:]
        x = torch.cat((x_forward, x_reverse), 1)
        x = self.fc(x)
        x = self.gelu(x)
        if self.fusion:
            a = mfcc_data
            for layer in self.fca:
                a = layer(a)
            x = torch.cat((x, a), dim=1)  # Fusion Layer
            for layer in self.fusions:
                x = layer(x)
        x = self.final(x)
        x = self.softmax(x)
        return x

    def init_hidden(self, batch_size, target_device):
        weight = next(self.parameters()).data
        # Bi directional LSTM
        hidden = (weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device),
                  weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device))
        return hidden

    def to(self, device):
        super().to(device)
        if self.fusion:
            self.fusions = [layer.to(device) for layer in self.fusions]
            self.fca = [layer.to(device) for layer in self.fca]

    def zero_grad(self, **kwargs):
        super().zero_grad()
        if self.fusion:
            for layer in self.fusions:
                layer.zero_grad()
            for layer in self.fca:
                layer.zero_grad()


def get_data_loader(seq, mask, y, mfcc_data=None, batch_size=16, random_seed=42):
    g = torch.Generator()
    g.manual_seed(random_seed)
    data = TensorDataset(seq, mask, mfcc_data, y)
    sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size, generator=g)
    return data, sampler, data_loader


def run_model(model, data_loader, loss_fcn, optimizer, target_device, is_training, clip_at=None):
    if is_training:
        print('Training Model')
        model.train()
    else:
        print('Evaluating')
        model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    model_predictions, model_labels = [], []
    # iterate over batches
    for step, batch in enumerate(data_loader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(data_loader)))
        # push the batch to gpu
        batch = [r.to(target_device) for r in batch]
        sent_id, mask, mfcc_means, labels = batch
        h = model.init_hidden(len(labels), target_device)
        if is_training:
            model.zero_grad()  # clear previously calculated gradients
            # get model predictions for the current batch
            predictions = model(sent_id, mask, mfcc_means, h)
        else:
            with torch.no_grad():
                predictions = model(sent_id, mask, mfcc_means, h)
        # compute the loss between actual and predicted values
        loss = loss_fcn(predictions, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        if is_training:
            loss.backward()  # backward pass to calculate the gradients
            if clip_at:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_at)
            # update parameters
            optimizer.step()
        # model predictions are stored on GPU. So, push it to CPU
        predictions = predictions.detach().cpu().numpy()
        # append the model predictions
        model_predictions.append(predictions)
        model_labels.append(labels.detach().cpu().numpy())
        del batch
    # compute the training loss of the epoch
    avg_loss = total_loss / len(data_loader)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    model_predictions = np.concatenate(model_predictions, axis=0)
    model_labels = np.concatenate(model_labels, axis=0)
    # returns the loss and predictions
    model_predictions = np.argmax(model_predictions, axis=1)
    return avg_loss, model_predictions, model_labels, model


def k_fold_model_preparation(base_model, device, data, sequences, attention_masks, targets, max_seq_len=75,
                             fusion=False,
                             k_folds=5, epochs=5, balance_classes=False, dropout_level=0.25, lr=1e-5,
                             hidden_dim=256, n_layers=6, clip_at=1.0):
    torch.manual_seed(42)
    if fusion:
        print('Running Fusion Model')
    else:
        print('Running Text Only Classification')
    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {x: {} for x in range(k_folds)}
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(data[['Lyric', 'mfcc_mean']], data['iGenre'])):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_mfcc = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in train_ids])
        test_mfcc = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in test_ids])
        train_data, train_sampler, train_data_loader = get_data_loader(sequences[train_ids],
                                                                       attention_masks[train_ids],
                                                                       targets[train_ids],
                                                                       train_mfcc)
        test_data, test_sampler, test_data_loader = get_data_loader(sequences[test_ids],
                                                                    attention_masks[test_ids],
                                                                    targets[test_ids],
                                                                    test_mfcc)
        best_valid_loss = float('inf')
        model = BertMfccFusion(base_model, fusion=fusion, hidden_dim=hidden_dim, seq_len=max_seq_len, n_layers=n_layers,
                               dropout_level=dropout_level)
        model.to(device)
        if balance_classes:
            class_wts = compute_class_weight('balanced',
                                             np.unique(targets[train_ids].tolist()),
                                             targets[train_ids].tolist())
            print(f'Class Weights : {class_wts}')
            # convert class weights to tensor
            weights = torch.tensor(class_wts, dtype=torch.float)
            weights = weights.to(device)
            # loss function
            loss_fcn = nn.NLLLoss(weight=weights)
        else:
            loss_fcn = nn.NLLLoss()
        # empty lists to store training and validation loss of each epoch
        train_losses, valid_losses = [], []
        # define the optimizer
        best_train_predictions, best_test_predictions, best_train_labels, best_test_labels = [], [], [], []
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_schedulers = [ReduceLROnPlateau(optimizer, 'min'), ExponentialLR(optimizer, 0.9)]
        # for each epoch
        results[fold]['train_precision'] = []
        results[fold]['train_recall'] = []
        results[fold]['train_f1'] = []
        results[fold]['validation_precision'] = []
        results[fold]['validation_recall'] = []
        results[fold]['validation_f1'] = []
        for epoch in range(epochs):
            print('Epoch {:} / {:}'.format(epoch + 1, epochs))
            # train model
            train_loss, train_predictions, train_labels, model = run_model(model, train_data_loader, loss_fcn,
                                                                           optimizer,
                                                                           device, True, clip_at)
            # evaluate model
            valid_loss, test_predictions, test_labels, model = run_model(model, test_data_loader, loss_fcn, optimizer,
                                                                         device, False, clip_at)
            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_train_predictions = train_predictions
                best_test_predictions = test_predictions
                best_train_labels = train_labels
                best_test_labels = test_labels
                torch.save(model.state_dict(), f'saved_weights_Fold{fold}.pt')
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            print(f'Losses - Train : {train_loss:.3f} / Validation : {valid_loss:.3f}')
            results[fold]['train_precision'].append(precision_score(train_labels, train_predictions))
            results[fold]['train_recall'].append(recall_score(train_labels, train_predictions))
            results[fold]['train_f1'].append(f1_score(train_labels, train_predictions))
            results[fold]['validation_precision'].append(precision_score(test_labels, test_predictions))
            results[fold]['validation_recall'].append(recall_score(test_labels, test_predictions))
            results[fold]['validation_f1'].append(f1_score(test_labels, test_predictions))
            torch.cuda.empty_cache()

            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(valid_loss)
        print('On Train Data')
        print(classification_report(best_train_labels, best_train_predictions))
        print('On Test Data')
        print(classification_report(best_test_labels, best_test_predictions))
        results[fold]['train_losses'] = train_losses
        results[fold]['validation_losses'] = valid_losses
    return results


def plot_results(results, model_name):
    fig = plt.figure(figsize=[20, 10])
    epochs = len(results[0]['train_precision'])
    x_label = f'{len(results)} Fold and {epochs} Epochs'
    legend_labels = ['Train', 'Validation']

    def subplot_routine(key1, key2, title, loss=False):
        plt.plot([x for k in results for x in results[k][key1]])
        plt.plot([x for k in results for x in results[k][key2]])
        plt.grid()
        plt.xlabel(x_label)
        plt.title(title)
        plt.legend(legend_labels)
        if not loss:
            plt.ylim([0, 1.1])
        else:
            b, t = plt.ylim()
            plt.ylim(np.floor(b), np.ceil(t))

    gs = GridSpec(2, 3, figure=fig)
    plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    subplot_routine('train_losses', 'validation_losses', 'Losses', True)
    plt.subplot(2, 3, 4)
    subplot_routine('train_precision', 'validation_precision', 'Precision')
    plt.subplot(2, 3, 5)
    subplot_routine('train_recall', 'validation_recall', 'Recall')
    plt.subplot(2, 3, 6)
    subplot_routine('train_f1', 'validation_f1', 'F1')
    plt.suptitle(f'Metrics for {model_name}')
    plt.tight_layout()
    plt.show()


def process_data_w_base_model(data, tokenizer, max_seq_len=25):
    txt = list(data.apply(lambda x: x.Lyric, axis=1))
    all_tokens = tokenizer.batch_encode_plus(txt,
                                             max_length=max_seq_len,
                                             padding='longest',
                                             truncation=True,
                                             return_token_type_ids=False)
    sequences = torch.tensor(all_tokens['input_ids'])
    attention_masks = torch.tensor(all_tokens['attention_mask'])
    targets = torch.tensor(data['iGenre'].tolist())
    return sequences, attention_masks, targets
