import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW
import numpy as np


class TextMfccFusion(nn.Module):
    def __init__(self, bert, fusion=False, dropout_level=0.25):
        super(TextMfccFusion, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_level)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fusion = fusion
        if self.fusion:
            self.fc2 = nn.Linear(512, 512)
            self.fca1 = nn.Linear(14, 64)
            self.fca2 = nn.Linear(64, 128)
            self.fusion1 = nn.Linear(640, 512)  # 512 + 128 : fc2 + fca2
            self.fusion2 = nn.Linear(512, 2)
        else:
            self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, mfcc_data):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.fusion:
            a = self.fca1(mfcc_data)
            a = self.relu(a)
            a = self.dropout(a)
            a = self.fca2(a)
            x = self.relu(x)  # Activation for output from text features
            x = torch.cat((x, a), dim=1)  # Fusion Layer
            x = self.fusion1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fusion2(x)
        x = self.softmax(x)
        return x


def get_data_loader(seq, mask, y, mfcc_data=None, batch_size=16, random_seed=42):
    g = torch.Generator()
    g.manual_seed(random_seed)
    data = TensorDataset(seq, mask, mfcc_data, y)
    sampler = RandomSampler(data)
    data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size, generator=g)
    return data, sampler, data_loader


def run_model(model, data_loader, loss_fcn, optimizer, target_device, is_training):
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
        if step % 20 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(data_loader)))
        # push the batch to gpu
        batch = [r.to(target_device) for r in batch]
        sent_id, mask, mfcc_means, labels = batch
        if is_training:
            model.zero_grad()  # clear previously calculated gradients
        # get model predictions for the current batch
        predictions = model(sent_id, mask, mfcc_means)
        # compute the loss between actual and predicted values
        loss = loss_fcn(predictions, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        if is_training:
            loss.backward()  # backward pass to calculate the gradients
            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    return avg_loss, model_predictions, model_labels


def k_fold_model_preparation(base_model, device, fusion, data, sequences, attention_masks, targets,
                             k_folds=5, epochs=5, balance_classes=False):
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
        model = TextMfccFusion(base_model, fusion)
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
        optimizer = AdamW(model.parameters(), lr=1e-5)
        # for each epoch
        results[fold]['train_precision'] = []
        results[fold]['train_recall'] = []
        results[fold]['train_f1'] = []
        results[fold]['validation_precision'] = []
        results[fold]['validation_recall'] = []
        results[fold]['validation_f1'] = []
        for epoch in range(epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
            # train model
            train_loss, train_predictions, train_labels = run_model(model, train_data_loader, loss_fcn, optimizer,
                                                                    device, True)
            # evaluate model
            valid_loss, test_predictions, test_labels = run_model(model, test_data_loader, loss_fcn, optimizer,
                                                                  device, False)
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
            print(f'Training Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
            results[fold]['train_precision'].append(precision_score(train_labels, train_predictions))
            results[fold]['train_recall'].append(recall_score(train_labels, train_predictions))
            results[fold]['train_f1'].append(f1_score(train_labels, train_predictions))
            results[fold]['validation_precision'].append(precision_score(test_labels, test_predictions))
            results[fold]['validation_recall'].append(recall_score(test_labels, test_predictions))
            results[fold]['validation_f1'].append(f1_score(test_labels, test_predictions))
            torch.cuda.empty_cache()
        print('On Train Data')
        print(classification_report(best_train_labels, best_train_predictions))
        print('On Test Data')
        print(classification_report(best_test_labels, best_test_predictions))
        results[fold]['train_losses'] = train_losses
        results[fold]['validation_losses'] = valid_losses
    return results


def plot_results(results, model_name):
    plt.figure(figsize=[20, 5])
    epochs = len(results[0]['train_precision'])
    x_label = f'{len(results)} Fold and {epochs} Epochs'
    legend_labels = ['Train', 'Validation']

    def subplot_routine(key1, key2, title):
        plt.plot([x for k in results for x in results[k][key1]])
        plt.plot([x for k in results for x in results[k][key2]])
        plt.xlabel(x_label)
        plt.title(title)
        plt.legend(legend_labels)
        plt.ylim([0, 1.1])

    plt.subplot(1, 3, 1)
    subplot_routine('train_precision', 'validation_precision', 'Precision')
    plt.subplot(1, 3, 2)
    subplot_routine('train_recall', 'validation_recall', 'Recall')
    plt.subplot(1, 3, 3)
    subplot_routine('train_f1', 'validation_f1', 'F1')
    plt.suptitle(f'Metrics for {model_name}')
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
