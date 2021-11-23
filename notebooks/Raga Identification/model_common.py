import datetime
from abc import ABC

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.data import IterableDataset, DataLoader


def get_devices():
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
    else:
        gpu = torch.device("cpu")
    cpu = torch.device("cpu")
    return gpu, cpu


def get_loss_function(balance_classes, labels, run_on, loss_fcn=nn.CrossEntropyLoss):
    if balance_classes:
        class_wts = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(labels),
                                         y=labels
                                         )
        print(f'Class Weights : {class_wts}')
        # convert class weights to tensor
        weights = torch.tensor(class_wts, dtype=torch.float)
        weights = weights.to(run_on)
        # loss function
        loss_fcn = loss_fcn(weight=weights)
    else:
        loss_fcn = loss_fcn()
    return loss_fcn


def update_best_result(best_scores, valid_loss, train_labels, train_predictions, test_labels, test_predictions,
                       model=None, model_file_name=f'saved_weights_Fold_0.pt'):
    if valid_loss < best_scores['valid_loss']:
        best_scores['valid_loss'] = valid_loss
        best_scores['train_predictions'] = train_predictions
        best_scores['test_predictions'] = test_predictions
        best_scores['train_labels'] = train_labels
        best_scores['test_labels'] = test_labels
        if model:
            torch.save(model.state_dict(), model_file_name)
    return best_scores


def update_results_dict(results, train_labels, train_predictions, test_labels, test_predictions,
                        average='binary', pos_label=1):
    results['train_precision'].append(precision_score(train_labels, train_predictions, average=average,
                                                      pos_label=pos_label))
    results['train_recall'].append(recall_score(train_labels, train_predictions, average=average, pos_label=pos_label))
    results['train_f1'].append(f1_score(train_labels, train_predictions, average=average, pos_label=pos_label))
    results['validation_precision'].append(precision_score(test_labels, test_predictions, average=average,
                                                           pos_label=pos_label))
    results['validation_recall'].append(recall_score(test_labels, test_predictions, average=average,
                                                     pos_label=pos_label))
    results['validation_f1'].append(f1_score(test_labels, test_predictions, average=average, pos_label=pos_label))
    results['train_labels'].append([train_labels])
    results['test_labels'].append([test_labels])
    results['train_predictions'].append([train_predictions])
    results['test_predictions'].append([test_predictions])
    return results


def run_model(model, dataset, loss_fcn, optimizer, is_training, run_on, clip_at=None, lstm_model=False, report_at=20):
    predict = lambda mdl, data, hidden: mdl(data, hidden) if hidden else mdl(data)
    hid = lambda mdl, data, _run_on, is_lstm: mdl.init_hidden(data.shape[0], _run_on) if is_lstm else None
    if is_training:
        model.train()
    else:
        model.eval()
    total_loss, total_accuracy, model_predictions, model_labels, aud_data = 0, 0, [], [], None
    for step, batch in enumerate(dataset):
        if step % report_at == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataset)))
        # push the batch to gpu
        batch = [r.to(run_on) for r in batch]
        aud_data, labels = batch
        h = hid(model, aud_data, run_on, lstm_model)
        if is_training:
            for o in optimizer:
                o.zero_grad()
            model.zero_grad()
            predictions = predict(model, aud_data, h)
        else:
            with torch.no_grad():
                predictions = predict(model, aud_data, h)
        # compute the loss between actual and predicted values
        loss = loss_fcn(predictions, labels)
        predictions = predictions.detach().cpu().numpy()
        total_loss = total_loss + loss.item()
        if is_training:
            loss.backward()  # backward pass to calculate the gradients
            if clip_at:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_at)
            # update parameters
            for o in optimizer:
                o.step()
        # append the model predictions
        model_predictions.append(predictions)
        model_labels.append(labels.detach().cpu().numpy())
        del batch
    # compute the training loss of the epoch
    avg_loss = total_loss / len(dataset)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    model_predictions = np.concatenate(model_predictions, axis=0)
    model_labels = np.concatenate(model_labels, axis=0)
    # returns the loss and predictions
    model_predictions = np.argmax(model_predictions, axis=1)
    return avg_loss, model_predictions, model_labels, model


def plot_results(results, model_name):
    fig = plt.figure(figsize=[20, 10])
    epochs = len(results[0]['train_precision'])
    x_label = f'{len(results)} Fold and {epochs} Epochs'
    legend_labels = ['Train', 'Val']

    def subplot_routine(key1, key2, title, loss=False):
        plt.plot([x for k in results for x in results[k][key1]])
        plt.plot([x for k in results for x in results[k][key2]])
        plt.grid()
        plt.xlabel(x_label)
        plt.legend([f'{x} {title}' for x in legend_labels])
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


class AbsDataset(IterableDataset, ABC):
    def __init__(self):
        super().__init__()

    def get_data_loader(self, batch_size=32, random_seed=42):
        g = torch.Generator()
        g.manual_seed(random_seed)
        return DataLoader(self, batch_size=batch_size, generator=g)


def train_model(data, prepare_data_hnd, gpu, **kwargs):
    torch.manual_seed(42)
    start_time = datetime.datetime.now()

    n_labels = kwargs['n_labels']
    title = kwargs['title']
    report = kwargs['report'] if 'report' in kwargs else 20
    lstm = kwargs['lstm'] if 'lstm' in kwargs else False
    lr = kwargs['lr'] if 'lr' in kwargs else 1e-4
    clip_at = kwargs['clip_at'] if 'clip_at' in kwargs else None
    k_folds = kwargs['k_folds'] if 'k_folds' in kwargs else 5
    balance_classes = kwargs['balance_classes'] if 'balance_classes' in kwargs else True
    dropout = kwargs['dropout'] if 'dropout' in kwargs else 0.25
    plot = kwargs['plot'] if 'plot' in kwargs else True
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 5
    mdlargs = kwargs['mdlargs'] if 'mdlargs' in kwargs else {}
    feature = kwargs['feature'] if 'feature' in kwargs else ['mfcc_mean']
    target = kwargs['target'] if 'target' in kwargs else 'RagamCode'
    loss_fcn = kwargs['loss_fcn'] if 'loss_fcn' in kwargs else None

    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {x: {} for x in range(k_folds)}
    labels = data[target]
    ovl_best_scores = {'valid_loss': float('inf'),
                       'train_predictions': [],
                       'test_predictions': [],
                       'train_labels': [],
                       'test_labels': []
                       }
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(data[feature], data[target])):
        print(f'FOLD {fold + 1} \n Data Sizes (Train/Test) : {len(train_ids)}/{len(test_ids)}')
        fold_start = datetime.datetime.now()
        # empty lists to store training and validation loss of each epoch
        train_losses, valid_losses = [], []
        best_scores = {'valid_loss': float('inf'),
                       'train_predictions': [],
                       'test_predictions': [],
                       'train_labels': [],
                       'test_labels': []
                       }
        # for each epoch
        results[fold] = {
            'train_f1': [],
            'validation_f1': [],
            'train_precision': [],
            'validation_precision': [],
            'train_recall': [],
            'validation_recall': [],
            'train_labels': [],
            'test_labels': [],
            'train_predictions': [],
            'test_predictions': []
        }
        train_data, test_data, train_labels, test_labels = prepare_data_hnd(data, train_ids, test_ids)
        model = kwargs['model'](n_labels, dropout=dropout, **mdlargs)
        model.to(gpu)
        if not loss_fcn:
            loss_fcn = get_loss_function(balance_classes, data[target].tolist(), gpu, nn.CrossEntropyLoss)
        # define the optimizer
        optimizer = [torch.optim.Adam(model.parameters(), lr=lr)]
        lr_schedulers = [ReduceLROnPlateau(optimizer[0], patience=3, factor=0.1, threshold=1e-9, mode='min'),
                         ExponentialLR(optimizer[0], gamma=0.9)]
        for epoch in range(epochs):
            e_start = datetime.datetime.now()
            # train model
            train_loss, train_predictions, train_labels, model = run_model(model,
                                                                           train_data.get_data_loader(batch_size=32),
                                                                           loss_fcn, optimizer, run_on=gpu,
                                                                           is_training=True, clip_at=clip_at,
                                                                           lstm_model=lstm, report_at=report)
            # evaluate model
            valid_loss, test_predictions, test_labels, model = run_model(model,
                                                                         test_data.get_data_loader(batch_size=32),
                                                                         loss_fcn, optimizer, run_on=gpu,
                                                                         is_training=False, clip_at=clip_at,
                                                                         lstm_model=lstm, report_at=report)
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(valid_loss)
            torch.cuda.empty_cache()
            # save the best model
            best_scores = update_best_result(best_scores,
                                             valid_loss,
                                             train_labels, train_predictions,
                                             test_labels, test_predictions)
            ovl_best_scores = update_best_result(ovl_best_scores,
                                                 valid_loss,
                                                 train_labels, train_predictions,
                                                 test_labels, test_predictions,
                                                 model=model,
                                                 model_file_name=f'saved_weights_Fold_{fold}.pt')
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            results[fold] = update_results_dict(results[fold],
                                                train_labels, train_predictions,
                                                test_labels, test_predictions)
            e_end = datetime.datetime.now()
            print(
                f'Epoch {epoch + 1}/{epochs} : Training Loss: {train_loss:.3f} / Validation Loss : {valid_loss:.3f} [Time Taken : {(e_end - e_start).total_seconds()} seconds]')
        print('*** Confusion Matrix - Training ***')
        print(confusion_matrix(best_scores['train_labels'], best_scores['train_predictions']))
        print('*** Confusion Matrix - Validation ***')
        print(confusion_matrix(best_scores['test_labels'], best_scores['test_predictions']))
        results[fold]['train_losses'] = train_losses
        results[fold]['validation_losses'] = valid_losses
        print(f'Fold {fold + 1} : {(datetime.datetime.now() - fold_start).total_seconds()} seconds')
    end_time = datetime.datetime.now()
    print(f'Overall Time : {(end_time - start_time).total_seconds()} seconds')
    print('*** Confusion Matrix - Training ***')
    print(confusion_matrix(ovl_best_scores['train_labels'], ovl_best_scores['train_predictions']))
    print('*** Confusion Matrix - Validation ***')
    print(confusion_matrix(ovl_best_scores['test_labels'], ovl_best_scores['test_predictions']))
    if plot:
        plot_results(results, title)
    return results


def results_to_df(results):
    p = pd.DataFrame(results[0])
    for i in range(1, len(results)):
        p = pd.concat([p, pd.DataFrame(results[i])], axis=0)
    p.sort_values(by=['validation_f1','train_f1'], ascending=False, inplace=True)
    return p
