import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from common_helpers import get_loss_function, seq_layer_size, \
    FusionTypes, update_best_result, update_results_dict, results_to_df
from datasets import FtDataSet, FtMelDataset, FtMfccDataSet


class FtSentVectorsModel(nn.Module):
    def __init__(self, dropout_level=0.25):
        super().__init__()
        self.FT_VEC_SIZE = 300
        self.fc = nn.Sequential(
            nn.Linear(300, 512), nn.ReLU(), nn.Dropout(dropout_level),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout_level),
            nn.Linear(512, 1024), nn.ReLU(), nn.Dropout(dropout_level)
        )
        self.final = nn.Linear(1024, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_vector):
        x = self.fc(sent_vector)
        x = self.final(x)
        x = self.softmax(x)
        return x


class FtMfccFusionModel(FtSentVectorsModel):
    def __init__(self, dropout_level=0.25, fusion=False, mfcc_len=41):
        super().__init__(dropout_level=dropout_level)
        self.mfcc_len = mfcc_len
        self.fusion = fusion
        if self.fusion:
            self.fca = nn.Sequential(nn.Linear(self.mfcc_len, 64),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_level),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_level),
                                     nn.Linear(128, 256),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_level),
                                     nn.Linear(256, 512),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_level),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_level))
            self.fusion_layers = nn.Sequential(nn.Linear(2048, 512),
                                               nn.ReLU(),
                                               nn.Dropout(dropout_level),
                                               nn.Linear(512, 768),
                                               nn.ReLU(),
                                               nn.Dropout(dropout_level),
                                               nn.Linear(768, 1024),
                                               nn.ReLU(),
                                               nn.Dropout(dropout_level))

    def forward(self, sent_vector, mfcc_data):
        x = self.fc(sent_vector)
        if self.fusion:
            a = self.fca(mfcc_data)
            x = torch.cat((x, a), dim=1)
            x = self.fusion_layers(x)
        x = self.final(x)
        x = self.softmax(x)
        return x


class FtMelFusionModel(FtSentVectorsModel):
    def __init__(self, dropout_level=0.25, fusion=False, img_height=80, img_width=80):
        super().__init__(dropout_level=dropout_level)
        self.fusion = fusion
        self.img_width = img_width
        self.img_height = img_height
        if self.fusion:
            self.mel_layers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2), padding='same'),
                nn.MaxPool2d(kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), padding='same'),
                nn.MaxPool2d(kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding='same'),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2), padding='same'),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Flatten()
            )
            # Fusion layers
            s = seq_layer_size(self.mel_layers, [1, 3, self.img_height, self.img_width])
            s += 1024  # Size of last layer in fc.
            self.fusions = nn.Sequential(
                nn.Linear(s, 512),
                nn.GELU(),
                nn.Dropout(dropout_level),
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Dropout(dropout_level)
            )

    def forward(self, sent_vector, mel_data):
        x = self.fc(sent_vector)
        if self.fusion:
            a = self.mel_layers(mel_data.permute(0, 3, 1, 2).float())
            x = torch.cat((x, a), dim=1)
            x = self.fusions(x)
        x = self.final(x)
        x = self.softmax(x)
        return x


def run_model(model, dataset, loss_fcn, optimizer, is_training, run_on, clip_at=None, is_fusion=False):
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
    aud_data = None
    for step, batch in enumerate(dataset):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataset)))
        # push the batch to gpu
        batch = [r.to(run_on) for r in batch]
        if is_fusion:
            sent_vector, aud_data, labels = batch
        else:
            sent_vector, labels = batch
        if is_training:
            model.zero_grad()  # clear previously calculated gradients
            # get model predictions for the current batch
            if is_fusion:
                predictions = model(sent_vector, aud_data)
            else:
                predictions = model(sent_vector)
        else:
            with torch.no_grad():
                if is_fusion:
                    predictions = model(sent_vector, aud_data)
                else:
                    predictions = model(sent_vector)
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
    avg_loss = total_loss / len(dataset)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    model_predictions = np.concatenate(model_predictions, axis=0)
    model_labels = np.concatenate(model_labels, axis=0)
    # returns the loss and predictions
    model_predictions = np.argmax(model_predictions, axis=1)
    return avg_loss, model_predictions, model_labels, model


def prepare_train_test_data(data, ft_feature, targets, fusion, train_ids, test_ids, IMG_PATH=None, img_height=80,
                            img_width=80):
    train_ft_vectors = torch.tensor([[_ for _ in data[ft_feature].iloc[x]] for x in train_ids])
    test_ft_vectors = torch.tensor([[_ for _ in data[ft_feature].iloc[x]] for x in test_ids])
    if fusion == FusionTypes.MFCC:
        train_aud = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in train_ids])
        test_aud = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in test_ids])
        train_data = FtMfccDataSet(train_ft_vectors, targets[train_ids], train_aud)
        test_data = FtMfccDataSet(test_ft_vectors, targets[test_ids], test_aud)
    elif fusion == FusionTypes.MEL:
        train_aud = [data['MP3 File'].iloc[x][0:-4] + ".png" for x in train_ids]
        test_aud = [data['MP3 File'].iloc[x][0:-4] + ".png" for x in test_ids]
        train_data = FtMelDataset(train_ft_vectors, targets[train_ids], train_aud, image_path=IMG_PATH,
                                  image_height=img_height, image_width=img_width)
        test_data = FtMelDataset(test_ft_vectors, targets[test_ids], test_aud, image_path=IMG_PATH,
                                 image_height=img_height, image_width=img_width)
    else:
        train_data = FtDataSet(train_ft_vectors, targets[train_ids])
        test_data = FtDataSet(test_ft_vectors, targets[test_ids])
    return train_data, test_data


def get_model_to_train(fusion, dropout_level, run_on, mfcc_len=41,
                       img_height=80, img_width=80):
    if fusion == FusionTypes.MFCC:
        model = FtMfccFusionModel(fusion=True, dropout_level=dropout_level, mfcc_len=mfcc_len)
        is_fusion = True
    elif fusion == FusionTypes.MEL:
        model = FtMelFusionModel(fusion=True, dropout_level=dropout_level,
                                 img_height=img_height, img_width=img_width)
        is_fusion = True
    else:
        model = FtSentVectorsModel(dropout_level=dropout_level)
        is_fusion = False
    model.to(run_on)
    return model, is_fusion


def run_k_fold(device, data, ft_feature, fusion=None, k_folds=5,
               epochs=5, balance_classes=False,
               dropout_level=0.25, lr=1e-5,
               clip_at=1.0, mfcc_len=41, img_height=80, img_width=80, img_path=None):
    start_time = datetime.datetime.now()
    torch.manual_seed(42)
    labels = torch.tensor(data['iGenre'].tolist())
    if fusion == FusionTypes.TXT:
        print('Running Text Only Classification')
    else:
        print('Running Fusion Classification')
    k_fold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {x: {} for x in range(k_folds)}
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(data[['Lyric']], data['iGenre'])):
        print(f'FOLD {fold}')
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
            'validation_recall': []
        }
        train_data, test_data = prepare_train_test_data(data, ft_feature, labels, fusion,
                                                        train_ids, test_ids, img_path, img_height=img_height,
                                                        img_width=img_width)
        model, is_fusion = get_model_to_train(fusion, mfcc_len=mfcc_len, img_height=img_height,
                                              img_width=img_width, run_on=device, dropout_level=dropout_level)
        loss_fcn = get_loss_function(balance_classes, labels[train_ids], device)
        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_schedulers = [ReduceLROnPlateau(optimizer, 'min'),
                         ExponentialLR(optimizer, 0.9)]
        for epoch in range(epochs):
            e_start = datetime.datetime.now()
            print('Epoch {:} / {:}'.format(epoch + 1, epochs))
            # train model
            train_loss, train_predictions, train_labels, model = run_model(model,
                                                                           train_data.get_data_loader(),
                                                                           loss_fcn,
                                                                           optimizer,
                                                                           run_on=device,
                                                                           is_training=True,
                                                                           clip_at=clip_at,
                                                                           is_fusion=is_fusion)
            # evaluate model
            valid_loss, test_predictions, test_labels, model = run_model(model,
                                                                         test_data.get_data_loader(),
                                                                         loss_fcn,
                                                                         optimizer,
                                                                         run_on=device,
                                                                         is_training=False,
                                                                         clip_at=clip_at,
                                                                         is_fusion=is_fusion)
            print(f'Losses - Train : {train_loss:.3f} / Validation : {valid_loss:.3f}')
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(valid_loss)
            torch.cuda.empty_cache()
            # save the best model
            best_scores = update_best_result(best_scores,
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
            print(f'Time for epoch : {(e_end - e_start).total_seconds()} seconds')
        print('On Train Data')
        print(classification_report(best_scores['train_labels'], best_scores['train_predictions']))
        print('On Test Data')
        print(classification_report(best_scores['test_labels'], best_scores['test_predictions']))
        results[fold]['train_losses'] = train_losses
        results[fold]['validation_losses'] = valid_losses
        print(f'Time for fold {fold} : {(datetime.datetime.now() - fold_start).total_seconds()} seconds')
    end_time = datetime.datetime.now()
    print(f'Overall Time : {(end_time - start_time).total_seconds()} seconds')
    return results, results_to_df(results)
