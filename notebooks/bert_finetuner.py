import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from transformers import AutoModel, AutoTokenizer

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from datasets import BertDataset, BertMelDataset, BertMfccDataset
from common_helpers import *


def load_bert_model(chk_point):
    model = AutoModel.from_pretrained(chk_point)
    tokenizer = AutoTokenizer.from_pretrained(chk_point)
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


class BertFineTuningModel(nn.Module):
    def __init__(self, bert, hidden_dim=256, seq_len=75, n_layers=6, dropout_level=0.25):
        super().__init__()
        self.BERT_VEC_SIZE = 768  # 768 is fixed from bert models
        self.bert = bert
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTM(self.BERT_VEC_SIZE,
                            self.hidden_dim,
                            self.n_layers,
                            dropout=dropout_level,
                            batch_first=True,
                            bidirectional=True)
        self.fw = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 512),
            nn.GELU()
        )
        self.final = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward_bert(self, sent_id, mask, hidden):
        bert_out, _ = self.bert(sent_id, attention_mask=mask, return_dict=False)
        lstm_out, hidden = self.lstm(bert_out, hidden)
        x_forward = lstm_out[range(len(lstm_out)), self.seq_len - 1, :self.hidden_dim]
        x_reverse = lstm_out[:, 0, self.hidden_dim:]
        x = torch.cat((x_forward, x_reverse), 1)
        return x

    def forward(self, sent_id, mask, hidden):
        x = self.forward_bert(sent_id, mask, hidden)
        x = self.fw(x)
        x = self.final(x)
        x = self.softmax(x)
        return x

    def init_hidden(self, batch_size, target_device):
        weight = next(self.parameters()).data
        # Bi directional LSTM
        hidden = (weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device),
                  weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device))
        return hidden


class BertMfccFusionModel(BertFineTuningModel):
    def __init__(self, bert, hidden_dim=256, seq_len=75, n_layers=6, dropout_level=0.25, fusion=False, MfccLen=41):
        super().__init__(bert, hidden_dim, seq_len, n_layers, dropout_level)
        self.fusion = fusion
        if self.fusion:
            self.MfccLen = MfccLen
            # Fully connected audio layer
            self.fca = nn.Sequential(
                nn.Linear(MfccLen, 64),
                nn.GELU(),
                nn.Dropout(dropout_level),
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Dropout(dropout_level)
            )
            # Fusion layers
            s = seq_layer_size(self.fca, [1, MfccLen])
            s += seq_layer_size(self.fw, [1, 2 * self.hidden_dim])
            self.fusions = nn.Sequential(nn.Linear(s, 512),
                                         nn.GELU(),
                                         nn.Dropout(dropout_level),
                                         nn.Linear(512, 512),
                                         nn.GELU(),
                                         nn.Dropout(dropout_level))
            self.final = nn.Linear(512, 2)
        else:
            self.final = nn.Linear(2 * self.hidden_dim, 2)

    def forward(self, sent_id, mask, mfcc_data, hidden):
        x = self.forward_bert(sent_id, mask, hidden)
        x = self.fw(x)
        if self.fusion:
            x = fusion_layers(self.fca, mfcc_data, x, self.fusions)
        x = self.final(x)
        x = self.softmax(x)
        return x


class BertMelFusionModel(BertFineTuningModel):
    def __init__(self, bert, fusion=False, hidden_dim=256, seq_len=75, n_layers=6, dropout_level=0.25,
                 img_height=80, img_width=80):
        super().__init__(bert, hidden_dim, seq_len, n_layers, dropout_level)
        self.img_height = img_height
        self.img_width = img_width
        self.fusion = fusion
        if self.fusion:
            # Fully connected audio layer with MEL Spectrogram
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
            s += seq_layer_size(self.fw, [1, 2 * self.hidden_dim])
            self.fusions = nn.Sequential(
                nn.Linear(s, 512),
                nn.GELU(),
                nn.Dropout(dropout_level),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(dropout_level)
            )
            self.final = nn.Linear(512, 2)
        else:
            self.final = nn.Linear(2 * self.hidden_dim, 2)

    def forward(self, sent_id, mask, mel_data, hidden):
        x = self.forward_bert(sent_id, mask, hidden)
        x = self.fw(x)
        if self.fusion:
            x = fusion_layers(self.mel_layers, mel_data.permute(0, 3, 1, 2).float(), x, self.fusions)
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
            sent_id, mask, aud_data, labels = batch
        else:
            sent_id, mask, labels = batch
        h = model.init_hidden(len(labels), run_on)
        if is_training:
            model.zero_grad()  # clear previously calculated gradients
            # get model predictions for the current batch
            if is_fusion:
                predictions = model(sent_id, mask, aud_data, h)
            else:
                predictions = model(sent_id, mask, h)
        else:
            with torch.no_grad():
                if is_fusion:
                    predictions = model(sent_id, mask, aud_data, h)
                else:
                    predictions = model(sent_id, mask, h)
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


def prepare_train_test_data(data, sequences, attention_masks, targets, fusion, train_ids, test_ids, IMG_PATH=None):
    if fusion == FusionTypes.MFCC:
        train_aud = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in train_ids])
        test_aud = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in test_ids])
        train_data = BertMfccDataset(sequences[train_ids], attention_masks[train_ids], targets[train_ids], train_aud)
        test_data = BertMfccDataset(sequences[test_ids], attention_masks[test_ids], targets[test_ids], test_aud)
    elif fusion == FusionTypes.MEL:
        train_aud = [data['MP3 File'].iloc[x][0:-4] + ".png" for x in train_ids]
        test_aud = [data['MP3 File'].iloc[x][0:-4] + ".png" for x in test_ids]
        train_data = BertMelDataset(sequences[train_ids], attention_masks[train_ids], targets[train_ids],
                                    train_aud, IMG_PATH)
        test_data = BertMelDataset(sequences[test_ids], attention_masks[test_ids], targets[test_ids],
                                   test_aud, IMG_PATH)
    else:
        train_data = BertDataset(sequences[train_ids], attention_masks[train_ids], targets[train_ids])
        test_data = BertDataset(sequences[test_ids], attention_masks[test_ids], targets[test_ids])
    return train_data, test_data


def get_model_to_train(fusion, base_model, hidden_dim, seq_len, n_layers, dropout_level, run_on, mfcc_len=41,
                       img_height=80, img_width=80):
    if fusion == FusionTypes.MFCC:
        model = BertMfccFusionModel(base_model, fusion=True, hidden_dim=hidden_dim, seq_len=seq_len,
                                    n_layers=n_layers, dropout_level=dropout_level, MfccLen=mfcc_len)
        is_fusion = True
    elif fusion == FusionTypes.MEL:
        model = BertMelFusionModel(base_model, fusion=True, hidden_dim=hidden_dim, seq_len=seq_len,
                                   n_layers=n_layers, dropout_level=dropout_level,
                                   img_height=img_height, img_width=img_width)
        is_fusion = True
    else:
        model = BertFineTuningModel(base_model, hidden_dim=hidden_dim, seq_len=seq_len,
                                    n_layers=n_layers, dropout_level=dropout_level)
        is_fusion = False
    model.to(run_on)
    return model, is_fusion


def run_k_fold(base_model, device, data, sequences, attention_masks,
               labels, max_seq_len=75, fusion=None, k_folds=5,
               epochs=5, balance_classes=False,
               dropout_level=0.25, lr=1e-5,
               hidden_dim=256, n_layers=6, clip_at=1.0,
               mfcc_len=41, img_height=80, img_width=80, img_path=None):
    start_time = datetime.datetime.now()
    torch.manual_seed(42)
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
        train_data, test_data = prepare_train_test_data(data, sequences, attention_masks, labels, fusion,
                                                        train_ids, test_ids, img_path)
        model, is_fusion = get_model_to_train(fusion, base_model, hidden_dim=hidden_dim, seq_len=max_seq_len,
                                              n_layers=n_layers, mfcc_len=mfcc_len, img_height=img_height,
                                              img_width=img_width, run_on=device, dropout_level=dropout_level)
        loss_fcn = get_loss_function(balance_classes, labels[train_ids], device)
        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_schedulers = [ReduceLROnPlateau(optimizer, 'min'),
                         ExponentialLR(optimizer, 0.9)]
        for epoch in range(epochs):
            e_start = datetime.datetime.now()
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
            print('Epoch {:} / {:}, Train Losses : {:}/ Valid Losses : {:} [Time : {:} seconds]'.format(
                  epoch + 1, epochs, train_loss, valid_loss, (e_end - e_start).total_seconds()))
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
