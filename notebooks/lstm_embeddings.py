from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from common_helpers import *
import datetime
from datasets import LstmDataset, LstmMfccDataset, LstmMelDataset


class LstmModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len, n_layers=6, bidir=False, dropout_level=0.1, max_norm=2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, max_norm=max_norm)
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.n_layers, dropout=dropout_level, batch_first=True,
                            bidirectional=self.bidir)
        if self.bidir:
            self.fc = nn.Sequential(nn.Linear(2 * hidden_dim, 2), nn.LogSoftmax(dim=0))
        else:
            self.fc = nn.Sequential(nn.Linear(hidden_dim, 2), nn.LogSoftmax(dim=0))

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        x_forward = lstm_out[range(len(lstm_out)), self.seq_len - 1, :self.hidden_dim]
        x_reverse = lstm_out[:, 0, self.hidden_dim:]
        x = torch.cat((x_forward, x_reverse), 1)
        x = self.fc(x)
        x = x.view(batch_size, -1)
        return x

    def init_hidden(self, batch_size, target_device):
        weight = next(self.parameters()).data
        if self.bidir:
            hidden = (weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device),
                      weight.new(2 * self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device))
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(target_device))
        return hidden


def forward_fusion(model, x, aud_data, aud_layers, hidden):
    batch_size = x.size(0)
    embeds = model.embedding(x)
    lstm_out, hidden = model.lstm(embeds, hidden)
    x_forward = lstm_out[range(len(lstm_out)), model.seq_len - 1, :model.hidden_dim]
    x_reverse = lstm_out[:, 0, model.hidden_dim:]
    x = torch.cat((x_forward, x_reverse), 1)
    x = model.fc(x)
    if model.fusion:
        x = fusion_layers(aud_layers, aud_data, x, model.fusion_layers)
    x = model.softmax(x)
    x = x.view(batch_size, -1)
    return x


class LstmMfccModel(LstmModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len, fusion=False, n_layers=6, dropout_level=0.1,
                 mfcc_len=41):
        super().__init__(vocab_size, embedding_dim, hidden_dim, seq_len, n_layers, dropout_level)
        self.fusion = fusion
        self.mfcc_len = mfcc_len
        if self.fusion:
            self.fc = nn.Linear(2 * hidden_dim, 512)
            self.fca = nn.Sequential(nn.Linear(self.mfcc_len, 64),
                                     nn.GELU(),
                                     nn.Linear(64, 128),
                                     nn.GELU())
            self.fusion_layers = nn.Sequential(nn.Linear(128 + 512, 512),
                                               nn.GELU(),
                                               nn.Linear(512, 2),
                                               nn.GELU())

    def forward(self, x, mfcc, hidden):
        x = forward_fusion(self, x, mfcc, self.fca, hidden)
        return x


class LstmMelModel(LstmModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len, fusion=False, n_layers=6, dropout_level=0.1,
                 img_height=80, img_width=80):
        super().__init__(vocab_size, embedding_dim, hidden_dim, seq_len, n_layers, dropout_level)
        self.fusion = fusion
        self.img_height = img_height
        self.img_width = img_width
        if self.fusion:
            self.fc = nn.Linear(2 * hidden_dim, 512)
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
            s += 512
            self.fusion_layers = nn.Sequential(nn.Linear(s, 512),
                                               nn.GELU(),
                                               nn.Linear(512, 2),
                                               nn.GELU())

    def forward(self, x, mel_data, hidden):
        x = forward_fusion(self, x, mel_data.permute(0, 3, 1, 2).float(), self.mel_layers, hidden)
        return x


def build_word_tokenizer(sentences, word_threshold=1):
    word_sequences = [[] for _ in sentences]
    # Dictionary to create word to frequency
    word_counter = Counter()
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            word_counter.update([word])
            word_sequences[i].append(word)
    word_counter = {k: v for k, v in word_counter.items() if v > word_threshold}
    word_counter = sorted(word_counter, key=word_counter.get, reverse=True)
    word2idx = defaultdict(int)
    idx2word = defaultdict(str)
    for i, word in enumerate(word_counter):
        word2idx[word] = i
        idx2word[i] = word
    return word_counter, word_sequences, word2idx, idx2word


def pad_input(sent_sequence, seq_len):
    features = np.zeros((len(sent_sequence), seq_len), dtype=int)
    for i, sentence in enumerate(sent_sequence):
        if len(sentence) != 0:
            features[i, -len(sentence):] = np.array(sentence)[:seq_len]
    return features


def tokenize(sentences, word2idx, seq_len=None):
    token_matrix = [[] for _ in sentences]
    for i, sentence in enumerate(sentences):
        token_matrix[i] = [word2idx[word] for word in sentence.split()]
    if seq_len:
        token_matrix = pad_input(token_matrix, seq_len)
    return token_matrix


def run_model(model, data_loader, loss_fcn, optimizer, target_device, is_training, clip_at=1.0, is_fusion=False):
    if is_training:
        model.train()
    else:
        model.eval()
    total_loss, total_accuracy = 0, 0
    # empty list to save model predictions
    model_predictions, model_labels = [], []
    # iterate over batches
    aud_data = None
    for step, batch in enumerate(data_loader):
        # push the batch to gpu
        batch = [r.to(target_device) for r in batch]
        if is_fusion:
            sent_vectors, aud_data, labels = batch
        else:
            sent_vectors, labels = batch
        h = model.init_hidden(len(labels), target_device)
        if is_training:
            model.zero_grad()  # clear previously calculated gradients
            # get model predictions for the current batch
            if is_fusion:
                predictions = model(sent_vectors, aud_data, h)
            else:
                predictions = model(sent_vectors, h)
        else:
            with torch.no_grad():
                if is_fusion:
                    predictions = model(sent_vectors, aud_data, h)
                else:
                    predictions = model(sent_vectors, h)
        loss = loss_fcn(predictions, labels)
        # add on to the total loss
        total_loss = total_loss + loss.item()
        if is_training:
            loss.backward()  # backward pass to calculate the gradients
            if clip_at:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_at)
            # update parameters
            optimizer.step()
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
    model_predictions = np.argmax(model_predictions, axis=1)
    return avg_loss, model_predictions, model_labels, model


def get_model_to_train(fusion, vocab_size, embedding_dim, hidden_dim, seq_len, n_layers, dropout_level,
                       run_on, mfcc_len=41, img_height=80, img_width=80, bidir=False, max_norm=2):
    if fusion == FusionTypes.MFCC:
        model = LstmMfccModel(vocab_size, embedding_dim, hidden_dim, seq_len,
                              fusion=True, n_layers=n_layers, dropout_level=dropout_level,
                              mfcc_len=mfcc_len)
        is_fusion = True
    elif fusion == FusionTypes.MEL:
        model = LstmMelModel(vocab_size, embedding_dim, hidden_dim, seq_len,
                             fusion=True, n_layers=n_layers, dropout_level=dropout_level,
                             img_height=img_height, img_width=img_width)
        is_fusion = True
    else:
        model = LstmModel(vocab_size, embedding_dim, hidden_dim, seq_len,
                          n_layers=n_layers, dropout_level=dropout_level, bidir=bidir, max_norm=max_norm)
        is_fusion = False
    model.to(run_on)
    return model, is_fusion


def prepare_train_test_data(data, fusion, train_ids, test_ids, max_seq_len, word_threshold,
                            img_path, img_height=80, img_width=80):
    x_train = [data['Lyric'].iloc[x] for x in train_ids]
    x_test = [data['Lyric'].iloc[x] for x in test_ids]
    y_train = [data['iGenre'].iloc[x] for x in train_ids]
    y_test = [data['iGenre'].iloc[x] for x in test_ids]
    word_counter, word_sequences, word2idx, idx2word = build_word_tokenizer(x_train, word_threshold=word_threshold)
    vocab_size = len(word2idx) + 1
    train_tokens = tokenize(x_train, word2idx, seq_len=max_seq_len)
    test_tokens = tokenize(x_test, word2idx, seq_len=max_seq_len)
    if fusion == FusionTypes.MFCC:
        train_mfcc = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in train_ids])
        test_mfcc = torch.tensor([[_ for _ in data['mfcc_mean'].iloc[x]] for x in test_ids])
        train_data = LstmMfccDataset(train_tokens, train_mfcc, y_train)
        test_data = LstmMfccDataset(test_tokens, test_mfcc, y_test)
    elif fusion == FusionTypes.MEL:
        train_aud = [data['MP3 File'].iloc[x][0:-4] + ".png" for x in train_ids]
        test_aud = [data['MP3 File'].iloc[x][0:-4] + ".png" for x in test_ids]
        train_data = LstmMelDataset(train_tokens, train_aud, y_train, image_path=img_path, image_width=img_width,
                                    image_height=img_height)
        test_data = LstmMelDataset(test_tokens, test_aud, y_test, image_path=img_path, image_width=img_width,
                                   image_height=img_height)
    else:
        train_data = LstmDataset(train_tokens, y_train)
        test_data = LstmDataset(test_tokens, y_test)
    return vocab_size, train_data, test_data


def run_k_fold(device, data, max_seq_len=75, fusion=None, k_folds=5,
               epochs=5, balance_classes=False, embedding_dim=256,
               dropout_level=0.25, lr=1e-5,
               hidden_dim=256, n_layers=6, clip_at=1.0,
               mfcc_len=41, img_height=80, img_width=80, img_path=None,
               word_threshold=5, batch_size=16, bidir=False, max_norm=2):
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
        vocab_size, train_data, valid_data = prepare_train_test_data(data, fusion, train_ids, test_ids, max_seq_len,
                                                                     word_threshold, img_path, img_height, img_width)
        model, is_fusion = get_model_to_train(fusion, vocab_size, embedding_dim, hidden_dim, max_seq_len,
                                              n_layers, dropout_level, device, mfcc_len,
                                              img_height, img_width, bidir=bidir, max_norm=max_norm)
        loss_fcn = get_loss_function(balance_classes, labels[train_ids], device)
        # define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_schedulers = [ReduceLROnPlateau(optimizer, 'min')]
        for epoch in range(epochs):
            e_start = datetime.datetime.now()
            # train model
            train_loss, train_predictions, train_labels, model = run_model(model,
                                                                           train_data.get_data_loader(
                                                                               batch_size=batch_size),
                                                                           loss_fcn,
                                                                           optimizer,
                                                                           device,
                                                                           is_training=True,
                                                                           clip_at=clip_at,
                                                                           is_fusion=is_fusion)
            # evaluate model
            valid_loss, valid_predictions, valid_labels, model = run_model(model,
                                                                           valid_data.get_data_loader(
                                                                               batch_size=batch_size),
                                                                           loss_fcn,
                                                                           optimizer,
                                                                           device,
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
                                             valid_labels, valid_predictions,
                                             model=model,
                                             model_file_name=f'saved_weights_Fold_{fold}.pt')
            # append training and validation loss
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            results[fold] = update_results_dict(results[fold],
                                                train_labels, train_predictions,
                                                valid_labels, valid_predictions)
            e_end = datetime.datetime.now()
            print(
                f'Epoch {epoch + 1}/{epochs}, Train Loss {train_loss:.3f}/Validation Loss {valid_loss:.3f} [Time:  {(e_end - e_start).total_seconds()} seconds]')
        print('On Train Data')
        print(confusion_matrix(best_scores['train_labels'], best_scores['train_predictions']))
        print('On Validation Data')
        print(confusion_matrix(best_scores['test_labels'], best_scores['test_predictions']))
        results[fold]['train_losses'] = train_losses
        results[fold]['validation_losses'] = valid_losses
        print(f'Time for fold {fold} : {(datetime.datetime.now() - fold_start).total_seconds()} seconds')
    end_time = datetime.datetime.now()
    print(f'Overall Time : {(end_time - start_time).total_seconds()} seconds')
    return results, results_to_df(results)
