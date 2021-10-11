from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import enum

from sklearn.metrics import precision_score, f1_score, recall_score


def get_devices():
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
    else:
        gpu = torch.device("cpu")
    cpu = torch.device("cpu")
    return gpu, cpu


class FusionTypes(enum.Enum):
    TXT = 0
    MFCC = 1
    MEL = 2


class DummyModel(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.layers = sequential

    def forward(self, x):
        return self.layers(x)


def seq_layer_size(layer, ip_shape):
    if isinstance(layer, nn.Sequential):
        obj = DummyModel(layer)
        op = obj(torch.randn(ip_shape))
        return op.data.shape[1]
    elif hasattr(layer, 'out_features'):
        return layer.out_features
    return 0


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


def update_results_dict(results, train_labels, train_predictions, test_labels, test_predictions):
    results['train_precision'].append(precision_score(train_labels, train_predictions))
    results['train_recall'].append(recall_score(train_labels, train_predictions))
    results['train_f1'].append(f1_score(train_labels, train_predictions))
    results['validation_precision'].append(precision_score(test_labels, test_predictions))
    results['validation_recall'].append(recall_score(test_labels, test_predictions))
    results['validation_f1'].append(f1_score(test_labels, test_predictions))
    return results


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


def results_to_df(results):
    p = pd.DataFrame(results[0])
    for i in range(1, len(results)):
        p = pd.concat([p, pd.DataFrame(results[i])], axis=0)
    p.sort_values(by='validation_f1', ascending=False, inplace=True)
    return p
