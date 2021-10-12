import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from PIL import Image


def get_data_loader(dataset, batch_size=16, random_seed=42):
    g = torch.Generator()
    g.manual_seed(random_seed)
    return DataLoader(dataset, batch_size=batch_size, generator=g)


class BertDataset(Dataset):
    def __init__(self, sequences, masks, labels):
        self.sequences = sequences
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.masks[index], self.labels[index]

    def get_data_loader(self, batch_size=16, random_seed=42):
        return get_data_loader(self, batch_size, random_seed)


class BertMelDataset(IterableDataset):
    def __init__(self, sequences, masks, labels, image_names, image_path, image_width=80, image_height=80):
        super().__init__()
        self.sequences = sequences
        self.masks = masks
        self.labels = labels
        self.image_names = image_names
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height

    def __load_image__(self, index):
        img_path = os.path.join(self.image_path, self.image_names[index])
        frame = np.asarray(Image.open(img_path))
        frame_resized = np.array(Image.fromarray(frame).resize((self.image_width, self.image_height)))
        frame_resized = frame_resized / 255.0
        return frame_resized

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for index, _ in enumerate(self.image_names):
            yield self.sequences[index], self.masks[index], self.__load_image__(index), self.labels[index]

    def __getitem__(self, index):
        return self.sequences[index], self.masks[index], self.__load_image__(index), self.labels[index]

    def get_data_loader(self, batch_size=16, random_seed=42):
        return get_data_loader(self, batch_size, random_seed)


class BertMfccDataset(BertDataset):
    def __init__(self, sequences, masks, labels, mfcc_data):
        super().__init__(sequences, masks, labels)
        self.mfcc_data = mfcc_data

    def __getitem__(self, index):
        s, m, label = super().__getitem__(index)
        return s, m, self.mfcc_data[index], label


class FtDataSet(Dataset):
    def __init__(self, sent_vectors, labels):
        self.sent_vectors = sent_vectors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sent_vectors[index], self.labels[index]

    def get_data_loader(self, batch_size=16, random_seed=42):
        return get_data_loader(self, batch_size, random_seed)


class FtMfccDataSet(FtDataSet):
    def __init__(self, sent_vectors, labels, mfcc_data):
        super().__init__(sent_vectors, labels)
        self.mfcc_data = mfcc_data

    def __getitem__(self, index):
        s, label = super().__getitem__(index)
        return s, self.mfcc_data[index], label


class FtMelDataset(IterableDataset):
    def __init__(self, sent_vectors, labels, image_names, image_path, image_width=80, image_height=80):
        super().__init__()
        self.sent_vectors = sent_vectors
        self.labels = labels
        self.image_names = image_names
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height

    def __load_image__(self, index):
        img_path = os.path.join(self.image_path, self.image_names[index])
        frame = np.asarray(Image.open(img_path))
        frame_resized = np.array(Image.fromarray(frame).resize((self.image_width, self.image_height)))
        frame_resized = frame_resized / 255.0
        return frame_resized

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for index, _ in enumerate(self.image_names):
            yield self.sent_vectors[index], self.__load_image__(index), self.labels[index]

    def __getitem__(self, index):
        return self.sent_vectors[index], self.__load_image__(index), self.labels[index]

    def get_data_loader(self, batch_size=16, random_seed=42):
        return get_data_loader(self, batch_size, random_seed)


class LstmDataset(Dataset):
    def __init__(self, tokens, labels):
        super().__init__()
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.tokens[index], self.labels[index]

    def get_data_loader(self, batch_size=16, random_seed=42):
        return get_data_loader(self, batch_size, random_seed)


class LstmMfccDataset(LstmDataset):
    def __init__(self, tokens, mfcc_data, labels):
        super().__init__(tokens, labels)
        self.mfcc_data = mfcc_data

    def __getitem__(self, index):
        s, label = super().__getitem__(index)
        return s, self.mfcc_data[index], label


class LstmMelDataset(IterableDataset):
    def __init__(self, tokens, image_names, labels, image_path, image_width=80, image_height=80):
        super().__init__()
        self.tokens = tokens
        self.labels = labels
        self.image_names = image_names
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height

    def __load_image__(self, index):
        img_path = os.path.join(self.image_path, self.image_names[index])
        frame = np.asarray(Image.open(img_path))
        frame_resized = np.array(Image.fromarray(frame).resize((self.image_width, self.image_height)))
        frame_resized = frame_resized / 255.0
        return frame_resized

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for index, _ in enumerate(self.image_names):
            yield self.tokens[index], self.__load_image__(index), self.labels[index]

    def __getitem__(self, index):
        return self.tokens[index], self.__load_image__(index), self.labels[index]

    def get_data_loader(self, batch_size=16, random_seed=42):
        return get_data_loader(self, batch_size, random_seed)
