import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms
import os
import settings
import pdb


class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.landmark_file_names = []
        self.cm_parameters = []
        self.training_set_size = 0
        self.validation_set_size = 0
        self.transform = transform

        dataset_index_file = self.data_root + "dataset_index.txt"

        with open(dataset_index_file) as f:
            self.landmark_file_names = [line.rstrip() for line in f]  # f.readlines()

        self.cm_parameters = pd.read_csv(self.data_root + "all_gt_poses.csv", header=None)

        self.training_set_size = int(len(self.landmark_file_names) * settings.TRAIN_RATIO)
        self.validation_set_size = len(self.landmark_file_names) - self.training_set_size

    def __len__(self):
        return len(self.landmark_file_names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        landmarks = pd.read_csv(self.landmark_file_names[index], header=None)
        sample = {'landmarks': landmarks, 'cm_parameters': self.cm_parameters.iloc[index, 1:3]}

        if self.transform:
            sample = self.transform(sample)  # look at shuffling here later

        return sample


class SingleDataset:

    def __init__(self, root_dir, transform=None):
        self.data_root = root_dir
        self.transform = transform
        self.training_set_size = 0
        self.validation_set_size = 0

        landmark_folder = self.data_root + "exported_matched_landmarks/"
        num_instances = len(os.listdir(landmark_folder))
        self.training_set_size = int(num_instances * settings.TRAIN_RATIO)
        self.validation_set_size = num_instances - self.training_set_size

    def __len__(self):
        landmark_folder = self.data_root + "exported_matched_landmarks/"
        num_instances = len(os.listdir(landmark_folder))

        return num_instances

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmark_file = self.data_root + "exported_matched_landmarks/training_" + str(idx) + ".csv"

        landmarks = pd.read_csv(landmark_file, header=None)
        all_cm_parameters = pd.read_csv(self.data_root + "gt_poses.csv", header=None)
        # Get the cm parameters for this particular match set (just theta and curvature)
        cm_parameters = all_cm_parameters.iloc[idx, 1:3]
        sample = {'landmarks': landmarks, 'cm_parameters': cm_parameters}

        if self.transform:
            sample = self.transform(sample)  # look at shuffling here later

        return sample


def CollateFn(batch):
    landmarks = torch.stack([x['landmarks'] for x in batch])
    cm_parameters = torch.stack([x['cm_parameters'] for x in batch])
    return landmarks, cm_parameters


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        return {'landmarks': torch.tensor(landmarks.values),
                'cm_parameters': torch.tensor(cm_parameters.values)}


class FixSampleSize(object):
    """Ensure all samples are of same size, either by duplicating points or subsampling"""

    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        if len(landmarks) > settings.K_MAX_MATCHES:
            # Length is too long, so take a random subsample
            processed_landmarks = landmarks[torch.randperm(len(landmarks))[:settings.K_MAX_MATCHES]]
        elif len(landmarks) < settings.K_MAX_MATCHES:
            # Length is too short, so repeat a few samples
            duplicates_needed = settings.K_MAX_MATCHES - len(landmarks)
            duplicates = landmarks[torch.randint(0, len(landmarks), (duplicates_needed,))]
            processed_landmarks = torch.cat((landmarks, duplicates))
        else:
            processed_landmarks = landmarks
        return {'landmarks': processed_landmarks,
                'cm_parameters': cm_parameters}


class Normalise(object):
    """Perform normalisation to scale landmark ranges to be between [-1, 1]."""

    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        scaled_landmarks = landmarks / settings.MAX_LANDMARK_RANGE_METRES
        return {'landmarks': scaled_landmarks,
                'cm_parameters': cm_parameters}


class LandmarksDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.root_dir = settings.DATA_DIR
        self.batch_size = settings.BATCH_SIZE
        self.transform = transforms.Compose([ToTensor(), Normalise(), FixSampleSize()])
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    # def prepare_data(self):
    #     raise NotImplementedError

    def setup(self, stage=None):
        # data_full = CustomDataset(data_root=settings.DATA_DIR, transform=self.transform)

        # Using just a single dataset for quicker training here:
        data_full = SingleDataset(root_dir=settings.EVALUATION_DATA_DIR, transform=self.transform)

        self.train_data, self.valid_data = random_split(data_full,
                                                        [data_full.training_set_size, data_full.validation_set_size])
        self.test_data = None

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)
        # collate_fn=CollateFn)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


def main():
    # Define a main loop to run and show some example data if this script is run as main
    transform = transforms.Compose([ToTensor(), Normalise(), FixSampleSize()])
    data_full = CustomDataset(data_root=settings.DATA_DIR, transform=transform)
    train_data, valid_data = random_split(data_full, [data_full.training_set_size, data_full.validation_set_size],
                                          generator=torch.Generator().manual_seed(
                                              0))  # seed is already set in settings with pl.seed_everything(0)

    print("Training dataset size:", len(train_data))
    print("Validation dataset size:", len(valid_data))

    sample_data = train_data[0]
    landmarks = np.array(sample_data['landmarks'])
    cm_parameters = np.array(sample_data['cm_parameters'])
    print("Landmarks shape:", landmarks.shape)
    print("CM parameters shape:", cm_parameters.shape)
    pdb.set_trace()


if __name__ == "__main__":
    main()
