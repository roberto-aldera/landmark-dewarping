import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import torch.nn.functional as func
import settings
import pdb


class LandmarkDataset:
    """Landmark dataset."""

    def __init__(self, root_dir, is_training_data=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): Specify if the required data is for training/validation, or testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if is_training_data:
            self.root_dir = root_dir + "training" + "/"
        else:
            self.root_dir = root_dir + "test" + "/"
        self.is_training_data = is_training_data
        self.transform = transform

    def __len__(self):
        if self.is_training_data:
            subset_size = settings.TOTAL_SAMPLES
        else:
            subset_size = 0
            raise NotImplementedError
        return subset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.is_training_data:
            landmark_file = self.root_dir + "matched_landmarks/training_" + str(idx) + ".csv"
        else:
            landmark_file = self.root_dir + "matched_landmarks/test_" + str(idx) + ".csv"

        landmarks = pd.read_csv(landmark_file, header=None)
        all_cm_parameters = pd.read_csv(self.root_dir + "gt_poses.csv", header=None)
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


class SubsetSampling(object):
    """Apply random subset sampling to pick N matches"""

    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        if len(landmarks) > settings.K_MAX_MATCHES:
            processed_landmarks = landmarks[torch.randperm(len(landmarks))[:settings.K_MAX_MATCHES]]
        else:
            processed_landmarks = landmarks
        return {'landmarks': processed_landmarks,
                'cm_parameters': cm_parameters}


class ZeroPadding(object):
    """Apply padding to handle variable-length data."""

    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        pad_size = max(settings.K_MAX_MATCHES - len(landmarks), 0)
        padded_landmarks = func.pad(landmarks, pad=(0, 0, 0, pad_size))
        return {'landmarks': padded_landmarks,
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
        self.transform = transforms.Compose([ToTensor(), Normalise(), SubsetSampling(), ZeroPadding()])

    # def prepare_data(self):
    #     raise NotImplementedError

    def setup(self, stage=None):
        data_full = LandmarkDataset(root_dir=settings.DATA_DIR, is_training_data=True,
                                    transform=self.transform)
        self.train_data, self.valid_data = random_split(data_full, [settings.TRAIN_SET_SIZE, settings.VAL_SET_SIZE])
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
    transform = transforms.Compose([ToTensor(), Normalise(), SubsetSampling(), ZeroPadding()])
    data_full = LandmarkDataset(root_dir=settings.DATA_DIR, is_training_data=True,
                                transform=transform)
    train_data, valid_data = random_split(data_full, [settings.TRAIN_SET_SIZE, settings.VAL_SET_SIZE],
                                          generator=torch.Generator().manual_seed(
                                              0))  # seed is already set in settings with pl.seed_everything(0)

    print("Training dataset size:", len(train_data))
    print("Training dataset size:", len(valid_data))

    sample_data = train_data[0]
    landmarks = np.array(sample_data['landmarks'])
    cm_parameters = np.array(sample_data['cm_parameters'])
    print("Landmarks shape:", landmarks.shape)
    print("CM parameters shape:", cm_parameters.shape)
    pdb.set_trace()


if __name__ == "__main__":
    main()
