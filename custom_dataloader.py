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

    def __init__(self, root_dir, data_subset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            data_subset_type (string): Specify if the required data is "training", "validation", or "test"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir + data_subset_type + "/"
        self.data_subset_type = data_subset_type
        self.transform = transform

    def __len__(self):
        subset_size = 0
        if self.data_subset_type == settings.TRAIN_SUBSET:
            subset_size = settings.TRAIN_SET_SIZE
        elif self.data_subset_type == settings.VAL_SUBSET:
            subset_size = settings.VAL_SET_SIZE
        elif self.data_subset_type == settings.TEST_SUBSET:
            subset_size = settings.TEST_SET_SIZE
        return subset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        landmark_file = self.root_dir + "matched_landmarks/" + self.data_subset_type + "_" + str(idx) + ".csv"
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
        # pdb.set_trace()
        return {'landmarks': torch.tensor(landmarks.values),
                'cm_parameters': torch.tensor(cm_parameters.values)}


class ZeroPadding(object):
    """Apply padding to handle variable-length data."""

    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        # pdb.set_trace()
        # TODO: Later when we are shuffling, just sample K landmarks every time
        k_expected_max_landmarks = 1200
        pad_size = k_expected_max_landmarks - len(landmarks)
        padded_landmarks = func.pad(landmarks, pad=(0, 0, 0, pad_size))
        return {'landmarks': padded_landmarks,
                'cm_parameters': cm_parameters}


class Normalise(object):
    """Perform normalisation."""

    # TODO: implement this properly, it's not finished.
    def __call__(self, sample):
        landmarks, cm_parameters = sample['landmarks'], sample['cm_parameters']
        mean = settings.LANDMARK_MEAN
        std_dev = settings.LANDMARK_STD_DEV
        scaled_cm_parameters = (cm_parameters - mean) / std_dev
        return {'landmarks': landmarks,
                'cm_parameters': scaled_cm_parameters}


class LandmarksDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.root_dir = "/workspace/data/landmark-distortion/tmp_data_store/"
        self.batch_size = 2
        # self.transform = transforms.Compose([transforms.ToTensor()])

    # def prepare_data(self):
    #     # Not sure we need this function yet
    #     # gt_cm_parameters = pd.read_csv(self.root_dir + "gt_poses.csv", header=None)
    #     raise NotImplementedError

    def setup(self, stage=None):
        data_transform_for_training = transforms.Compose([ToTensor(), Normalise(), ZeroPadding()])
        data = LandmarkDataset(root_dir=settings.DATA_DIR, data_subset_type=settings.TRAIN_SUBSET,
                               transform=data_transform_for_training)
        self.train_data = data
        self.valid_data = None
        self.test_data = None

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size)


def main():
    # Define a main loop to run and show some example data if this script is run as main
    landmark_dataset = LandmarkDataset(root_dir=settings.DATA_DIR, data_subset_type=settings.TRAIN_SUBSET,
                                       transform=None)
    sample_data = landmark_dataset[0]
    landmarks = np.array(sample_data['landmarks'])
    cm_parameters = np.array(sample_data['cm_parameters'])

    print("Landmarks shape:", landmarks.shape)
    print("CM parameters shape:", cm_parameters.shape)
    pdb.set_trace()


if __name__ == "__main__":
    main()
