from pointnet_dataloader import LandmarksDataModule
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from pointnet_dataloader import LandmarksDataModule, LandmarkDataset, ToTensor, Normalise, FixSampleSize
import settings

import pdb

if __name__ == "__main__":
    print("Running data checker...")
    # dm = LandmarksDataModule()
    transform = transforms.Compose([ToTensor(), Normalise(), FixSampleSize()])
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

    # Check limits of data
    cme_0 = []
    cme_1 = []
    all_landmarks = []
    for sample in train_data:
        landmarks = np.array(sample['landmarks'])
        cm_parameters = np.array(sample['cm_parameters'])
        cme_0.append(cm_parameters[0])
        cme_1.append(cm_parameters[1])
        all_landmarks.append(landmarks)
        # pdb.set_trace()
    print("Max cme 0, 1:", max(cme_0), ", ", max(cme_1))
    print("Min cme 0, 1:", min(cme_0), ", ", min(cme_1))
    print("Max and min landmark value:", np.max(all_landmarks), ", ", np.min(all_landmarks))

    print("Any landmark nans?", np.isnan(all_landmarks).any())
    print("Any cme_0 nans?", np.isnan(cme_0).any())
    print("Any cme_1 nans?", np.isnan(cme_1).any())

    print("Any landmark infs?", np.isinf(all_landmarks).any())
    print("Any cme_0 infs?", np.isinf(cme_0).any())
    print("Any cme_1 infs?", np.isinf(cme_1).any())

    pdb.set_trace()
