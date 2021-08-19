import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from pathlib import Path
import time
from pointnet import PointNet
from scorenet import ScoreNet
from pointnet_dataloader import LandmarksDataModule
import settings

if __name__ == '__main__':
    start_time = time.time()
    torch.autograd.set_detect_anomaly(True)

    Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default="scorenet")
    # parser.add_argument('--num_samples_to_evaluate', type=int, default=50, help="...")

    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.model_name == "pointnet":
        parser = PointNet.add_model_specific_args(parser)
    if temp_args.model_name == "scorenet":
        parser = ScoreNet.add_model_specific_args(parser)

    params = parser.parse_args()

    # pick model
    model = None
    if params.model_name == "pointnet":
        model = PointNet(params)
    if params.model_name == "scorenet":
        model = ScoreNet(params)

    trainer = pl.Trainer.from_argparse_args(params,
                                            default_root_dir=settings.MODEL_DIR,
                                            max_epochs=params.max_num_epochs)
    # gradient_clip_val=0.05)  # here if required
    dm = LandmarksDataModule()
    trainer.fit(model, dm)

    # save checkpoint that will be used for running evaluation
    path_to_model = "%s%s%s" % (settings.MODEL_DIR, params.model_name, ".ckpt")
    trainer.save_checkpoint(path_to_model)

    print("Finished Training")
    print("--- Training execution time: %s seconds ---" % (time.time() - start_time))
