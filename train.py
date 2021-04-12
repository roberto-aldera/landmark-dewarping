import pytorch_lightning as pl
from argparse import ArgumentParser
from pathlib import Path
import time
from cmnet import CMNet
from custom_dataloader import LandmarksDataModule
import settings

if __name__ == '__main__':
    start_time = time.time()

    Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default="cmnet", help="cmnet or ...")
    # parser.add_argument('--num_samples_to_evaluate', type=int, default=50, help="...")

    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.model_name == "cmnet":
        parser = CMNet.add_model_specific_args(parser)

    params = parser.parse_args()

    # pick model
    model = None
    if params.model_name == "cmnet":
        model = CMNet(params)

    trainer = pl.Trainer.from_argparse_args(params,
                                            default_root_dir=settings.MODEL_DIR,
                                            max_epochs=params.max_num_epochs)
    dm = LandmarksDataModule()
    trainer.fit(model, dm)

    # save checkpoint that will be used for running evaluation
    path_to_model = "%s%s%s" % (settings.MODEL_DIR, params.model_name, ".ckpt")
    trainer.save_checkpoint(path_to_model)

    print("Finished Training")
    print("--- Training execution time: %s seconds ---" % (time.time() - start_time))
