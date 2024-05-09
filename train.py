import argparse
import os
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from models import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data_loader(configs, sort=True, drop_last=True, group_size=4):
    """
    Prepare dataset and dataloader of training
    @param: configs: a tuple consists of preprocess_config, model_config, train_config
    @rtype: object
    @return: a tuple (dataset, loader)
    """
    preprocess_config, model_config, train_config = configs
    batch_size = train_config["optimizer"]["batch_size"]

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=sort, drop_last=drop_last
    )

    # Get dataloader
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    return dataset, loader


def prepare_logger(train_config):
    """
    prepare logger for training
    @param train_config: train_config
    @return: a tuple consists of (train_log_path, val_log_path, train_logger, val_logger)
    """
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    return train_log_path, val_log_path, train_logger, val_logger


def train(configs, data):
    """
    The training process of ctts
    @param configs: a tuple consists of preprocess_config, model_config, train_config
    @param data: necessary training datas
    """

    # parse train data
    preprocess_config, model_config, train_config = configs
    dataset, loader, model, optimizer, Loss, vocoder, train_log_path, val_log_path, train_logger, val_logger = data

    # Training
    step = 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    # Initialize tqdm
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = 0
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batches in loader:
            for batch in batches:
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)
                total_loss = losses[0]
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                # Save logs
                if step % log_step == 0:
                    losses = [loss.item() for loss in losses]
                    message1 = f"Step {step}/{total_step},"
                    message2 = ("Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, "
                                "Energy Loss: {:.4f}, Duration Loss: {:.4f}").format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                # Evaluate
                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                # Save checkpoint
                if step % save_step == 0:
                    model_state_dict = model.module.state_dict()

                    torch.save(
                        {
                            "models": model_state_dict,
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


def main(args, configs):
    print("Prepare training ...")

    # Read config
    preprocess_config, model_config, train_config = configs

    # Get dataset and dataloader
    dataset, loader = prepare_data_loader(configs)

    # Prepare models
    model, optimizer = get_model(args, configs, device, train=True, strict_load=False)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print(f"Number of parameters: {num_param}")

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    train_log_path, val_log_path, train_logger, val_logger = prepare_logger(train_config)

    train_data = (
        dataset,
        loader,
        model,
        optimizer,
        Loss,
        vocoder,
        train_log_path,
        val_log_path,
        train_logger,
        val_logger
    )

    # Start training
    train(configs, train_data)


if __name__ == "__main__":
    # torch.manual_seed(3407)

    config_root = "./config"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ESD_en",
        help="Name of the models used. For example: LJSpeech, ESD_en, ESD_zh"
    )

    parser.add_argument(
        "-pp",
        "--pretrain_path",
        type=str,
        help="path to the pretrained models"
    )

    input_args = parser.parse_args()

    # Get path by models name
    preprocess_config_path = os.path.join("./config/", input_args.model, "preprocess.yaml")
    model_config_path = os.path.join("./config/", input_args.model, "model.yaml")
    train_config_path = os.path.join("./config/", input_args.model, "train.yaml")

    # Load config files
    input_preprocess_config = yaml.load(
        open(preprocess_config_path, "r"), Loader=yaml.FullLoader
    )
    input_model_config = yaml.load(
        open(model_config_path, "r"), Loader=yaml.FullLoader
    )
    input_train_config = yaml.load(
        open(train_config_path, "r"), Loader=yaml.FullLoader
    )
    input_configs = (input_preprocess_config, input_model_config, input_train_config)

    # Start train
    main(input_args, input_configs)
