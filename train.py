import os
import argparse
import json

import torch
from asteroid.data import LibriMix
from torch.utils.data import DataLoader
from asteroid_filterbanks.transforms import mag
from asteroid.dsp.vad import ebased_vad
from asteroid.losses import deep_clustering_loss

from model import make_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

def main(conf):
    train_set = LibriMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    val_set = LibriMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["deepclustering"].update({"n_src": conf["data"]["n_src"]})

    # Define the model, loss function and optimizer
    model = make_model(conf)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=conf["optim"]["lr"], momentum=conf["optim"]["momentum"])

    # Train the model
    train(train_loader, model, optimizer)

def train(train_loader, model, optimizer, epsilon=1e-8):
    size = len(train_loader.dataset)
    model.train()

    for batch, (mixture, sources) in enumerate(train_loader):
        mixture, sources = mixture.to(device), sources.to(device)

        # Compute magnitude spectrograms and ideal ratio mask (IRM)
        sources_magnitude_spectrogram = mag(model.encoder(sources))

        # Normalise to get the real_mask. Maximise to get the binary mask
        real_mask = sources_magnitude_spectrogram / (sources_magnitude_spectrogram.sum(1, keepdim=True) + epsilon)
        binary_mask = real_mask.argmax(1)

        # Compute loss
        est_embeddings = model(mixture)
        spectral_magnitude = mag(model.encoder(mixture.unsqueeze(1)))
        silence_mask = ebased_vad(spectral_magnitude)
        dc_loss = deep_clustering_loss(est_embeddings, binary_mask, silence_mask)
        
        # deep_clustering_loss is a tensor. Use its mean for backpropagation
        loss = dc_loss.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:7f} [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
