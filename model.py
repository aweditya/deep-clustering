import torch
from torch import nn
from sklearn.cluster import KMeans

from asteroid import torch_utils
from asteroid_filterbanks.transforms import mag, apply_mag_mask
from asteroid_filterbanks import make_enc_dec
from asteroid.dsp.vad import ebased_vad
from asteroid.masknn.recurrent import SingleRNN
from asteroid.utils.torch_utils import pad_x_to_y

# This is the base Deep Clustering model without the Mask Inference head used in Chimera++
# Adapted from https://github.com/asteroid-team/asteroid/blob/master/egs/wsj0-mix/DeepClustering/model.py

def make_model(conf):
    encoder, decoder = make_enc_dec('stft', **conf["filterbank"])
    embedding = Embedding(encoder.n_feats_out // 2, **conf["deepclustering"])
    model = Model(encoder, embedding, decoder)
    return model

class Embedding(nn.Module):
    def __init__(
        self,
        channel_in,
        n_src=2,
        rnn_type='lstm',
        n_layers=2,
        hidden_layer_size=600,
        dropout=0.3,
        embedding_dim=40,
        take_log=True,
        epsilon=1e-8
    ):
        super().__init__()
        self.channel_in = channel_in # channel_in = freq
        self.n_src = n_src
        self.take_log = take_log
        self.embedding_dim = embedding_dim
        self.lstm = SingleRNN(
            rnn_type,
            channel_in,
            hidden_layer_size,
            n_layers=n_layers,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        lstm_output_dim = 2 * hidden_layer_size
        self.embedding_layer = nn.Linear(lstm_output_dim, channel_in * embedding_dim)
        self.embedding_activation = nn.Tanh()
        self.epsilon = epsilon

    def forward(self, input_data):
        batch_size, _, frames = input_data.shape
        if self.take_log:
            x = torch.log(input_data + self.epsilon)

        # LSTM layers
        lstm_output = self.lstm(x.permute(0, 2, 1))
        lstm_output = self.dropout(lstm_output)

        # Fully connected layer
        embedding_out = self.embedding_layer(lstm_output) # Shape is (batch_size, time, freq * embedding_size)
        embedding_out = self.embedding_activation(embedding_out)

        # Make shape (batch_size, freq, time, embedding_size)
        embedding_out = embedding_out.view(batch_size, frames, -1, self.embedding_dim)
        embedding_out = embedding_out.transpose(1, 2)

        # Make shape (batch_size, freq * time, embedding_size)
        embedding_out = embedding_out.reshape(batch_size, -1, self.embedding_dim)

        # Normalise (the embedding vector for each time * freq bin should be of unit norm)
        embedding_norm = torch.norm(embedding_out, p=2, dim=-1, keepdim=True)
        normalised_embedding = embedding_out / (embedding_norm + self.epsilon)

        return normalised_embedding

class Model(nn.Module):
    def __init__(self, encoder, embedding, decoder):
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_representation = self.encoder(x)
        spectral_magnitude = mag(tf_representation)
        normalised_embedding = self.embedding(spectral_magnitude)
        return normalised_embedding

    def cluster(self, x):
        kmeans = KMeans(n_clusters=self.Embedding.n_src)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_representation = self.encode(x)
        spectral_magnitude = mag(tf_representation)
        normalised_embedding = self.embedding(spectral_magnitude)

        # Ignore time-frequency with energy < -40dB
        # ebased_vad = Energy based voice activity detection
        retained_bins = ebased_vad(spectral_magnitude)
        retained_embedding = normalised_embedding[retained_bins.view(1, -1)]

        clusters = kmeans.fit_predict(retained_embedding.cpu().data.numpy())

        # Create masks
        est_masks = []
        for i in range(self.Embedding.n_src):
            mask = ~retained_bins
            mask[retained_bins] = torch.from_numpy((clusters == i)).to(mask.device)
            est_masks.append(mask.float())

        # Apply the mask
        estimated_masks = torch.stack(est_masks, dim=1)
        masked_representation = apply_mag_mask(tf_representation, estimated_masks)
        # Pad masked audio to have same size as original
        separated_wav = pad_x_to_y(self.decoder(masked), x)
        return separated_wav
