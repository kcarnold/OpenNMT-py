import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import h5py


class VecsEncoder(nn.Module):
    """
    A placeholder "encoder" than uses pre-supplied vectors stored in an HDF5 file.

    Linear layers are used to initialize the decoder LSTM's hidden and cell states.
    """
    def __init__(self, rnn_size, h5_filename):
        super(VecsEncoder, self).__init__()
        self.h5_filename = h5_filename
        self.num_objs, self.feature_dim = self._open_h5_file()
        self.init_hidden = nn.Linear(in_features=self.feature_dim, out_features=rnn_size, bias=True)
        self.init_cell = nn.Linear(in_features=self.feature_dim, out_features=rnn_size, bias=True)

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
            del self.h5_file

    def load_pretrained_vectors(self, opt):
        # Pass in needed options only when modify function definition.
        pass

    def forward(self, input, lengths=None):
        "See :obj:`onmt.modules.EncoderBase.forward()`"

        vecs = Variable(
            self.load_h5_data(input),
            requires_grad=False)#,
            #volatile=input.volatile)

        return self.forward_given_vecs(vecs)

    def _open_h5_file(self):
        self.h5_file = h5py.File(self.h5_filename, 'r')
        self.h5_dataset = self.h5_file['image_features']
        num_vecs, num_objs, feature_dim = self.h5_dataset.shape
        return num_objs, feature_dim


    def load_h5_data(self, indices):
        batch_size = len(indices)
        vecs = np.empty((batch_size, self.num_objs, self.feature_dim))
        for i, idx in enumerate(indices):
            vecs[i] = self.h5_dataset[idx]

        # Features need to be first, since they're analogous to words.
        vecs = vecs.transpose(1, 0, 2)
        # vecs: objs x batch_size x feature_dim

        return torch.FloatTensor(vecs)

    def forward_given_vecs(self, vecs):
        # vecs: objs x batch_size x feature_dim
        mean_feature = torch.mean(vecs, dim=0)  # batch_size x feature_dim

        # Construct the hidden and cell states.
        hidden_state = F.tanh(self.init_hidden(mean_feature))
        cell_state = F.tanh(self.init_cell(mean_feature))
        # hidden_state: batch_size x rnn_size

        # To make this look like the output of a sequence RNN, states need to
        # have an extra first dimension (per decoder layer) and be packed in a
        # tuple.

        enc_final = (
            hidden_state.unsqueeze(0),
            cell_state.unsqueeze(0)
        )

        return enc_final, vecs
