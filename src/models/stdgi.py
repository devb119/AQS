import torch.nn as nn
import torch 
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, h_ft, x_ft, hid_ft):
        super(Discriminator, self).__init__()
        self.fc = nn.Bilinear(h_ft, x_ft, out_features=hid_ft)
        self.linear = nn.Linear(hid_ft, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, h, x, x_c):
        ret1 = self.relu(self.fc(h, x)) 
        ret1 = self.linear(ret1)
        ret2 = self.relu(self.fc(h, x_c))
        ret2 = self.linear(ret2)
        ret = torch.cat((ret1, ret2), 2)
        return self.sigmoid(ret)
    
class GCN(nn.Module):
    def __init__(self, infea, outfea, act="relu", bias=True):
        super(GCN, self).__init__()
        # define cac lop fc -> act
        self.fc = nn.Linear(infea, outfea, bias=False)
        self.act = nn.ReLU() if act == "relu" else nn.ReLU()

        # init bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outfea))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        # init weight
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        # neu la lop fully connectedd
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        # import pdb; pdb.set_trace()
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(
                torch.bmm(adj, torch.squeeze(adj, torch.squeeze(seq_fts, 0)))
            )
        else:
            out = torch.bmm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        return self.act(out)



class GCN_2_layers(torch.nn.Module):
    def __init__(self, hid_ft1, hid_ft2, out_ft, act='relu') -> None:
        super(GCN_2_layers, self).__init__()
        self.gcn_1 = GCN(hid_ft1, hid_ft2, act)
        self.gcn_2 = GCN(hid_ft2, out_ft, act)
        self.relu = nn.ReLU()
    
    def forward(self, x, adj, sparse=False):
        
        x = self.gcn_1(x, adj)
        x  = self.gcn_2(x, adj)
        return x 
    
class TemporalGCN(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, batch_size: int=1, improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True):
        super(TemporalGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        # self.config = config
        # breakpoint()
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCN_2_layers(hid_ft1=self.in_channels, hid_ft2=self.hidden_dim, out_ft=self.out_channels )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(self.batch_size,X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, adj, H):
        h = self.conv_z(X, adj)
        Z = torch.cat([h, H], axis=2) # (b, 207, 64)
        Z = self.linear_z(Z) # (b, 207, 32)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, adj, H):
        conv = self.conv_r(X, adj)
        R = torch.cat([conv, H], axis=2) # (b, 207, 64)
        R = self.linear_r(R) # (b, 207, 32)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, adj, H, R):
        H_tilde = torch.cat([self.conv_h(X, adj), H * R], axis=2) # (b, 207, 64)
        H_tilde = self.linear_h(H_tilde) # (b, 207, 32)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde   # # (b, 207, 32)
        return H

    def forward(self,X: torch.FloatTensor, adj: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, adj, H)
        R = self._calculate_reset_gate(X, adj, H)
        H_tilde = self._calculate_candidate_state(X, adj, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H
    
class TGCN_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(TGCN_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn_gcn = TemporalGCN(hid_ft1, out_ft, hid_ft2)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.fc(x)) # [12, 19, 200] 
        raw_shape = x.shape
        h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device=torch.device('cuda')) #(1, 19, 400)
        # breakpoint()
        list_h = []
        for i in range(raw_shape[1]):
            x_i = x[:,i,:,:].squeeze(1) # 1, 19, 200 
            e = adj[:,i,:,:].squeeze(1) # 1, 19, 19
            h = self.rnn_gcn(x_i, e, h)
            list_h.append(h)
        h_ = torch.stack(list_h, dim=1)
        return h_
    
class Attention_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Attention_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn_gcn = TemporalGCN(hid_ft1, out_ft, hid_ft2)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # breakpoint()
        x = self.relu(self.fc(x)) # 12, 19, 200 
        raw_shape = x.shape
        h = torch.zeros(raw_shape[0],raw_shape[2], self.out_dim, device=torch.device('cuda')) #(1, 19, 400)
        list_h = []
        for i in range(raw_shape[1]):
            x_i = x[:,i,:,:].squeeze(1) # 1, 19, 200 
            e = adj[:,i,:,:].squeeze(1) # 1, 19, 19
            h = self.rnn_gcn(x_i, e, h)
            list_h.append(h)
        h_ = torch.stack(list_h, dim=1)
        return h_
    

class Attention_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid, act_en="relu", stdgi_noise_min=0.4, stdgi_noise_max=0.7,num_input_station= 0):
        super(Attention_STDGI, self).__init__()
        print("Init Attention_Encoder model ...")
        self.encoder = Attention_Encoder(
            in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft, act=act_en
        )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max 

    def forward(self, x, x_k, adj):
        """_summary_

        Args:
            x (_type_): [32, 12, 4, 7]
            x_k (_type_): [32, 12, 4, 7]
            adj (_type_): [32, 12, 4, 4]

        Returns:
            _type_: _description_
        """
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        ret = self.disc(h[:,-1,:,:], x_k[:,-1,:,:], x_c[:,-1,:,:])
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts
        
    def embedd(self, x, adj):
        h = self.encoder(x, adj)
        return h