import torch
import torch.nn as nn

class Combine1Loss(nn.Module):
    def __init__(self,gcn_encoder, feature_encoder, temporal_encoder, decoder):
        super().__init__()
        self.gnn_encoder = gcn_encoder
        self.feature_encoder = feature_encoder
        self.temporal_encoder = temporal_encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
    
    def get_loss(self, data):
        x, G, l, x_satellite, lat_index, lon_index, y = data
        y_pred = self.forward(data)
        
        return self.mse_loss(y_pred, y)
    
    def get_satellite_representation(self, x_satellite, lat_index, lon_index):
        satellite_embed = self.feature_encoder(x_satellite)
        satellite_target_embed = satellite_embed[:, :, lat_index[0], lon_index[0]]
        return satellite_target_embed
    
    def get_idw_representation(self,x,G,l):
        gnn_embed = self.gnn_encoder(x,G[:,:,:,:,0])[:,-1,:,:] ## [32, 12, 4, 64]
        idw_embed = torch.bmm(l.unsqueeze(1),gnn_embed)
        return idw_embed
    
    def forward(self, data):
        x, G, l, x_satellite, lat_index, lon_index, y = data
        # satellite_representation = self.get_satellite_representation(x_satellite, lat_index, lon_index)
        # x_satellite [batch_size, timestep, feature, width, height]
        gnn_representation = self.get_idw_representation(x,G,l)
        # Temporal attention
        # satellite_representation = self.temporal_encoder(x_satellite[:, :, 2:, lat_index[0], lon_index[0]].to(torch.float32), gnn_representation).squeeze() # [batch_size, timestep, hidden]
        # Feature attention
        satellite_representation = self.feature_encoder(x_satellite[:, :, 2:, lat_index[0], lon_index[0]].to(torch.float32), gnn_representation).squeeze() # [batch_size, timestep, hidden]
        # import pdb;pdb.set_trace()
        embedded_y = torch.cat((satellite_representation, gnn_representation.squeeze()), dim=1) # (batch_size, 128)
        # GNN only
        # y_pred = self.decoder(gnn_representation)
        # GNN only
        # return y_pred.reshape(y.shape)
        # GNN station data + satellite data
        y_pred = self.decoder(embedded_y)
        # GNN station data + satellite data
        return y_pred