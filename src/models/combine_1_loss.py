import torch
import torch.nn as nn

class Combine1Loss(nn.Module):
    # def __init__(self,gcn_encoder, srcnn, feature_encoder, temporal_encoder, decoder, satellite_handler):
    def __init__(self,gcn_encoder, feature_encoder, temporal_encoder, decoder, satellite_handler):
        super().__init__()
        self.gnn_encoder = gcn_encoder
        self.feature_encoder = feature_encoder
        self.temporal_encoder = temporal_encoder
        self.decoder = decoder
        # self.srcnn = srcnn
        self.satellite_handler = satellite_handler
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
        gnn_representation = self.get_idw_representation(x,G,l)
        # for i in range(12):
        #     x_satellite[:, i, 2:, :, :] = self.srcnn(x_satellite[:, i, 2:, :, :].to(torch.float32))
        
        # Temporal attention
        if self.satellite_handler == "temporal_att":
        # x_satellite [batch_size, timestep, feature, width, height]
            satellite_representation_t = self.temporal_encoder(x_satellite[:, :, 2:, lat_index[0], lon_index[0]].to(torch.float32), x_satellite[:, -1,2:, lat_index[0], lon_index[0]].unsqueeze(dim=1).to(torch.float32)).squeeze() # [batch_size, timestep, hidden]
            embedded_y = torch.cat((satellite_representation_t, gnn_representation.squeeze()), dim=1) # (batch_size, 128)
        # Feature attention
        elif self.satellite_handler == "feature_att":
            satellite_representation_f = self.feature_encoder(x_satellite[:, :, 2:, lat_index[0], lon_index[0]].to(torch.float32), 
                                                                x_satellite[:, -1,2:, lat_index[0], lon_index[0]].unsqueeze(dim=1).to(torch.float32)).squeeze() # [batch_size, timestep, hidden]
            embedded_y = torch.cat((satellite_representation_f, gnn_representation.squeeze()), dim=1) # (batch_size, 128)
        elif self.satellite_handler == "concat":
            satellite_representation_t = self.temporal_encoder(x_satellite[:, :, 2:, lat_index[0], lon_index[0]].to(torch.float32), x_satellite[:, -1,2:, lat_index[0], lon_index[0]].unsqueeze(dim=1).to(torch.float32)).squeeze() # [batch_size, timestep, hidden]
            satellite_representation_f = self.feature_encoder(x_satellite[:, :, 2:, lat_index[0], lon_index[0]].to(torch.float32), 
                                                                x_satellite[:, -1,2:, lat_index[0], lon_index[0]].unsqueeze(dim=1).to(torch.float32)).squeeze() # [batch_size, timestep, hidden]
            satellite_representation = torch.concat((satellite_representation_t, satellite_representation_f), dim=1)
            embedded_y = torch.cat((satellite_representation, gnn_representation.squeeze()), dim=1) # (batch_size, 128)
        elif self.satellite_handler == "gnn":
            embedded_y = gnn_representation.squeeze()
        # GNN only
        y_pred = self.decoder(embedded_y)
        return y_pred