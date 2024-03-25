import torch
import torch.nn as nn

class Combine1Loss(nn.Module):
    def __init__(self,gcn_encoder, cnn_encoder, decoder):
        super().__init__()
        self.gnn_encoder = gcn_encoder
        self.cnn_encoder = cnn_encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
    
    def get_loss(self, data):
        x, G, l, cnn_x, lat_index, lon_index, y = data
        y_pred = self.forward(data)
        
        return self.mse_loss(y_pred, y)
    
    def get_cnn_representation(self, cnn_x, lat_index, lon_index):
        cnn_embed = self.cnn_encoder(cnn_x)
        cnn_target_embed = cnn_embed[:, :, lat_index[0], lon_index[0]]
        return cnn_target_embed
    
    def get_idw_representation(self,x,G,l):
        gnn_embed = self.gnn_encoder(x,G[:,:,:,:,0])[:,-1,:,:] ## [32, 12, 4, 64]
        idw_embed = torch.bmm(l.unsqueeze(1),gnn_embed)
        return idw_embed
    
    def forward(self, data):
        x, G, l, cnn_x, lat_index, lon_index, y = data
        # cnn_representation = self.get_cnn_representation(cnn_x, lat_index, lon_index)
        # import pdb; pdb.set_trace()
        cnn_representation = self.cnn_encoder(cnn_x[:, 2:, lat_index[0], lon_index[0]].to(torch.float32))
        gnn_representation = self.get_idw_representation(x,G,l)
        # import pdb; pdb.set_trace()
        embedded_y = torch.cat((cnn_representation, gnn_representation.squeeze()), dim=1) # (batch_size, 128)
        # GNN only
        y_pred = self.decoder(gnn_representation)
        # GNN station data + satellite data
        # y_pred = self.decoder(embedded_y)
        # GNN only
        return y_pred.reshape(y.shape)
        # GNN station data + satellite data
        # return y_pred