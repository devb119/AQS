import torch
import torch.nn as nn 
from tqdm import tqdm 

class Decoder(nn.Module):
    def __init__(
        self,
        in_ft,
        out_ft=1,
        fc_hid_dim=64,
        cnn_hid_dim = 64
    ):
        super(Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.fc = nn.Linear(in_ft, cnn_hid_dim)
        self.fc2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        """
        x.shape = batch_size x x (n) x num_ft
        """

        ret = self.fc(x)
        ret = self.relu(ret)  # (128,64)
        ret = self.fc2(ret)  # (64,1) 
        return ret
    

class CombineModel(nn.Module):
    def __init__(self,gcn_encoder, cnn_encoder, decoder):
        super().__init__()
        self.gnn_encoder = gcn_encoder
        self.cnn_encoder = cnn_encoder
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def get_l1_loss(self, y_pred, y):
        reshaped_y_pred = torch.reshape(y_pred, y.shape)
        return self.mse_loss(reshaped_y_pred, y)
    
    def get_l2_loss(self, y_pred, y):
        # reshaped_y_pred = torch.squeeze(y_pred, dim=1) if len(y_pred.shape) == 3 else y_pred
        reshaped_y_pred = torch.reshape(y_pred, y.shape)
        return self.mse_loss(reshaped_y_pred, y)
    
    def get_l3_loss(self, gnn_embed, cnn_embed):
        # return self.mse_loss(gnn_embed.squeeze(), cnn_embed.squeeze())
        return self.cosine_loss(gnn_embed.squeeze(), cnn_embed.squeeze(), torch.tensor([1]*gnn_embed.shape[0]).to("cuda"))
    
    def get_combined_loss(self, weights, data):
        x, G, l, cnn_x, lat_index, lon_index, y = data 
        
        cnn_representation = self.get_cnn_representation(cnn_x, lat_index, lon_index)
        gnn_representation = self.get_idw_representation(x,G,l)
        
        gnn_prediction = self.get_predicted_gnn(gnn_representation)
        cnn_prediction = self.get_predicted_cnn(cnn_representation)
        
        alpha, beta, gamma = weights
        loss1 = self.get_l1_loss(cnn_prediction, y)
        loss2 = self.get_l2_loss(gnn_prediction, y)
        loss3 = self.get_l3_loss(gnn_representation, cnn_representation)
        
        combined_loss = alpha * loss1 + beta * loss2 + gamma * loss3
        
        return {"loss1": loss1,
                "loss2": loss2,
                "loss3": loss3,
                "combined_loss": combined_loss}
    
    def get_cnn_representation(self,cnn_x, lat_index, lon_index):
        cnn_embed = self.cnn_encoder(cnn_x)
        cnn_target_embed = cnn_embed[:, :, lat_index[0], lon_index[0]]
        # list_ft = []
        # for batch_id in range(cnn_x.shape[0]):
        #     lat, lon = lat_index[batch_id], lon_index[batch_id]
        #     ft = cnn_embed[batch_id,:, lon, lat]
        #     list_ft.append(ft) 
        # cnn_target_embed = torch.stack(list_ft,0)
        return cnn_target_embed
    
    def get_idw_representation(self,x,G,l):
        gnn_embed = self.gnn_encoder(x,G[:,:,:,:,0])[:,-1,:,:] ## [32, 12, 4, 64]
        idw_embed = torch.bmm(l.unsqueeze(1),gnn_embed)
        return idw_embed
    
    def test(self,cnn_x, lat, lon):
        cnn_embed = self.get_cnn_representation(cnn_x, lat, lon)
        return self.get_predicted_cnn(cnn_embed)
    
    def get_predicted_gnn(self,idw_embed):
        output = self.decoder(idw_embed)
        return output
    
    def get_predicted_cnn(self, cnn_target_embed):
        output = self.decoder(cnn_target_embed)
        return output
        