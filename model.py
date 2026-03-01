import torch.nn as nn
import torch
import timm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 64, dropout = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)
    
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x = (B, T, D)
        attn_weights = self.attention(x) # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim = 1) # (B, T, 1)

        pooled = torch.sum(attn_weights * x, dim = 1) # (B, D)

        return pooled

class VitBackbone(nn.Module):
    def __init__(self, model_name = 'vit_base_patch16_224', pretrained = True):
        super().__init__() 

        self.vit = timm.create_model(model_name, pretrained = pretrained, num_classes = 0, global_pool = 'avg')

        self.out_dim = self.vit.num_features

    def forward(self, x):
        # x = (B * T, C, H, W)
        return self.vit(x)


class VitTransformer(nn.Module):
    # Input = (B, T, C, H, W) = (B, 16, 3, 244, 244)
    # Output = (B, num_classes)

    def __init__(self, num_classes = 100):
        super().__init__()

        # ViT backbone 
        self.vitbackbone = VitBackbone()
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model = self.vitbackbone.out_dim,
            max_len = 64,
            dropout = 0.1
        )

        # Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.vitbackbone.out_dim,
            nhead = 8,
            dim_feedforward = self.vitbackbone.out_dim * 4,
            dropout = 0.3,
            activation = 'gelu',
            batch_first = True,
            norm_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = 4)

        # Attention Pooling
        self.attn_pool = AttentionPooling(self.vitbackbone.out_dim)

        # Classifier
        self.fc = nn.Sequential(
            nn.LayerNorm(self.vitbackbone.out_dim),
            nn.Dropout(0.4),
            nn.Linear(self.vitbackbone.out_dim, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  

        for m in self.attn_pool.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)     
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # (B * T, C, H, W)
        features = self.vitbackbone(x) # (B * T, D)
        features = features.view(B, T, -1) # (B, T, D)
        
        # Transformer: (B, T, D) -> (B, T, D)
        features = self.pos_encoder(features) # (B, T, D)
        features = self.transformer(features) # (B, T, D)

        # Attention Pooling: (B, T, D) -> (B, D)
        features = self.attn_pool(features) # (B, D)

        # Classifier: (B, D) -> (B, num_classes)
        output = self.fc(features) # (B, num_classes)

        return output