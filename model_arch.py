import torch
import torch.nn as nn
import timm

# ----------------------------------------
# TASK 2: MIL MODEL (Patient Level)
# ----------------------------------------
class LIMUCSytem(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.25):
        super().__init__()
        # Matches notebook: convnext_tiny, pretrained=True, num_classes=0
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, global_pool='avg')
        
        # Attention Mechanism
        self.attention_V = nn.Sequential(nn.Linear(768, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(768, 256), nn.Sigmoid())
        self.attention_w = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.mil_classifier = nn.Linear(768, num_classes)

    def forward_patient(self, x):
        # x shape: [Batch_Size (Num_Images), 3, 224, 224]
        feats = self.dropout(self.backbone(x)) # [N, 768]
        
        # MIL Attention
        A_V = self.attention_V(feats)
        A_U = self.attention_U(feats)
        # Calculate weights
        weights = torch.softmax(self.attention_w(A_V * A_U).transpose(1, 0), dim=1)
        
        # Aggregate features
        M = torch.mm(weights, feats)
        
        # Final Classification
        return self.mil_classifier(M), weights

# ----------------------------------------
# TASK 3: REGRESSOR MODEL (Image Level)
# ----------------------------------------
class MayoRegressor(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super().__init__()
        # Independent backbone for regressor to avoid weight conflicts
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, global_pool='avg')
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate), 
            nn.Linear(768, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # Matches notebook: Sigmoid * 3.0 to scale output 0-3
        return torch.sigmoid(self.head(self.backbone(x))) * 3.0