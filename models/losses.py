## Cox loss as defined in Verghese MINOTAUR project (not published yet)

import torch

class CoxLoss(torch.nn.Module):
    """Computes the negative partial log-likelihood for the Cox proportional hazards model"""
    
    def __init__(self):
        super(CoxLoss, self).__init__()
    
    def forward(self, survtime, censor, hazard_pred, device=None):
        """
        Computes the negative partial log-likelihood for the Cox proportional hazards model.
        
        Args:
            survtime: Tensor of survival times (batch_size,).
            censor: Tensor of censor indicators (batch_size,) - 1 if event observed, 0 if censored.
            hazard_pred: Predicted hazard values from the model (batch_size,).
            device: Optional torch device. If None, inferred from hazard_pred.
        
        Returns:
            loss_cox: The computed Cox loss.
        """
        # Ensure inputs are tensors on the correct device
        if device is None:
            device = hazard_pred.device
        
        if not isinstance(survtime, torch.Tensor):
            survtime = torch.tensor(survtime, device=device, dtype=torch.float32)
        else:
            survtime = survtime.to(device).float()
            
        if not isinstance(censor, torch.Tensor):
            censor = torch.tensor(censor, device=device, dtype=torch.float32)
        else:
            censor = censor.to(device).float()
        
        # Risk set matrix: R_mat[i, j] = 1 if survtime[j] >= survtime[i]
        R_mat = (survtime[None, :] >= survtime[:, None]).float()
        
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        
        # Compute the log-risk for each sample
        log_risk = torch.log(torch.sum(exp_theta * R_mat, dim=1) + 1e-8)  # Add small epsilon for numerical stability
        
        # Compute the negative partial log-likelihood loss
        loss_cox = -torch.mean((theta - log_risk) * censor)
        return loss_cox