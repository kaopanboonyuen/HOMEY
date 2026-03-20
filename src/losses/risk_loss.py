import torch
import torch.nn as nn

class RiskAwareLoss(nn.Module):
    """
    Weighted BCE loss with severity-aware weighting
    """

    def __init__(self, class_weights=None, severity_weights=None):
        super(RiskAwareLoss, self).__init__()
        self.class_weights = class_weights
        self.severity_weights = severity_weights
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        """
        logits: [B, num_classes]
        targets: [B, num_classes]
        """
        loss = self.bce(logits, targets.float())
        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, -1)
        if self.severity_weights is not None:
            loss = loss * self.severity_weights.view(1, -1)
        return loss.mean()