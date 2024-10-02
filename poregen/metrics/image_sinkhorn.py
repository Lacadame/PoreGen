import torch
import os
import pathlib
import numpy as np

MPATH = pathlib.Path(os.getcwd()).resolve()


class SinkhornLoss(torch.nn.Module):
    def __init__(self, lambda_param=0.5, weight=10**7, size_average=False,
                 max_iter=100, device='cuda'):
        super(SinkhornLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.device = device
        self.M = self.load_or_create_M().half()
        self.lambda_param = lambda_param
        self.K = torch.exp(-self.M / self.lambda_param).to(device).half()
        self.max_iter = max_iter
        
    def forward(self, input, target, additional_data=None):
        mse_loss = torch.mean((input - target) ** 2)
        # Apply sinkhorn divergence to each image and channel separately
        batch_size, channels, _, _ = input.shape
        sh_total = 0
        sh_total = self.image_sinkhorn_divergence_vectorized(input, target)

        total_loss = mse_loss + self.weight * sh_total
        return total_loss if self.size_average else total_loss / batch_size

    def image_sinkhorn_divergence_vectorized(self, R, C, epsilon=1e-9):
        # Reshape to treat each channel as a separate batch entry
        batch_size, channels, height, width = R.shape
        R = R.reshape(batch_size * channels, -1).to(self.device).half()
        C = C.reshape(batch_size * channels, -1).to(self.device).half()

        r = (R + torch.abs(torch.min(R, dim=1, keepdim=True)[0])).half()
        c = (C + torch.abs(torch.min(C, dim=1, keepdim=True)[0])).half()

        u = torch.ones_like(r, device=self.device, dtype=torch.float16)
        v = torch.ones_like(c, device=self.device, dtype=torch.float16)

        # Ensure K and M are appropriately expanded and converted to float16
        self.K = self.K.expand(batch_size * channels, -1, -1).half()
        self.M = self.M.expand(batch_size * channels, -1, -1).half()

        for _ in range(self.max_iter):
            v = (c / (self.K.transpose(1, 2) @ u.unsqueeze(-1) + 0.001).squeeze(-1)).half()
            u = (r / (self.K @ v.unsqueeze(-1) + 0.001).squeeze(-1)).half()

            if torch.max(torch.norm(torch.bmm(torch.diag_embed(u), self.K @ v.unsqueeze(-1)) - r.unsqueeze(-1), dim=1, p=1)) < epsilon:
                break
        result = (torch.sum(torch.bmm(torch.diag_embed(u), self.K @ torch.diag_embed(v)) * self.M, dim=(1, 2))).view(batch_size, channels).sum(dim=1)
        del  R, C, r, c, u, v
        torch.cuda.empty_cache()
        return result
    
    def initialize(self):
        self.M = self.load_or_create_M().half()
        self.K = torch.exp(-self.M / self.lambda_param).to(self.device).half()

    def load_or_create_M(self):
        filename = 'M.pt'
        if os.path.exists(filename):
            return torch.load(filename).to(self.device)
        else:
            M = self.create_distance_matrix(128, 128)  # Supõe uma função que cria a matriz
            torch.save(M, filename)
            return M
    def reset(self, lambda_param=None, weight=None):
        if lambda_param is not None:
            self.lambda_param = lambda_param
        if weight is not None:
            self.weight = weight
        if hasattr(self, 'K'):
            del self.K
        if hasattr(self, 'M'):
            del self.M
        torch.cuda.empty_cache()

    
    def image_sinkhorn_divergence(self, R, C, epsilon=1e-9, max_iter=100):
        """
        Function that implements the Sinkhorn-Knopp algorithm to compute the Sinkhorn divergence.
        """

        r = R.flatten() + torch.abs(torch.min( R.flatten()))
        c = C.flatten() + torch.abs(torch.min(C.flatten()))

        r, c = r.to(self.device), c.to(self.device)
        u = torch.ones_like(r, device=self.device)
        v = torch.ones_like(c, device=self.device)

        for _ in range(max_iter):
            v = c / (self.K.t() @ u + 0.001)
            u = r / (self.K @ v + 0.001)

            if torch.norm(torch.diag(u) @ (self.K @ v) - r, p=1) < epsilon:
                break
        del r
        del c
        P_lambda = torch.diag(u) @ self.K @ torch.diag(v)
        divergence = torch.sum(P_lambda * self.M.to(self.device))
        del u
        del v
        return divergence

    def create_distance_matrix(self, m, n):
        total_points = m * n
        coordinates = [(i, j) for i in range(m) for j in range(n)]
        distance_matrix = np.zeros((total_points, total_points))

        for i in range(total_points):
            for j in range(total_points):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance_matrix[i, j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        return torch.tensor(distance_matrix / np.sum(distance_matrix), dtype=torch.float16)
    
    """ TODO: otimizar o tamaho do K e M usando triu e escrever M em função de K
    usar float 16
    verificar como o stepcheckpoint funciona
    """

    