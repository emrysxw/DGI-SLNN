import torch
import torch.nn as nn
import torch.nn.functional as F

class Spi(nn.Module):
    def __init__(self, img_size = 128, img_pixels = 16384, base_nums = 1024):
        super(Spi, self).__init__()
        self.img_size = img_size
        self.base_nums = base_nums
        self.img_pixels = img_pixels
        self.params = nn.Parameter(torch.randn(img_pixels, base_nums))

    def forward(self, x):
        
        x = torch.reshape(x, [-1, self.img_pixels])
        i = torch.matmul(x, self.params)

        # out = 1/self.base_nums * torch.matmul(i, self.params.t()) - \
        # torch.mean(torch.sum(self.params,0).reshape(1,self.base_nums) * i,1).reshape(-1,1) @ \
        # torch.mean(self.params, 1).reshape(1,self.img_pixels) / torch.mean(torch.sum(self.params,0)) 
        # output = torch.reshape(out, [-1, self.img_size, self.img_size])

        out = (1 / self.base_nums) * torch.matmul(i, self.params.t()) - \
              (torch.mean(i, 1, keepdim=True) / torch.mean(torch.sum(self.params, 0))) @ \
              ((1 / self.base_nums) * torch.matmul(torch.sum(self.params, 0).unsqueeze(0), self.params.t()))

        output = torch.reshape(out, [-1, self.img_size, self.img_size])

        return output

class CovarianceReg_Loss(nn.Module):
    def __init__(self, reg, loss_fn):
        super(CovarianceReg_Loss, self).__init__()
        self.regularizer = reg
        self.loss_fn = loss_fn

    def forward(self, x, y, net):
        mse = self.loss_fn(x, y)
        params = next(net.parameters())
        covariance_matrix = torch.matmul(params, params.t())
        diagonal = torch.diagonal(covariance_matrix, 0)
        off_diagonal = (torch.sum(torch.abs(covariance_matrix)) - torch.sum(torch.abs(diagonal))) / (params.size(0) * (params.size(0) - 1))
        diagonal_var = torch.var(diagonal)
        totloss = self.regularizer * (0.1 * diagonal_var + off_diagonal) + mse
        return totloss