from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch as th
import numpy as np

# CUSTOM SLIM LAYERS

class SlimMLP(nn.Linear):
    def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True,
                 device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_features, max_out_features, bias, device, dtype)
        self.max_in_features = self.in_features = max_in_features
        self.max_out_features = self.out_features = max_out_features
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_features = max(1, int(self.rho * self.max_in_features))
        if self.slim_out:
            self.out_features = max(1,int(self.rho * self.max_out_features))
        #print(f'B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        y = F.linear(input, weight, bias)
        #utils.speak(f'RHO:{self.rho} IN:{weight.shape} OUT:{y.shape}')
        return y
        
class SlimConv2d(nn.Conv2d):
    def __init__(self, max_in_channels: int, max_out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_channels, max_out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, device=device, dtype=dtype)
        self.max_in_channels = self.in_channels = max_in_channels
        self.max_out_channels = self.out_channels = max_out_channels
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_channels = max(1,int(self.rho * self.max_in_channels))
        if self.slim_out:
            self.out_channels = max(1,int(self.rho * self.max_out_channels))
        #print(f'conv2d B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_channels,:self.in_channels,:,:]
        #print(f'conv2d A4-shape:{weight.shape}')
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        #utils.speak(f'RHO:{self.rho} IN:{weight.shape} OUT:{y.shape}')
        return y
        
class SlimConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, max_in_channels: int, max_out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_channels, max_out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, device=device, dtype=dtype)
        self.max_in_channels = self.in_channels = max_in_channels
        self.max_out_channels = self.out_channels = max_out_channels
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor, output_size = None) -> Tensor:
        if self.slim_in:
            self.in_channels = max(1,int(self.rho * self.max_in_channels))
        if self.slim_out:
            self.out_channels = max(1,int(self.rho * self.max_out_channels))
        #print(f'trans2d B4-shape:{self.weight.shape}')
        weight = self.weight[:self.in_channels,:self.out_channels,:,:]
        #print(f'trans2d A4-shape:{weight.shape}')
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        num_spatial_dims = 2
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size,
        num_spatial_dims, self.dilation)
        y = F.conv_transpose2d(input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        #utils.speak(f'RHO:{self.rho} IN:{weight.shape} OUT:{y.shape}')
        return y
        
class SlimBatchNorm2d(nn.Module):
    def __init__(self, max_channels, rhos):
        super().__init__()
        self.max_channels = max_channels
        self.idx_map = {}
        bns = []
        for idx, rho in enumerate(rhos):
            self.idx_map[rho] = idx
            n_channels = max(1,int(max_channels*rho))
            bns.append(nn.BatchNorm2d(n_channels))
        self.bn = nn.ModuleList(bns)
        self.rho = 1
    def forward(self, input):
        idx = self.idx_map[self.rho]
        y = self.bn[idx](input)
        return y

def set_slim(model, rho):
    for module in model.modules():
        if 'slim' in str(type(module)):
            module.rho = rho

def forward_slim_val(model, DL, device, criterion, mem_optim=True,
                 rhos=None, low_rho=0.25, high_rho=0.75, nRhos=2 # rhos handle rhoming, values between (0,1)
                      ):
    losses = []
    for i, data in enumerate(DL):
        x, y = data
        x, y = x.to(device=device), y.to(device=device)
        with th.no_grad():
            p = model(x)
            loss = criterion(p, y)
            losses.append(float(loss.detach().cpu()))
        if rhos is None:
            rhos = np.random.uniform(low=low_rho, high=high_rho, size=nrhos)
            rho_samples = [low_rho] + list(sample_rho)
        for rho in rhos:
            set_slim(model, rho)
            with th.no_grad():
                p2 = model(x)
                loss = criterion(p2, p)
                losses.append(float(loss.detach().cpu()))
        set_slim(model, 1)
        if mem_optim:
            del x, y, p # clear mem from gpu
    return float(np.mean(losses))
            
def forward_slim_train(model, DL, device, criterion, mem_optim=True, 
                 rhos=None, low_rho=0.25, high_rho=0.75, nRhos=2, soft_targets=False, scale_rho=False
                      ):
    losses = []
    for i, data in enumerate(DL):
        # sample rhos using sandwich rule
        if rhos is None:
            rhos = np.random.uniform(low=low_rho, high=high_rho, size=nrhos)
            rho_samples = [low_rho] + list(sample_rho)
        x, y = data
        x, y = x.to(device=device), y.to(device=device)
        model.optimizer.zero_grad()
        if soft_targets:
            p = model(x)
            loss = criterion(p, y)
            loss.backward(retain_graph=True)
            losses.append(float(loss.detach().cpu()))
            for rho in rhos:
                if rho >= 1:
                    continue
                set_slim(model, rho)
                p2 = model(x)
                if scale_rho:
                    loss = rho * criterion(p2, p)
                else:
                    loss = criterion(p2, p)
                loss.backward(retain_graph=True)
                losses.append(float(loss.detach().cpu()))
        else:
            for rho in rhos:
                set_slim(model, rho)
                p = model(x)
                if scale_rho:
                    loss = rho * criterion(p, y)
                else:
                    loss = criterion(p, y)
                loss.backward(retain_graph=True)
                losses.append(float(loss.detach().cpu()))
        # update weights
        model.optimizer.step()
        # clean up
        if mem_optim: # clear mem from gpu
            if soft_targets:
                del x, y, p, p2
            else:
                del x, y, p
        set_slim(model, 1)
    return float(np.mean(losses))
    
def foward_slim_predictions(model, DL, device, mem_optim=True, DL_includes_y=False, 
                 rhos=[],  # rhos handle rhoming, values between (0,1)
                      ):
    P = {rho:[] for rho in rhos}
    for i, data in enumerate(DL):
        if DL_includes_y:
            x, y = data
        else:
            x = data
        x = x.to(device=device)
        with th.no_grad():
            for rho in rhos:
                set_slim(model, rho)
                p = model(x)
                P[rho].append(p.cpu().detach().numpy())
            set_slim(model, 1)
            if mem_optim:
                del p # clear mem from gpu
        if mem_optim:
            del x # clear mem from gpu
    return {rho:np.vstack(P[rho]) for rho in rhos}
    