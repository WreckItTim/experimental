import torch
import torch.nn.functional as F
#from torchsummary import summary
from torch import nn

# from modules import DilatedResidualBlock, NLB, DGNL, DepthWiseDilatedResidualBlock

# from torchvision.models import resnet18, ResNet18_Weights


# # This is the new model
# class DGNLNet(nn.Module):
#     def __init__(self, bottleneck_channel=128):
#         super(DGNLNet, self).__init__()
#         self.mean = torch.zeros(1, 3, 1, 1)
#         self.std = torch.zeros(1, 3, 1, 1)
#         self.mean[0, 0, 0, 0] = 0.485
#         self.mean[0, 1, 0, 0] = 0.456
#         self.mean[0, 2, 0, 0] = 0.406
#         self.std[0, 0, 0, 0] = 0.229
#         self.std[0, 1, 0, 0] = 0.224
#         self.std[0, 2, 0, 0] = 0.225

#         self.mean = nn.Parameter(self.mean)
#         self.std = nn.Parameter(self.std)
#         self.mean.requires_grad = False
#         self.std.requires_grad = False

#         self.bottle_fit = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=64),
# 			nn.SELU(inplace=True),
            
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
# 			nn.GroupNorm(num_groups=32, num_channels=64),
# 			nn.SELU(inplace=True),
            
# 			nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=512),
# 			nn.SELU(inplace=True),
            
# 			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
# 			nn.GroupNorm(num_groups=32, num_channels=512),
# 			nn.SELU(inplace=True),
            
# 			nn.Conv2d(512, bottleneck_channel, kernel_size=3, stride=2, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=bottleneck_channel, num_channels=bottleneck_channel),
# 			nn.SELU(inplace=True),
            
# 			# split point
            
# 			nn.ConvTranspose2d(bottleneck_channel, 512, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=512),
# 			nn.SELU(inplace=True),
            
#             nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=256),
# 			nn.SELU(inplace=True),
            
# 			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=128),
# 			nn.SELU(inplace=True),
            
# 			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=64),
# 			nn.SELU(inplace=True),
            
# 			nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
# 			nn.GroupNorm(num_groups=32, num_channels=32),
# 			nn.SELU(inplace=True),
# 			nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
# 			nn.SELU(inplace=True),
# 			nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False),
# 			nn.Sigmoid()
# 		)
        
#         for m in self.modules():
#             if isinstance(m, nn.ReLU):
#                 m.inplace = True

#     def forward(self, x):
#         x = (x - self.mean) / self.std
#         depth_pred = self.bottle_fit(x)        
#         return depth_pred



class DGNLNet_v1(nn.Module):
	def __init__(self, num_features=64):
		super(DGNLNet_v1, self).__init__()
		self.mean = torch.zeros(1, 3, 1, 1)
		self.std = torch.zeros(1, 3, 1, 1)
		self.mean[0, 0, 0, 0] = 0.485
		self.mean[0, 1, 0, 0] = 0.456
		self.mean[0, 2, 0, 0] = 0.406
		self.std[0, 0, 0, 0] = 0.229
		self.std[0, 1, 0, 0] = 0.224
		self.std[0, 2, 0, 0] = 0.225

		self.mean = nn.Parameter(self.mean)
		self.std = nn.Parameter(self.std)
		self.mean.requires_grad = False
		self.std.requires_grad = False

		############################################ Depth prediction network
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 256, 4, stride=2, padding=1), 
			nn.GroupNorm(num_groups=32, num_channels=256), 
			nn.SELU(inplace=True)
		)

		self.conv5 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=4, dilation=4),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

		self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

		self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

		self.conv10 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

		self.depth_pred = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)
		for m in self.modules():
			if isinstance(m, nn.ReLU):
				m.inplace = True

	def forward(self, x):
		x = (x - self.mean) / self.std

		#print(x.shape)
		d_f1 = self.conv1(x)
		#print(d_f1.shape)
		d_f2 = self.conv2(d_f1)
		#print(d_f2.shape)
		d_f3 = self.conv3(d_f2)
		#print(d_f3.shape)
		d_f4 = self.conv4(d_f3)
		#print(d_f4.shape)
		self.d_f5 = self.conv5(d_f4)
		# print(self.d_f5.shape)
		self.d_f6 = self.conv6(self.d_f5)
		#print(d_f6.shape)
		self.d_f7 = self.conv7(self.d_f6)
		#print(d_f7.shape)
		self.d_f8 = self.conv8(self.d_f7)
		#print(d_f8.shape)
		self.d_f9 = self.conv9(self.d_f8)
		#print(d_f9.shape)
		self.d_f10 = self.conv10(self.d_f9)
		#print(d_f10.shape)
		self.depth_pred_out = self.depth_pred(self.d_f10)
		#print(depth_pred.shape)

		return self.depth_pred_out


class DGNLNet_v1_Head_Split(nn.Module):
	def __init__(self, num_features=64, bn_channel=12):
		super(DGNLNet_v1_Head_Split, self).__init__()
		self.mean = torch.zeros(1, 3, 1, 1)
		self.std = torch.zeros(1, 3, 1, 1)
		self.mean[0, 0, 0, 0] = 0.485
		self.mean[0, 1, 0, 0] = 0.456
		self.mean[0, 2, 0, 0] = 0.406
		self.std[0, 0, 0, 0] = 0.229
		self.std[0, 1, 0, 0] = 0.224
		self.std[0, 2, 0, 0] = 0.225

		self.mean = nn.Parameter(self.mean)
		self.std = nn.Parameter(self.std)
		self.mean.requires_grad = False
		self.std.requires_grad = False

		############################################ Depth prediction network
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 4, stride=3),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=3),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(64, bn_channel, 4, stride=3),
			nn.GroupNorm(num_groups=bn_channel, num_channels=bn_channel),
			nn.SELU(inplace=True)
		)

		# #this is for testing to get to orginal shape needed
		# self.convT = nn.Sequential(
		# 	nn.ConvTranspose2d(bn_channel, 64, kernel_size=4, stride=2, padding=(2,0)),
		# 	nn.GroupNorm(num_groups=32, num_channels=64),
		# 	nn.SELU(inplace=True)
		# )

		# self.convT1 = nn.Sequential(
		# 	nn.Conv2d(64, 128, 3, padding=2, dilation=2),
		# 	nn.GroupNorm(num_groups=64, num_channels=128),
		# 	nn.SELU(inplace=True)
		# )

		for m in self.modules():
			if isinstance(m, nn.ReLU):
				m.inplace = True

	def forward(self, x):
		x = (x - self.mean) / self.std

		#print(x.shape)
		d_f1 = self.conv1(x)
		#print(d_f1.shape)
		d_f2 = self.conv2(d_f1)
		#print(d_f2.shape)
		d_f3 = self.conv3(d_f2)
		# print(d_f3.shape)
		# d_T = self.convT(d_f3)
		# print(d_T.shape)
		# d_T1 = self.convT1(d_T[:,:,:(d_T.shape[2]-1),:(d_T.shape[3]-1)])
		# print(d_T1.shape)

		#d_f3 = torch.quantize_per_tensor(d_f3, scale=1.094, zero_point=12, dtype=torch.quint8)

		return d_f3

class DGNLNet_v1_Tail_Split(nn.Module):
	def __init__(self, num_features=64, bn_channel=12):
		super(DGNLNet_v1_Tail_Split, self).__init__()
		self.mean = torch.zeros(1, 3, 1, 1)
		self.std = torch.zeros(1, 3, 1, 1)
		self.mean[0, 0, 0, 0] = 0.485
		self.mean[0, 1, 0, 0] = 0.456
		self.mean[0, 2, 0, 0] = 0.406
		self.std[0, 0, 0, 0] = 0.229
		self.std[0, 1, 0, 0] = 0.224
		self.std[0, 2, 0, 0] = 0.225

		self.mean = nn.Parameter(self.mean)
		self.std = nn.Parameter(self.std)
		self.mean.requires_grad = False
		self.std.requires_grad = False

		############################################ Depth prediction network
		self.decoder_pre = nn.Sequential(
			nn.ConvTranspose2d(bn_channel, 64, kernel_size=4, stride=2, padding=(2,0)),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)
		
		self.decoder = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=64, num_channels=128),
			nn.SELU(inplace=True),

			nn.Conv2d(128, 256, 3, padding=2, dilation=2), 
			nn.GroupNorm(num_groups=32, num_channels=256), 
			nn.SELU(inplace=True),

			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		) #conv5 out is target

		#end decoder

		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=4, dilation=4),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

		self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

		self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

		self.conv10 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

		self.depth_pred = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)
		for m in self.modules():
			if isinstance(m, nn.ReLU):
				m.inplace = True

	def forward(self, x):

		#x = torch.dequantize(x)

		x = self.decoder_pre(x)
		# print(x.shape)
		self.d_f5 = self.decoder(x[:,:,:,:(x.shape[3]-1)])
          
		# print(self.d_f5.shape)
		self.d_f6 = self.conv6(self.d_f5)
		self.d_f7 = self.conv7(self.d_f6)
		self.d_f8 = self.conv8(self.d_f7)
		self.d_f9 = self.conv9(self.d_f8)
		self.d_f10 = self.conv10(self.d_f9)
		self.depth_pred_out = self.depth_pred(self.d_f10)

		return self.depth_pred_out