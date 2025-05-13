import torch
import itertools
import numpy as np
import copy
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from models.raft import viz

from models.utils.utils import InputPadder
from models.loss.losses import DiceLoss, FocalLoss, CharbonnierLoss
from models.loss.STLoss import STLoss
from models.loss.boundary_loss import boundary_loss, compute_sdf


def preprocess(tensor_img):
    # 确保输入是一个PyTorch张量
    assert isinstance(tensor_img, torch.Tensor), "Input must be a PyTorch tensor"

    # 计算98百分位数并进行裁剪
    max98 = torch.percentile(tensor_img, 98, dim=(2, 3), keepdim=True)
    tensor_img = torch.clamp(tensor_img, min=0, max=max98)

    # 确保图像尺寸至少为512x512
    batch_size, channels, height, width = tensor_img.shape
    crop_size = 512  # 假设crop_size是512
    if width < crop_size:
        pad_width = (crop_size - width) // 2
        padding = (0, 0, pad_width, crop_size - width - pad_width)
        tensor_img = torch.nn.functional.pad(tensor_img, padding, mode='constant', value=0)
    if height < crop_size:
        pad_height = (crop_size - height) // 2
        padding = (pad_height, crop_size - height - pad_height, 0, 0)
        tensor_img = torch.nn.functional.pad(tensor_img, padding, mode='constant', value=0)

    # 归一化
    tensor_img = tensor_img / max98

    return tensor_img


class MyModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='STCnet', dataset_mode='Coronary_mask') # st
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_content', type=float, default=1)
            # DP and DDP loss depend on the data, we recommend [0.1, 1]
            parser.add_argument('--lambda_D', type=float, default=1.0, help='weight for D loss')
            parser.add_argument('--lambda_TP', type=float, default=0.1, help='weight for Dice loss')




        parser.add_argument('--flow_cpt', type=str, default='/media/tjubme/F508AD7CCCADB296/WGP/Annotation-free-Medical-Image-Enhancement-main/checkpoints/raft/raft_coronary.pth', help='the path of raft model')
        parser.add_argument('--n_feats', type=int, default=30, help='the number of STnet feats') #stnet is 32, and stcnet is 30
        parser.add_argument('--n_groups', type=int, default=4, help='the number of STnet feats')
        # parser.add_argument('--input_nc', type=int, default=1, help='the number of STnet feats')
        parser.add_argument('--flow_nc', type=int, default=1, help='the number of STnet feats')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.input_nc = opt.input_nc

        self.loss_names = ['D', 'G_L1', 'G', 'Dice', 'S', 'ST']

        # self.loss_names = [ 'G']

        # self.visual_names = ['real_SA1', 'real_SA','real_SA2', 'real_SA3', 'fake_SB', 'real_SB', 'S_mask', 'B_Mask', 'f_flow', 'b_flow']
        # self.visual_names = ['real_SA1', 'real_SA', 'real_SA2', 'real_SA3', 'fake_SB', 'real_SB']
        # self.visual_names = ['real_SA', 'F_Flow', 'B_Flow', 'real_SB', 'fake_SB', 'S_mask', 'B_Mask']
        # self.visual_names = ['real_SA1', 'real_SA2', 'real_SA3', 'fake_SB', 'real_SB', 'S_mask', 'B_Mask']

        # self.netG = networks.define_G(1, opt.flow_nc, opt.n_feats, opt.netG, gpu_ids=self.gpu_ids)
        self.netG = networks.define_G(1, opt.flow_nc, opt.n_feats, opt.n_groups, opt.netG, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D(1, opt.ndf, opt.netD, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        # self.netF = networks.define_F(3, opt.netF, opt.flow_cpt)


        # self.TPLoss =

        if self.isTrain:
            self.model_names = ['G', 'D']
            self.visual_names = ['real_SA', 'F_Flow', 'B_Flow', 'real_SB', 'fake_SB', 'S_mask', 'B_Mask']
            # self.visual_names = ['real_SA', 'F_Flow', 'B_Flow', 'real_SB', 'fake_SB', 'S_mask']
            self.criterionSegment = FocalLoss(2)
            self.criterionDice = DiceLoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.stloss = STLoss()
            self.boundary = boundary_loss
            # self.criterionL1 = torch.nn.MSELoss(reduction='sum')


            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        else:  # during test time, only load G
            self.model_names = ['G']
            # self.visual_names = ['real_SA', 'fake_SB']
            self.visual_names = ['real_SB', 'fake_SB']


    # def set_input(self, input):
    def set_input(self, input, f_flow, b_flow):
        """
        处理输入
        """
        if self.isTrain:
            AtoB = self.opt.direction == 'AtoB'
            # self.real_SA1 = input['SA1' if AtoB else 'SB'].to(self.device)
            # self.real_SA2 = input['SA2' if AtoB else 'SB'].to(self.device)
            # self.real_SA3 = input['SA3' if AtoB else 'SB'].to(self.device)
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)

            self.F_Flow = f_flow.to(self.device)
            self.B_Flow = b_flow.to(self.device)

            self.S_mask = input['S_mask'].to(self.device)
            self.image_paths = input['SA_path']

            # self.F_Flow = input['f_flow' if AtoB else 'SA'].to(self.device)
            # self.B_Flow = input['b_flow' if AtoB else 'SA'].to(self.device)
            # self.f_flow = viz(self.F_Flow.detach())
            # self.b_flow = viz(self.B_Flow.detach())

            # padder = InputPadder(self.real_SA1.shape)
            # sa1, sa2, sa3 = padder.pad(self.real_SA1, self.real_SA1, self.real_SA1)
            # self.netF.eval()
            # with torch.no_grad():
            #     _, self.F_Flow = self.netF(self.real_SA2, self.real_SA1, iters=20, test_mode=True)
            #     self.f_flow = viz(self.F_Flow.detach())
            #     _, self.B_Flow = self.netF(self.real_SA2, self.real_SA3, iters=20, test_mode=True)
            #     self.b_flow = viz(self.B_Flow.detach())

            # sa1 = copy.copy(self.real_SA1)
            # B, N, H, W = sa1.shape
            # sa1 = sa1.expand(B, 3, H, W)
            # sa2 = copy.copy(self.real_SA2)
            # sa2 = sa2.expand(B, 3, H, W)
            # sa3 = copy.copy(self.real_SA3)
            # sa3 = sa3.expand(B, 3, H, W)
            # self.netF.eval()
            # padder = InputPadder(sa1.shape)
            # sa1, sa2, sa3 = padder.pad(sa1, sa2, sa3)
            # _, self.F_Flow = self.netF(sa1, sa2, iters=20, test_mode=True)
            # self.f_flow = viz(self.F_Flow.detach())
            # _, self.B_Flow = self.netF(sa3, sa2, iters=20, test_mode=True)
            # self.b_flow = viz(self.B_Flow.detach())

        else:
            AtoB = self.opt.direction == 'AtoB'
            # self.real_SA1 = input['SA1' if AtoB else 'SB'].to(self.device)
            # self.real_SA2 = input['SA2' if AtoB else 'SB'].to(self.device)
            # self.real_SA3 = input['SA3' if AtoB else 'SB'].to(self.device)
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)  #ceshi xiugai
            self.image_paths = input['SA_path']

            self.F_Flow = f_flow.to(self.device)
            self.B_Flow = b_flow.to(self.device)

            # padder = InputPadder(self.real_SA1.shape)
            # sa1, sa2, sa3 = padder.pad(self.real_SA1, self.real_SA1, self.real_SA1)
            # self.netF.eval()
            # with torch.no_grad():
            #     _, self.F_Flow = self.netF(sa1, sa2, iters=20, test_mode=True)
            #     self.f_flow = viz(self.F_Flow.detach())
            #     _, self.B_Flow = self.netF(sa3, sa2, iters=20, test_mode=True)
            #     self.b_flow = viz(self.B_Flow.detach())



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_SB = self.netG(self.real_SA, self.F_Flow, self.B_Flow)
        # self.fake_SB = torch.tanh(self.fake_SB)
        # self.fake_SB = self.netG(self.real_SA, self.real_SA1, self.real_SA3)
        # self.fake_SB = self.netG(self.real_SA)

    def test(self):
        with torch.no_grad():
            self.fake_SB = self.netG(self.real_SA, self.F_Flow, self.B_Flow)

    def backward_D(self):
        self.B_mask = self.netD(self.fake_SB.detach())

        self.loss_S = self.criterionSegment(self.B_mask, self.S_mask.squeeze(1))
        self.loss_Dice = self.criterionDice(self.B_mask, self.S_mask)

        self.loss_D = self.loss_S + self.loss_Dice
        self.loss_D.backward()
        # pass
        # for name, param in self.netD.named_parameters():
        #     if param.grad is not None:
        #         print(f'{name} grad: {param.grad.norm()}')

    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        Generator should fool the DP
        """
        self.B_mask = self.netD(self.fake_SB)

        gt_sdf = compute_sdf(self.S_mask, self.S_mask.shape)
        self.loss_B = self.boundary(self.B_mask, gt_sdf)
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB)
        self.loss_ST = self.stloss(self.fake_SB, self.real_SB, self.S_mask)
        self.loss_S = self.criterionSegment(self.B_mask, self.S_mask.squeeze(1))
        self.loss_Dice = self.criterionDice(self.B_mask, self.S_mask)
        # # #
        self.loss_D = self.loss_S + self.loss_Dice + self.loss_B
        # #
        self.loss_G = self.loss_G_L1*10 + self.loss_D * self.opt.lambda_D + self.loss_ST*10
        # self.loss_G = self.loss_G_L1 + self.loss_D * self.opt.lambda_D
        # self.loss_G = self.loss_G_L1*10
        self.loss_G.backward()

        # for name, param in self.netG.named_parameters():
        #     if param.grad is not None:
        #         print(f'{name} grad: {param.grad.norm()}')


    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
        #
        # # update G
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        self.B_Mask = F.softmax(self.B_mask, dim=1)
        self.B_Mask = self.B_Mask[:, 1, :, :]
        self.B_Mask = self.B_Mask.unsqueeze(1)
        # pass
