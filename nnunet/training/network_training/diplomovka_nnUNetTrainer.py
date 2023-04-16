import torch
from torch import nn

from nnunet.network_architecture.AG_unet import AG_unet
from nnunet.network_architecture.RSE_unet import RSE_unet
from nnunet.network_architecture.SE_unet import SE_unet
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2020.nnUNetTrainerV2BraTSRegions_moreDA import \
    nnUNetTrainerV2BraTSRegions_DA4_BN_BD


class diplomovka_baseline(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    320,
                                    encoder_scale=1)
        if torch.cuda.is_available():
            self.network.cuda()
        self.max_num_epochs = 100
        print(self.network)


class diplomovka_nnUNetTrainer(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    320,
                                    encoder_scale=1)
        if torch.cuda.is_available():
            self.network.cuda()
        # self.network.inference_apply_nonlin = nn.Sigmoid()
        self.max_num_epochs = 100
        print(self.network)


class diplomovka_largeUnet_trainer(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    # torch.cuda.empty_cache()
    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    1024,
                                    encoder_scale=2)
        if torch.cuda.is_available():
            self.network.cuda()
        # self.network.inference_apply_nonlin = nn.Sigmoid()
        self.max_num_epochs = 100


class AG_trainer(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        self.max_num_epochs = 100
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = AG_unet(self.num_input_channels, self.base_num_features, self.num_classes,
                               len(self.net_num_pool_op_kernel_sizes),
                               self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                               dropout_op_kwargs,
                               net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                               self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 320,
                               encoder_scale=1,
                               axial_attention=True, heads=4, dim_heads=36, volume_shape=(128, 160, 112),
                               no_attention=[0])
        if torch.cuda.is_available():
            self.network.cuda()
        # self.network.inference_apply_nonlin = nn.Sigmoid()
        print(self.network)


class AG_trainer_large(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        self.max_num_epochs = 100
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = AG_unet(self.num_input_channels, self.base_num_features, self.num_classes,
                               len(self.net_num_pool_op_kernel_sizes),
                               self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                               dropout_op_kwargs,
                               net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                               self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 512,
                               encoder_scale=2,
                               axial_attention=True, heads=4, dim_heads=36, volume_shape=(128, 128, 128),
                               no_attention=[4])
        if torch.cuda.is_available():
            self.network.cuda()


class SE_trainer(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        self.max_num_epochs = 100
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = SE_unet(self.num_input_channels, self.base_num_features, self.num_classes,
                               len(self.net_num_pool_op_kernel_sizes),
                               self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                               dropout_op_kwargs,
                               net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                               self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 320,
                               encoder_scale=1)
        print(self.network)
        if torch.cuda.is_available():
            self.network.cuda()


class SE_trainer_large(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        self.max_num_epochs = 100
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = SE_unet(self.num_input_channels, self.base_num_features, self.num_classes,
                               len(self.net_num_pool_op_kernel_sizes),
                               self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                               dropout_op_kwargs,
                               net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                               self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 512,
                               encoder_scale=2)
        print(self.network)
        if torch.cuda.is_available():
            self.network.cuda()


class RSE_trainer(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        self.max_num_epochs = 100
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = RSE_unet(self.num_input_channels, self.base_num_features, self.num_classes,
                                len(self.net_num_pool_op_kernel_sizes),
                                self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 320,
                                encoder_scale=1)
        print(self.network)
        if torch.cuda.is_available():
            self.network.cuda()


class RSE_trainer_large(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def initialize_network(self):
        self.max_num_epochs = 100
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.BatchNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.BatchNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = RSE_unet(self.num_input_channels, self.base_num_features, self.num_classes,
                                len(self.net_num_pool_op_kernel_sizes),
                                self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs,
                                net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 512,
                                encoder_scale=2)
        print(self.network)
        if torch.cuda.is_available():
            self.network.cuda()
