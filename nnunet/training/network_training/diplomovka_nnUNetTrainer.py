import torch
from torch import nn

from nnunet.network_architecture.AG import AG
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.dice_loss import DC_and_BCE_loss
from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2020.nnUNetTrainerV2BraTSRegions_moreDA import \
    nnUNetTrainerV2BraTSRegions_DA4_BN_BD


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
                                    512,
                                    encoder_scale=2)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = nn.Sigmoid()
        self.max_num_epochs = 100


class AG_trainer(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):  # BL + AA
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
        self.network = AG(self.num_input_channels, self.base_num_features, self.num_classes,
                          len(self.net_num_pool_op_kernel_sizes),
                          self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                          dropout_op_kwargs,
                          net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                          self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, 320,
                          encoder_scale=1,
                          axial_attention=True, heads=1, dim_heads=4, volume_shape=(128, 160, 112), no_attention=[4])
        # print(self.network)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = nn.Sigmoid()


class diplomovka_baseline(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = DC_and_BCE_loss({}, {'batch_dice': True, 'do_bg': True, 'smooth': 0})
        self.max_num_epochs = 100
        if torch.cuda.is_available():
            self.network.cuda()
        print(self.network)
