import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet  #https://github.com/lukemelas/EfficientNet-PyTorch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
from torchvision.models.mobilenet import MobileNetV2
from torchvision.models.mobilenetv2 import model_urls
from torch.nn.parameter import Parameter
import torch.nn.functional as F

"""
usefull utilities understanding what is trainable 
for name, param in model_ft.named_parameters():
    if param.requires_grad:
        print (name, param.data)        
        
pip install torchsummary
from torchsummary import summary
device_id=0
summary(model_ft.to(device_id),(3,256,256))
"""

def replace_activation_layer(model, activation_new, activation_ref=nn.ReLU6):  #https://forums.fast.ai/t/change-activation-function-in-resnet-model/78456/13
    for child_name, child in model.named_children():
        # print(child)
        if isinstance(child, activation_ref):
            setattr(model, child_name, activation_new)# nn.LeakyReLU()
        else:
            # recurse
            replace_activation_layer(child, activation_new, activation_ref)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def enable_finetune_layers(model_ft, n_layers_finetune, n_layers_resnet18=5):
    # revert not trainable to trainable from the last layer backwards
    layer_requires_grad = n_layers_resnet18 - n_layers_finetune + 1  # last FC is always tuned=grad is True
    layer_requires_grad_list = list(range(layer_requires_grad, n_layers_resnet18))
    layer_requires_grad_list = [str(i) for i in layer_requires_grad_list]
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            layer_no = name.split('.')[0].split('layer')[-1]
            if layer_no in layer_requires_grad_list:
                param.requires_grad_(True)
                # print(layer_no)
            else:
                param.requires_grad_(False)
    if "fc" in dir(model_ft):
        if isinstance(model_ft.fc, nn.Sequential):
            model_ft.fc[0].requires_grad_(True)
        else:
            model_ft.fc.requires_grad_(True)

    return model_ft

def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
    """Constructs fully connected layer.

    Args:
        fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
        input_dim (int): input dimension
        dropout_p (float): dropout probability, if None, dropout is unused
    """
    if fc_dims is None:
        feature_dim = input_dim
        return None

    assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
        type(fc_dims))

    layers = []
    for dim in fc_dims:
        layers.append(nn.Linear(input_dim, dim))
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        input_dim = dim

    feature_dim = fc_dims[-1]

    return nn.Sequential(*layers)

def initialize_model(model_name, num_classes, feature_extract,
                     pretrained_type, n_layers_finetune=1,
                     dropout=0.2, dropblock_group='4', replace_relu=False,
                     len_handcrafted_features=0, **kwargs):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    use_pretrained = pretrained_type['source']
    path_pretrained = pretrained_type['path']
    # pooling_method = [kwargs['pooling_method'] if 'pooling_method' in kwargs else None][0]

    model_ft = None
    input_size = 0
    input_size = 224

    if model_name == "resnet":
        """ Resnet18
        """
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.resnet18(pretrained=use_pretrained == 'imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet18_256_win_n_lyrs_dropblock":
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = custom_resnet18(pretrained=use_pretrained=='imagenet', dropblock_group=dropblock_group)
        # set_parameter_requires_grad(model_ft, feature_extract) # all not trainable

        model_ft = enable_finetune_layers(model_ft, n_layers_finetune)

        model_ft.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            model_ft.fc
        )
        input_size = 256

        #     testing the status of require_grad
        print_req_grad_stat(model_ft)

    elif model_name == "resnet18_256_win_n_lyrs_notall":
        if replace_relu:
            raise ValueError('Not imp yet')

        if n_layers_finetune > 5:
            print("Warning asked to Fintune more than ResNet18 5 layers can do")

        n_layers_finetune = min(n_layers_finetune, 5)

        model_ft = models.resnet18(pretrained=use_pretrained=='imagenet')  # all trainable
        # set_parameter_requires_grad(model_ft, feature_extract) # all not trainable
        model_ft = enable_finetune_layers(model_ft, n_layers_finetune)

        model_ft.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            model_ft.fc
        )
        input_size = 256

        #     testing the status of require_grad
        print_req_grad_stat(model_ft)

    elif model_name == "alexnet":
        """ Alexnet
        """
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.alexnet(pretrained=use_pretrained=='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.vgg11_bn(pretrained=use_pretrained=='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":  # TODO : with fewer params recall :  r"""SqueezeNet 1.1 model from the `official SqueezeNet repo use : squeezenet1_1(pretrained=False, progress=True, **kwargs):
        """ Squeezenet
        """
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.squeezenet1_0(pretrained=use_pretrained=='imagenet')
        # set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        if 1: #partach to fine tune allmost all net
            n_layers_resnet18 = 13
            n_layers_finetune = min(n_layers_finetune, n_layers_resnet18)
            model_ft.features = enable_finetune_layers(model_ft.features, n_layers_finetune=n_layers_finetune,
                                                       n_layers_resnet18=n_layers_resnet18)
            # for name, param in model_ft.named_parameters():
            #     param.requires_grad_(True)
        input_size = 224
    elif model_name == "efficientnet_b0_w256":
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

        # for param in model_ft.parameters():
        #     param.requires_grad

        input_size = 256
    elif model_name == "densenet":
        if replace_relu:
            raise ValueError('Not imp yet')

        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained=='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        if replace_relu:
            raise ValueError('Not imp yet')

        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained=='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "mobilenet_v2":
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.mobilenet_v2(pretrained=use_pretrained=='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(num_ftrs, num_classes)
        )
        input_size = 224

    elif model_name == "mobilenet_v2_256_win":
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.mobilenet_v2(pretrained=use_pretrained=='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(num_ftrs, num_classes)
        )
        input_size = 256

    # custom_mobilenet_v2
    elif model_name == "mobilenet_v2_2FC_w256_fusion_3cls" or model_name == 'mobilenet_v2_2FC_w256_fusion_2cls':
        kwargs.update({'len_handcrafted_features': len_handcrafted_features,
                  'num_classes': num_classes,
                  'dropout': dropout,
                  'n_layers_finetune': n_layers_finetune,
                  'dim_fc': [256, 64], # cost reduction reduce coeff avoid overfitting
                  'baseline_model_n_classes': 1000,
                  'path_pretrained': path_pretrained})

        model_ft = custom_mobilenet_v2(pretrained=use_pretrained, **kwargs)

        if replace_relu:
            activation_new = nn.PReLU()
            print("Relu activation replaced by : {}".format(activation_new))
            # model_ft = replace_activation_layer(model_ft, activation_new=nn.LeakyReLU(), activation_ref=nn.ReLU6)
            model_ft = replace_activation_layer(model_ft, activation_new=activation_new, activation_ref=nn.ReLU6)


        input_size = 256

    elif model_name == "mobilenet_v2_2FC_w256_nlyrs":  # mobilenet_v2_256_win_n_lyrs

        kwargs.update({'len_handcrafted_features': len_handcrafted_features,
                  'num_classes': num_classes,
                  'dropout': dropout,
                  'n_layers_finetune': n_layers_finetune,
                  'dim_fc': [1024, 256],
                  'baseline_model_n_classes': 1000})

        model_ft = custom_mobilenet_v2(pretrained=use_pretrained == 'imagenet', **kwargs)

        if replace_relu:
            activation_new = nn.PReLU()
            print("Relu activation replaced by : {}".format(activation_new))
            # model_ft = replace_activation_layer(model_ft, activation_new=nn.LeakyReLU(), activation_ref=nn.ReLU6)
            model_ft = replace_activation_layer(model_ft, activation_new=activation_new, activation_ref=nn.ReLU6)


        input_size = 256

    # elif model_name == "mobilenet_v2_2FC_hcf_fc_w256_nlyrs":  # mobilenet_v2_256_win_n_lyrs
    #     if replace_relu:
    #         raise ValueError('Not imp yet')
    #
    #     kwargs = {'len_handcrafted_features': len_handcrafted_features,
    #               'num_classes': 2,
    #               'dropout': dropout,
    #               'n_layers_finetune': n_layers_finetune,
    #               'dim_fc': [1024, 256]}
    #
    #     model_ft = custom_mobilenet_v2(pretrained=use_pretrained, **kwargs)
    #
    #     if replace_relu:
    #         activation_new = nn.PReLU()
    #         print("Relu activation replaced by : {}".format(activation_new))
    #         # model_ft = replace_activation_layer(model_ft, activation_new=nn.LeakyReLU(), activation_ref=nn.ReLU6)
    #         model_ft = replace_activation_layer(model_ft, activation_new=activation_new, activation_ref=nn.ReLU6)
    #
    #
    #     input_size = 256

    elif model_name == "mobilenet_v2_cls_head_256_n_lyrs":  # mobilenet_v2_256_win_n_lyrs
        if replace_relu:
            raise ValueError('Not imp yet')

        model_ft = models.mobilenet_v2(pretrained=use_pretrained=='imagenet')
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        dim_fc = [1024, 256]
        model_ft.classifier = nn.Sequential(
            # nn.Dropout(p=dropout, inplace=False),  # can add that dropout also

            nn.Linear(num_ftrs, dim_fc[0]),
            # nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(dim_fc[0], dim_fc[1]),
            # nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(dim_fc[1], num_classes)
        )
        if replace_relu:
            activation_new = nn.PReLU()
            print("Relu activation replaced by : {}".format(activation_new))
            # model_ft = replace_activation_layer(model_ft, activation_new=nn.LeakyReLU(), activation_ref=nn.ReLU6)
            model_ft = replace_activation_layer(model_ft, activation_new=activation_new, activation_ref=nn.ReLU6)


        input_size = 256

        # model is all requires_grad  = true i.e trainable now set the the first N layers to true
        ch_acm = []
        for child in model_ft.children():
            ch_acm += [child]

        total_feat_ext_layers = len(ch_acm[0])  # assuming all model is under sequential()
        freezed_n_layers = max(0, total_feat_ext_layers - (
                    n_layers_finetune - (1+len(dim_fc))))  # 1 for the Vanilla classifier +2 FC already requires_grad=true
        #     run over the feature extraction part of the sequential modeling the [1] is the one of the classifier already set to requires grad=true
        layer_cnt = 0
        for ch in ch_acm[0]:
            if layer_cnt == freezed_n_layers:
                break
            layer_cnt += 1
            ch.requires_grad_(False)
        #     testing the status of require_grad
        print_req_grad_stat(model_ft)

    elif model_name == 'ConvNet11':
        if replace_relu:
            raise ValueError('Not imp yet')

        input_size = 256
        model_ft = ConvNet11(n_outputs=2, n_input_channels=3, dropout=0.5, abs_instead_of_first_relu=False,
                        use_handcrafted_features=0, activations='relu', depth_mult_fact=1.0, input_dim=input_size)
        # model_ft.enable_layers_finetune(n_layers=n_layers_finetune)

        print_req_grad_stat(model_ft)

    elif model_name == "mobilenet_v2_256_win_n_lyrs_depth_0p25": # exception when loading coeff TODO ::
        if replace_relu:
            raise ValueError('Not imp yet')
        model_urls = {
            'mobilenet_v2': 'https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2_0.25-b61d2159.pth',
        }
        from torch.hub import load_state_dict_from_url
        model = MobileNetV2()
        progress = True
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        model.load_state_dict(state_dict)
        # model = MobileNetV2()
        # progress = True
        # if pretrained:
        #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        #     model.load_state_dict(state_dict)

    elif model_name == "mobilenet_v2_256_win_n_lyrs" or model_name == "mobilenet_v2_64_win_n_lyrs" or model_name == "mobilenet_v2_224_win_n_lyrs":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained=='imagenet')
        # set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(num_ftrs, num_classes, bias=True)
        )
        if replace_relu:
            activation_new = nn.PReLU()
            print("Relu activation replaced by : {}".format(activation_new))
            # model_ft = replace_activation_layer(model_ft, activation_new=nn.LeakyReLU(), activation_ref=nn.ReLU6)
            model_ft = replace_activation_layer(model_ft, activation_new=activation_new, activation_ref=nn.ReLU6)

        if model_name == "mobilenet_v2_256_win_n_lyrs":
            input_size = 256
        elif model_name == "mobilenet_v2_64_win_n_lyrs":
            input_size = 64
        elif model_name == "mobilenet_v2_224_win_n_lyrs":
            input_size = 224
        # model is all requires_grad  = true i.e trainable now set the the first N layers to true
        ch_acm = []
        for child in model_ft.children():
            ch_acm += [child]

        total_feat_ext_layers = len(ch_acm[0])  # assuming all model is under sequential()
        freezed_n_layers = max(0, total_feat_ext_layers - (
                    n_layers_finetune - 1))  # 1 for the classifier already requires_grad=true
        #     run over the feature extraction part of the sequential modeling the [1] is the one of the classifier already set to requires grad=true
        layer_cnt = 0
        for ch in ch_acm[0]:
            if layer_cnt == freezed_n_layers:
                break
            layer_cnt += 1
            ch.requires_grad_(False)
        #     testing the status of require_grad
        print_req_grad_stat(model_ft)
    else:
        print("Invalid model name, exiting...")
        exit()

    # if args.fine_tune_pretrained_model_plan == 'freeze_and_add_nn':
    #     set_parameter_requires_grad(model_ft, feature_extract) # set all layer grad to false TODO on training set that model to eval
    #     # open_specified_layers(model, open_layers)

    return model_ft, input_size

def print_req_grad_stat(model_ft):
    req_grad = []
    for param in model_ft.parameters():
        req_grad += [param.requires_grad]
    print(req_grad)


# MC dropout
def dropout_test_time_on(m):
    if type(m) == nn.Dropout:
        m.train()


def dropout_test_time_off(m):
    if type(m) == nn.Dropout:
        m.eval()


from torchvision.models.resnet import BasicBlock, ResNet
from torch.hub import load_state_dict_from_url
from dev.augmentation.dropblock import DropBlock2D
from dev.augmentation import LinearScheduler
# from augmentation.dropblock import DropBlock2D
# from augmentation.scheduler import LinearScheduler


def custom_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                          **kwargs)


def _custom_resnet(arch, block, layers, pretrained, progress, **kwargs):
    from torchvision.models.resnet import BasicBlock, ResNet, model_urls

    model = ResNetCustom(block, layers, drop_prob=0.1, block_size=7, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000, drop_prob=0., block_size=5, dropblock_group='4'):
        super(ResNetCustom, self).__init__(block=block, layers=layers, num_classes=num_classes)
        # self.state_dict_vanilla = ResNetCustom.state_dict
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        self.blockdrop_mode = dropblock_group

    # Plain Vanilla forward method
    def _forward_pv(self, x):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dropblock(self.layer1(x))
        x = self.dropblock(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _forward_drop_block_group_3_4(self, x):
        self.dropblock.step()  # increment number of iterations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropblock(self.layer3(x))
        x = self.dropblock(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # make more sense since it doesn't interfere with BN as dropout should be stayed away from BN
    def _forward_drop_block_group_4(self, x):
        self.dropblock.step()  # increment number of iterations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropblock(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        if self.blockdrop_mode == '1_2':
            return self._forward_pv(x)
        if self.blockdrop_mode == '3_4':
            return self._forward_drop_block_group_3_4(x)
        if self.blockdrop_mode == '4':
            return self._forward_drop_block_group_4(x)

# adding HCF to mobilENtV2
def custom_mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
# all image 3 classes classification when loading the backbone based on MobileNet 1st load the MobileNet vanilla skeleton with n class=1000,
# then load pretrained coeff over MobileV2 with n=2 class at the classifier output, then  modifying model modify N classes =3
    model = MobileNetV2Custom(**kwargs)
    if pretrained == 'imagenet':
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    elif pretrained == 'checkpoint': # checkpoint loaded from
        # first load Vanilla MobileNetV2 then make the changes needed
        checkpoint = torch.load(kwargs['path_pretrained'], map_location=kwargs['device'])
        for k, v in checkpoint.items():
            if k.__contains__('classifier'): # first load the right 2 classes model pre-trained
                if v.shape[0] == 2:
                    kwargs.update({'baseline_model_n_classes': 2})
                    model = MobileNetV2Custom(**kwargs)
                    break
                    # kwargs['baseline_model_n_classes']

        model.load_state_dict(checkpoint)
        print("Loading pre-trained model all trainable at that point")

        # for the vanilla model load over the vanilla structure then modify structure + init that new struct
    model.modify_model_structure()
    return model


class MobileNetV2Custom(MobileNetV2):
    def __init__(self, **kwargs):
        super(MobileNetV2Custom, self).__init__(num_classes=kwargs['baseline_model_n_classes'], # when loading the backbone based on MobileNet 1st load the MobileNet vanilla skeleton with n class=1000, the load pretrained coeff over MobileV2 with n=2 class at the classifier output, then when modifying model modify N classes =3
                                                 width_mult=1.0,
                                                 inverted_residual_setting=None,
                                                 round_nearest=8,
                                                 block=None)

        self.n_layers_finetune = kwargs['n_layers_finetune']
        self.dim_fc = kwargs['dim_fc'] #[1024, 256]
        self.dropout = kwargs['dropout']
        self.num_classes = kwargs['num_classes']
        self.fusion_type = [kwargs['pooling_method'] if 'pooling_method' in kwargs else None][0]
        self.pooling_at_classifier = [kwargs['pooling_at_classifier'] if 'pooling_at_classifier' in kwargs else True][0]
        self.fc_sequential_type = [kwargs['fc_sequential_type'] if 'fc_sequential_type' in kwargs else '2FC'][0]
        self.debug = [kwargs['debug'] if 'debug' in kwargs else None][0]# TODO if inference then debug else no need ?
        self.positional_embeddings = [kwargs['positional_embeddings'] if 'positional_embeddings' in kwargs else None][0]
        self.transformer_param_list = [kwargs['transformer_param_list'] if 'transformer_param_list' in kwargs else None][0] #d_model, nhead=2, nhid=200, nlayers=2
        if self.transformer_param_list != None:
            self.transformer_d_model = self.transformer_param_list[0]
            self.n_head = self.transformer_param_list[1]
            self.nhid = self.transformer_param_list[2]  # dim_feedforward
            self.nlayer = self.transformer_param_list[3]
            self.pos_embed_type = self.transformer_param_list[4]
            assert self.transformer_d_model // self.n_head * self.n_head == self.transformer_d_model, "embed_dim must be divisible by num_heads or self.transformer_d_model/self.n_head shouldn't be fractional"

        #pos embedings
        self.positional_embed_len_post_proc = 0
        self.positional_embed_len = 0
        if self.positional_embeddings:
            if (self.positional_embeddings == 'raster_bitmap_8_8_1D' or
                self.positional_embeddings == 'raster_bitmap_8_8_mlp_1D' or
                self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D'):

                self.positional_embed_len = 8*8
                self.positional_embed_len_post_proc = self.positional_embed_len
                if self.positional_embeddings == 'raster_bitmap_8_8_mlp_1D':
                    self.positional_embed_len_post_proc = int(self.positional_embed_len/2)
                elif self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D':
                    self.positional_embed_len_post_proc = self.classifier[-1].in_features # Vanila MobileNetV2 feature space

        if 'device' in kwargs: #for created inside vecor appended
            self.device = kwargs['device']
        if len(self.dim_fc) != 2:
            raise ValueError("Supported only 2 non linear FCs ")

        self.len_handcrafted_features = kwargs['len_handcrafted_features']

    def modify_model_structure(self): # all these modules are new hence init them
#TODO if a new layer isn't specified it it woun't be trained by the open_specified_layers()!!!!
        self.open_layers_for_training = list()

        if self.fusion_type == 'avg_pooling':
            self.pool_method = Avg_Pooling()
            self.mlp = Mlp(in_features=self.dim_fc[1], drop=self.dropout)  # MLP not reducing dimention
            self.open_layers_for_training.append('mlp')
        elif self.fusion_type == 'lp_mean_pooling':
            self.pool_method = Lp_Avg_Pooling() # in GEM self.p = Parameter() prevent from loading pretrained
        elif self.fusion_type == 'gated_attention':
            self.open_layers_for_training.append('pool_method')  # onlt that is learnable
            if self.pooling_at_classifier == True:
                self.pool_method = GatedAttention(embeddings_dim=self.dim_fc[-1],
                                                      attention_dim=min(128, self.dim_fc[-1]),
                                                      dropout=self.dropout, debug_info=self.debug)
                self.mlp = Mlp(in_features=self.dim_fc[1], drop=self.dropout) # MLP not reducing dimention
                self.open_layers_for_training.append('mlp')
            else:
                self.pool_method = GatedAttention(embeddings_dim=1280, dropout=self.dropout,
                                                      debug_info=self.debug)

        elif self.fusion_type == 'transformer_san':
            self.pool_method = TransformerModel(d_model=self.transformer_d_model, nhead=self.n_head, nhid=self.nhid,
                                                nlayers=self.nlayer, pos_embed_type=self.pos_embed_type, dropout=0.5) # vanila nhead=6 ;d_model=512
            # match the FE dim=1280 to d_model of the transformer input
            self.dim_fc[1] = self.transformer_d_model
            self.open_layers_for_training.append('pool_method')  # onlt that is learnable
            self.fc_sequential_type = '1FC_all_dropout'
        else:
            self.pool_method = None
# dimension of the default MobilenNetV2 classifier
        self.vanila_MNV2_num_ftrs = self.classifier[-1].in_features # 1280

# positional embeddings for non Transformer - adhoc
        if (self.positional_embeddings == 'raster_bitmap_8_8_mlp_1D' or
            self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D'):

            self.positional_embeddings_mlp = nn.Sequential(nn.Linear(self.positional_embed_len,
                                                                     self.positional_embed_len_post_proc),
                                                                        nn.ReLU(inplace=True),
                                                                        nn.Dropout(p=self.dropout))
            # inititlaize the MLP in case there are pretrained weights the checkpoint loading comes later will reset them again
            self._init_weights(self.positional_embeddings_mlp)
            self.open_layers_for_training.append('positional_embeddings_mlp')

#Risky : only if it isn;t additive then it has point to add the N-pos embedd length
        if self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D':
            self.positional_embed_len_post_proc = 0

        if self.fc_sequential_type =='2FC':
            self.fc_sequential = nn.Sequential(
                # nn.Dropout(p=self.dropout, inplace=False), # aDDED in 3/5 HK was added only in 1FC model

                nn.Linear(self.vanila_MNV2_num_ftrs + self.len_handcrafted_features + self.positional_embed_len_post_proc, self.dim_fc[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout),

                nn.Linear(self.dim_fc[0], self.dim_fc[1]),
                nn.ReLU(inplace=True)
            )
            self.open_layers_for_training.append('fc_sequential')

        elif self.fc_sequential_type == '1FC_all_dropout':
            self.fc_sequential = nn.Sequential(
                nn.Dropout(p=self.dropout, inplace=False),
                nn.Linear(self.vanila_MNV2_num_ftrs + self.len_handcrafted_features + self.positional_embed_len_post_proc, self.dim_fc[1]),
                nn.ReLU(inplace=True),
            )
            self.open_layers_for_training.append('fc_sequential')
        else:
            raise
#inititlaize the MLP in case there are pretrained weights the checkpoint loading comes later will reset them again
        self._init_weights(self.fc_sequential)

        self.custom_classifier = nn.Sequential(
           nn.Dropout(p=self.dropout),
           nn.Linear(self.dim_fc[1], self.num_classes)
        )
        self.open_layers_for_training.append('custom_classifier')
# inititlaize the MLP in case there are pretrained weights the checkpoint loading comes later will reset them again
        self._init_weights(self.custom_classifier)


        # gather info about modules to determine gradient true for training N layers
        self.ch_acm = []
        for child in self.children():
            self.ch_acm += [child]

        self.enable_fine_tune_n_layer()  # redundant see open_specified_layers()

        #need to re define the unused classifier to support loaded mobileNetV2 pretrained models
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(self.vanila_MNV2_num_ftrs, 2)
        )# grad=False no need to train

    def _init_weights(self, module, init_range: float = 0.1) -> None:
        for name, sub_module in module.named_children():
            if type(sub_module) == nn.Linear:
                sub_module.weight.data.uniform_(-init_range, init_range)
            elif type(sub_module) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(sub_module.weight.data) # no debugged
                if hasattr(sub_module.bias, 'data'):
                    sub_module.bias.data.zero_()

    def enable_fine_tune_n_layer(self): # only require_grad=true for the relevant layers pay attention to open_specified_layers() funstionality
        total_feat_ext_layers = len(self.ch_acm[0])  # assuming all model is under sequential()

        freezed_n_layers = max(0, total_feat_ext_layers - (
                    self.n_layers_finetune - (1+len(self.dim_fc))))  # 1 for the Vanilla classifier +2 FC already requires_grad=true
        #     run over the feature extraction part of the sequential modeling the [1] is the one of the classifier already set to requires grad=true
        layer_cnt = 0
        for ch in self.ch_acm[0]:
            if layer_cnt == freezed_n_layers:
                break
            layer_cnt += 1
            ch.requires_grad_(False)
        #     testing the status of require_grad
        # print_req_grad_stat(model_ft)
        if self.debug:
            req_grad = []
            for param in self.parameters():
                req_grad += [param.requires_grad]
            print(req_grad)
# way to watch each layer + shape + require_grad
        # for idx, child in enumerate(self.children()):
        #     print("idx: {} child: {} ".format(idx, child))
        #     for param in child.parameters():
        #         print("param: {} shape : {} grad: {}".format(param, param.shape, param.requires_grad))

        return

    def _create_postioanl_embed_raster_bitmap_8_8_1D(self, tile_index_pos, pos_n_rows_m_cols):
        normalized_embed = True
        # self.positional_embed_len = 8*8
        all_pos_embedd = list()
        for id, (mat_dim, tind) in enumerate(zip(pos_n_rows_m_cols, tile_index_pos)):
            embed_mat = torch.zeros([len(tile_index_pos[id]), self.positional_embed_len])
            # uniq_ind_in_mat = torch.unique(torch.floor_divide(tind, (mat_dim.prod()/(self.positional_embed_len))))
            uniq_ind_in_mat = torch.floor_divide(tind, (mat_dim.prod()/(self.positional_embed_len)))
            if normalized_embed:
                cell_indication = 1/uniq_ind_in_mat.shape[0]
            else:
                cell_indication = 1
            embed_mat[np.arange(len(tile_index_pos[id])), uniq_ind_in_mat.type('torch.LongTensor')] = cell_indication
            all_pos_embedd.append(embed_mat)
        return all_pos_embedd

    def _custom_forward_impl_transformer_san(self, x): # SAN = Self Attention
        def pad_seq(seq, max_batch_len: int, pad_value: int):
            # IRL, use pad_sequence
            # https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pad_sequence.html
            return seq + (max_batch_len - len(seq)) * [pad_value]

        if self.len_handcrafted_features:
            images, handcrafted_features = x
            raise ValueError("Not imp. yet")
        else:
            images = x
            pad_token_id = 0
            batch_inputs_emb = list()
            batch_attention_masks = list()

            max_size = max([len(ex) for ex in images])

            if isinstance(images, (list, tuple)):
                for image in images:
                    x = self.features(image)
                    # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
                    x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)  # AvgPooling is inside forward() of MobileNet hence ReImplement
                    if self.len_handcrafted_features:
                        x = torch.cat((x, handcrafted_features.unsqueeze(dim=1)), 1)

                    # print('da')
                    self.embeddings = self.fc_sequential(x)  # squeeze embeddings to match for the SAN
                    attention_mask = [1] * self.embeddings.shape[0]
                    attention_mask = np.array(pad_seq(attention_mask, max_size, pad_token_id))
                    self.embeddings = torch.cat((self.embeddings, torch.zeros(max_size - self.embeddings.shape[0], self.embeddings.shape[1]).to(self.device)), 0)
                    batch_inputs_emb.append(self.embeddings)
                    batch_attention_masks.append(torch.Tensor(attention_mask).to(self.device))
            # Dynamic padding : added pad tokens to reach the length of the longest sequence of each mini batch instead of a fixed
                batch_inputs_emb = torch.stack(batch_inputs_emb) # blind image <=> sentence blind images <=> batch of sentences
                batch_attention_masks = torch.stack(batch_attention_masks)
                pooled_emb = self.pool_method(batch_inputs_emb, batch_attention_masks)  # https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        #poooling approach : avg pooling over all sequence
                if self.pool_method.pos_embed_type == 1: #TODO on inference/validation take pooling wothout the padding part of the sequence
                    # pooling (batch, seq_len_d_model) => (batch, d_model)
                    x = torch.mean(pooled_emb, 1) #avg pooling over the sequence length axis (N,d_model)
                elif self.pool_method.pos_embed_type == 2:
                    x = pooled_emb[:, 0, :] #according to VIT (ViTPooler) : https://github.com/huggingface/transformers/blob/fa84540e98a6af309c3007f64def5011db775a70/src/transformers/models/vit/modeling_vit.py#L435
                elif self.pool_method.pos_embed_type == 3: # VitPooler of huggingfaces  :see class ViTPooler
                    raise  # TODO VIT polloer
                elif self.pool_method.pos_embed_type == 4:
                    raise # # TODO : pooler with the relevant lebngth w/o the padding, ie the actual length
                x = self.custom_classifier(x)  # avg pooling of the last FC output
            else:
                raise ValueError("Not imp. yet")
        return x


    def _custom_forward_impl_embeddings_pooling(self, x): #Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. arXiv preprint arXiv:1802.04712. link.
        if self.len_handcrafted_features:
            images, handcrafted_features = x
            raise ValueError("Not imp. yet")
        else:
            if self.positional_embeddings is not None:
                images, tile_index_pos, pos_n_rows_m_cols = x
                if (self.positional_embeddings == 'raster_bitmap_8_8_1D' or
                    self.positional_embeddings == 'raster_bitmap_8_8_mlp_1D' or
                    self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D'):

                    positional_emb_minibatch = self._create_postioanl_embed_raster_bitmap_8_8_1D(tile_index_pos, pos_n_rows_m_cols)
                    positional_emb_minibatch = [positional_emb.to(self.device) for positional_emb in positional_emb_minibatch]
                    if (self.positional_embeddings == 'raster_bitmap_8_8_mlp_1D' or
                        self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D'):
                        positional_emb_minibatch = [self.positional_embeddings_mlp(positional_emb)
                                                    for positional_emb in positional_emb_minibatch]
                else:
                    raise ValueError("Not imp. yet")
            else:
                images = x # it has to support list hance only x is valid!!
        if self.debug:
            self.attention_final_weights_acc = list()

        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        pooled_x = list()
        if isinstance(images, (list, tuple)):
            for ind, image in enumerate(images):
                x = self.features(image)
                # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
                x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1) # AvgPooling is inside forward() of MobileNet hence ReImplement
                if self.len_handcrafted_features:
                    x = torch.cat((x, handcrafted_features.unsqueeze(dim=1)), 1)

                if self.positional_embeddings is not None:
                    if self.positional_embeddings == 'additive_raster_bitmap_8_8_mlp_1D':
                        x = x + positional_emb_minibatch[ind]
                    else:
                        x = torch.cat((x, positional_emb_minibatch[ind]), 1)

                self.fc_input = x # Test point for debugging for taking embeddings out of non linear domain
                if not self.pooling_at_classifier: # poolong over the input to the FC-NN

                    pooled_emb = self.pool_method(x) # avg pooling
                    if self.fusion_type != 'gated_attention':
                        self.embeddings = self.fc_sequential(pooled_emb)
                    elif self.fusion_type == 'transformer_san':
                        raise ValueError("Not imp. yet")
                    else:
                        self.embeddings = self.fc_sequential(pooled_emb)
                        if self.debug:
                            self.attention_final_weights_acc.append(self.pool_method.attention_final_weights.cpu().numpy())

                    x = self.custom_classifier(self.embeddings)
                else: # pooling over the output to the FC-NN linear fdomain before linear classiifer
# pos embed already injected in the 1280-non linear domain
                    if self.fusion_type != 'gated_attention':
                        self.embeddings = self.fc_sequential(x)
                        pooled_emb = self.pool_method(self.embeddings)  # avg pooling
                        x = self.custom_classifier(pooled_emb) # avg pooling of the last FC output
                    else:
                        self.embeddings = self.fc_sequential(x)
                        pooled_emb = self.pool_method(self.embeddings)
                        x_mlp = self.mlp(pooled_emb)
                        x = self.custom_classifier(x_mlp) # avg pooling of the last FC output
                        if self.debug:
                            self.attention_final_weights_acc.append(self.pool_method.attention_final_weights.detach().cpu().numpy())


                pooled_x.append(x)
            pooled_x = torch.stack(pooled_x)
        return pooled_x

    def _custom_forward_impl(self, x):
        if self.len_handcrafted_features:
            image, handcrafted_features = x
        else:
            image = x[0]

        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(image)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        if self.len_handcrafted_features:
            x = torch.cat((x, handcrafted_features.unsqueeze(dim=1)), 1)

        self.fc_input = x # for taking embeddings out of non linear domain
        self.embeddings = self.fc_sequential(x)
        x = self.custom_classifier(self.embeddings)
        # x = self.classifier(x)
        return x

    def forward(self, x): 
        if self.fusion_type == 'avg_pooling' or self.fusion_type == 'lp_mean_pooling' or self.fusion_type == 'gated_attention':
            return self._custom_forward_impl_embeddings_pooling(x)
        elif self.fusion_type == 'transformer_san':
            return self._custom_forward_impl_transformer_san(x)
        else:
            return self._custom_forward_impl(x)


"""
    def forward_return_multiple_features(self, x):
        if self.use_handcrafted_features:
            image, handcrafted_features = x
        else:
            image = x
        
        x = self.featuresSequential(image)
        # Flattening happens here
        x = x.view(x.shape[0], -1)

        if self.use_handcrafted_features:
            x = self.concatOp(x, handcrafted_features)

        features = self.fcSequential(x)

        x = self.finalSequential(features)

        return x, features

    def forward(self, x):
        return self.forward_return_multiple_features(x)[0]

"""

class ConvNet11(nn.Module):
    """
    Compared to SimpleConvNet10:
    Max pooling operations are replaced by strided convolutions to save memory.
    More fully connected layers to use concatenated hand-crafted features better.

    depth_mult_fact (defualt=1.0): multiplies the amount of preset filters by a value
    gt or lt 1 to oncrease/decrease number of output filters especially when input signal has more than 1 channel
    """

    def __init__(self, n_outputs, n_input_channels=3, dropout=None, abs_instead_of_first_relu=False,
                 use_handcrafted_features=0, activations='relu', depth_mult_fact=1.0,
                 input_dim=224, *args, **kwargs):
        """
        :param use_handcrafted_features: The number of hand crafted features to concatenate to a latent vector.
        By default, 0, which results in a single-input (image) model.
        """
        super(ConvNet11, self).__init__()
        self.n_outputs = n_outputs
        self.input_dim = input_dim
        self.use_handcrafted_features = use_handcrafted_features
        depth_mul = lambda n_channels: int(n_channels * float(depth_mult_fact))

        if depth_mult_fact < 1.0:
            raise ValueError("Depth multiplier {} should be greater than 1 ".format(depth_mult_fact))

        self.n_input_channels = n_input_channels

        bias_in_first_layer = not abs_instead_of_first_relu

        if activations == 'relu':
            activation_fn = nn.ReLU
        elif activations == 'relu6':
            activation_fn = nn.ReLU6
        elif activations == 'tanh':
            activation_fn = nn.Tanh
        else:
            raise ValueError("Activation function {} is unknown.".format(activations))

        layers = []
        layers.append(
            nn.Conv2d(in_channels=n_input_channels, out_channels=depth_mul(16), kernel_size=3, stride=1, padding=1,
                      bias=bias_in_first_layer))
        layers.append(nn.BatchNorm2d(depth_mul(16)))

        if abs_instead_of_first_relu:
            layers.append(Abs())
        else:
            layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=depth_mul(16),
                                out_channels=depth_mul(16), kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(depth_mul(16)))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=(depth_mul(16)),
                                out_channels=(depth_mul(32)), kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d((depth_mul(32))))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=(depth_mul(32)),
                                out_channels=(depth_mul(32)), kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d((depth_mul(32))))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=(depth_mul(32)),
                                out_channels=48, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(48))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(48))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=48, out_channels=40, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(40))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(40))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(40))
        layers.append(activation_fn())

        layers.append(nn.Conv2d(in_channels=40, out_channels=32, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(activation_fn())
        self.output_channels_conv_feature_ext = 32
        layers.append(nn.Conv2d(in_channels=32, out_channels=self.output_channels_conv_feature_ext,
                                kernel_size=3, stride=1, padding=1)) # stride=2
        layers.append(nn.BatchNorm2d(32))
        layers.append(activation_fn())

        # layers.append(nn.AdaptiveMaxPool2d(1))

        self.featuresSequential = nn.Sequential(*layers)

        self.concatOp = Concat()

        layers = []
        # layers.append(nn.Linear(32 + self.use_handcrafted_features, 64))

        # layers.append(nn.Linear(32 * 2 * 3 + self.use_handcrafted_features, 64))
        layers.append(nn.Linear(int(self.output_channels_conv_feature_ext*(self.input_dim/32)**2)
                                + self.use_handcrafted_features, 64))
        # layers.append(nn.BatchNorm1d(64))
        layers.append(activation_fn())

        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(64, 32))
        layers.append(nn.BatchNorm1d(32))
        layers.append(activation_fn())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(32, 32))
        layers.append(nn.BatchNorm1d(32))
        layers.append(activation_fn())
        self.fcSequential = nn.Sequential(*layers)

        layers = []
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(32, self.n_outputs))
        self.fc = nn.Sequential(*layers)

        self.model_init()

    def model_init(self):
        # weight initialization
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return

    def forward_return_multiple_features(self, x):
        if self.use_handcrafted_features:
            image, handcrafted_features = x
        else:
            image = x

        x = self.featuresSequential(image)
        # Flattening happens here
        x = x.view(x.shape[0], -1)

        if self.use_handcrafted_features:
            x = self.concatOp(x, handcrafted_features)

        features = self.fcSequential(x)

        x = self.fc(features)

        return x, features

    def forward(self, x):
        return self.forward_return_multiple_features(x)[0]

    def enable_layers_finetune(self, n_layers=0):
        ct = 0
        for child in self.children():
            ct += 1
            if ct < n_layers:
                for param in child.parameters():
                    param.requires_grad = False

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return x.abs()

class Concat(nn.Module):
    def forward(self, x, y):
        return torch.cat((x, y), 1)


#     GEM implemetation  - Generalized-Mean (GeM) pooling : https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065 ; https://arxiv.org/pdf/1711.02512.pdf
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

# model = se_resnet50(num_classes=1000, pretrained='imagenet')
# model.avg_pool = GeM()
class Avg_Pooling(nn.Module):
    def __init__(self, dim=0):
        super(Avg_Pooling, self).__init__()
        self.dim = dim
    def forward(self, x):
        return torch.mean(x, dim=self.dim)



class Lp_Avg_Pooling(nn.Module):
    def __init__(self, p=3, dim=0, eps=1e-6):
        super(Lp_Avg_Pooling, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=self.dim).pow(1./self.p) #gem(x, p=self.p, eps=self.eps)

# @ embeddings_dim=1280 and attention_dim=128, and n_tiles
# H[n_tiles x 1280] x V[1280x128] = [n_tiles*128]
# H[n_tiles x 1280] x U[1280x128] = [n_tiles*128]
# W[attention_dim x self.K]  = [128x1]
# W{TanH(VH) element_mul signum(UH)} = [n_tiles*128]*[128x1] = [n_tiles*1] i.e can;t watch ak it is variable length in order to understand the wieghted tiles
#=> a_k=dostmax(W{TanH(VH) element_mul signum(UH)}) : softmax([n_tiles*1]) =>[n_tiles*1] => a_k will be hard to explain generally but given specific image

# @ cls input embeddings_dim=64 and attention_dim=64 : V,U[64x64+bias-64], W[1x64 bias-64]
class GatedAttention(nn.Module):
    def __init__(self, embeddings_dim=1280, attention_dim=128, dropout=0, debug_info=False):
        super(GatedAttention, self).__init__()
        self.debug_info = debug_info
        self.dropout = dropout
        self.K = 1
        self.L = embeddings_dim # embeddings dim from input to FC @ embeddings_dim=1280 and attention_dim=128
        self.D = attention_dim
        self.attention_V = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Dropout(p=self.dropout, inplace=False),
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        self._init_weights(module=self.attention_U)
        self._init_weights(module=self.attention_V)

    def _init_weights(self, module, init_range: float = 0.1) -> None:
        for name, sub_module in module.named_children():
            if type(sub_module) == nn.Linear:
                sub_module.weight.data.uniform_(-init_range, init_range)
                if hasattr(sub_module.bias, 'data'):
                    sub_module.bias.data.zero_()

    def forward(self, H): # H : NxL  L-dim N-batch/bag
        # print(np.percentile(H.cpu().numpy().ravel(), 50), np.percentile(H.cpu().numpy().ravel(), 70))
        A_V = self.attention_V(H)  # NxD H:matrix of all the aggregated embeddings by 2018 {Max W} Attention-based Deep Multiple Instance Learning[330].pdf
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        if self.debug_info:
            self.attention_final_weights = A # for debug purpose
            # self.entropy_avg = (-self.attention_final_weights * torch.log(self.attention_final_weights)).sum() / \
            #                     self.attention_final_weights.shape[1]
            # print(self.entropy_avg)
        M = torch.mm(A, H)  # KxL

        return M.squeeze()
#https://github.com/rwightman/pytorch-image-models/blob/07d952c7a78ea0353361624825a4ca514a89f46d/timm/models/layers/mlp.py#L8
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x) # HK added to coupled with dropout+linear FC
        # x = self.drop(x)
        return x


#https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)  # dropout only in text NLP not in image embeddings
        return x

# ntokens = len(vocab) # the size of vocabulary
# emsize = 200 # embedding dimension
# nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
# nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
# nhead = 2 # the number of heads in the multiheadattention models
# dropout = 0.2 # the dropout value
class TransformerModel(nn.Module):  # deteriiorated to encoder only
#pos_embed_type=1 : avg pool over all seq len axis in emcoder output pos_embed_type=2: adding CLS learnable token
    def __init__(self, d_model=256, nhead=2, nhid=200, nlayers=2, pos_embed_type=1, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.max_expected_seq_len = 256
        self.d_model = d_model
        self.model_type = 'Transformer'
        self.goal = 'text_cls'
        self.pos_embed_type = pos_embed_type
        if self.pos_embed_type == 2: # learnable CLS token as in the paper VIT https://arxiv.org/pdf/2010.11929.pdf  / https://amaarora.github.io/2021/01/18/ViT.html#cls-token--position-embeddings
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) # learnable CLS token have to be pre-initialized
        self.pos_encoder = PositionalEncoding(d_model, dropout=0) # dropout=0 for NLP not to vision
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model) # word2Vec

        if self.goal == 'seq_2_seq':
            self.decoder = nn.Linear(d_model, ntoken=100)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self): #'seq_2_seq':
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.goal == 'seq_2_seq':
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, embeddings, embeddings_mask):
        # embeddings = self.encoder(embeddings) * math.sqrt(self.d_model) #Token encoder Word2Vec
        if self.goal == 'seq_2_seq':
            embeddings = self.pos_encoder(embeddings)
            output = self.transformer_encoder(embeddings, embeddings_mask=embeddings_mask)
            output = self.decoder(output)
        elif self.goal == 'text_cls':
            if self.pos_embed_type == 2: # VIT
                batch_size = embeddings.shape[0]
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                embeddings = torch.cat((cls_tokens, embeddings), dim=1)
                embeddings = self.pos_encoder(embeddings)
                output = self.transformer_encoder(embeddings)
            else:
                embeddings = self.pos_encoder(embeddings)
                output = self.transformer_encoder(embeddings, src_key_padding_mask=embeddings_mask)

        return output

def measure_transformer_weight(model):
    acm = 0
    for name, param in model.named_parameters():
        if 'pool_method' in name:
            print(name, param.shape, np.prod(param.shape))
            acm += np.prod(param.shape)
    print("total transformer in {}KB ".format(acm/1024))
    return

class ViTPooler(nn.Module): # huggingfaces pooler for VIT https://github.com/huggingface/transformers/blob/fa84540e98a6af309c3007f64def5011db775a70/src/transformers/models/vit/modeling_vit.py#L435
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
