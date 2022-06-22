
def fetch_model(session_config):
    model_name = session_config['Meta']['model_name']

    if model_name == 'resnet18':
        from models.ResNet18 import resnet18 as NN_model
        model_name = 'ResNet18'
    elif model_name == 'resnet50':
        from models.ResNet50 import resnet50 as NN_model
        model_name = 'ResNet50'
    elif model_name == 'wrn_28_4':
        from models.WRN_28_4 import wide_resnet_28_4 as NN_model
        model_name = 'WRN_28_4'
    elif model_name == 'vgg16_bn_Hydra':
        from models.VGG16_bn_Hydra import vgg16_bn as NN_model
    # elif model_name == 'resnet20_cifar':
    # from models.ResNet20_Cifar import resnet20_cifar as NN_model
    #     model_name = 'ResNet20_Cifar'
    # elif model_name == 'resnet56_cifar':
    #     from models.ResNet56_Cifar import resnet56_cifar as NN_model
    #     model_name = 'ResNet56_Cifar'
    # elif model_name == 'wrn_34_10':
    #     from models.WRN_34_10 import wide_resnet_34_10 as NN_model
    #     model_name = 'WRN_34_10'
    # elif model_name == 'vgg16':
    #     from models.VGG16 import vgg16 as NN_model
    # elif model_name == 'lenet5':
    #     from models.LeNet5 import lenet5 as NN_model
    # elif model_name == 'cw_net_cifar':
    #     from models.CW_Net_Cifar import cw_net as NN_model
    #     model_name = 'CW_Net_Cifar'
    else:
        raise NameError('Model name do not exit, please check "model_name" again !')

    return model_name, NN_model