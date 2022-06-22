import numpy as np

VGG16_Hydra = {
    'features.0': 'vgg16_bn_Hydra/inference/conv1_1',
    'features.1': 'vgg16_bn_Hydra/inference/conv1_1/bn',
    'features.3': 'vgg16_bn_Hydra/inference/conv1_2',
    'features.4': 'vgg16_bn_Hydra/inference/conv1_2/bn',
    'features.7': 'vgg16_bn_Hydra/inference/conv2_1',
    'features.8': 'vgg16_bn_Hydra/inference/conv2_1/bn',
    'features.10': 'vgg16_bn_Hydra/inference/conv2_2',
    'features.11': 'vgg16_bn_Hydra/inference/conv2_2/bn',
    'features.14': 'vgg16_bn_Hydra/inference/conv3_1',
    'features.15': 'vgg16_bn_Hydra/inference/conv3_1/bn',
    'features.17': 'vgg16_bn_Hydra/inference/conv3_2',
    'features.18': 'vgg16_bn_Hydra/inference/conv3_2/bn',
    'features.20': 'vgg16_bn_Hydra/inference/conv3_3',
    'features.21': 'vgg16_bn_Hydra/inference/conv3_3/bn',
    'features.24': 'vgg16_bn_Hydra/inference/conv4_1',
    'features.25': 'vgg16_bn_Hydra/inference/conv4_1/bn',
    'features.27': 'vgg16_bn_Hydra/inference/conv4_2',
    'features.28': 'vgg16_bn_Hydra/inference/conv4_2/bn',
    'features.30': 'vgg16_bn_Hydra/inference/conv4_3',
    'features.31': 'vgg16_bn_Hydra/inference/conv4_3/bn',
    'features.34': 'vgg16_bn_Hydra/inference/conv5_1',
    'features.35': 'vgg16_bn_Hydra/inference/conv5_1/bn',
    'features.37': 'vgg16_bn_Hydra/inference/conv5_2',
    'features.38': 'vgg16_bn_Hydra/inference/conv5_2/bn',
    'features.40': 'vgg16_bn_Hydra/inference/conv5_3',
    'features.41': 'vgg16_bn_Hydra/inference/conv5_3/bn',
    'classifier.0': 'vgg16_bn_Hydra/inference/fc1',
    'classifier.2': 'vgg16_bn_Hydra/inference/fc2',
    'classifier.4': 'vgg16_bn_Hydra/inference/fc3',
}

VGG16_ADMM = {
    'features.0': 'vgg16_bn_ADMM/inference/conv1_1',
    'features.1': 'vgg16_bn_ADMM/inference/conv1_1/bn',
    'features.3': 'vgg16_bn_ADMM/inference/conv1_2',
    'features.4': 'vgg16_bn_ADMM/inference/conv1_2/bn',
    'features.7': 'vgg16_bn_ADMM/inference/conv2_1',
    'features.8': 'vgg16_bn_ADMM/inference/conv2_1/bn',
    'features.10': 'vgg16_bn_ADMM/inference/conv2_2',
    'features.11': 'vgg16_bn_ADMM/inference/conv2_2/bn',
    'features.14': 'vgg16_bn_ADMM/inference/conv3_1',
    'features.15': 'vgg16_bn_ADMM/inference/conv3_1/bn',
    'features.17': 'vgg16_bn_ADMM/inference/conv3_2',
    'features.18': 'vgg16_bn_ADMM/inference/conv3_2/bn',
    'features.20': 'vgg16_bn_ADMM/inference/conv3_3',
    'features.21': 'vgg16_bn_ADMM/inference/conv3_3/bn',
    'features.24': 'vgg16_bn_ADMM/inference/conv4_1',
    'features.25': 'vgg16_bn_ADMM/inference/conv4_1/bn',
    'features.27': 'vgg16_bn_ADMM/inference/conv4_2',
    'features.28': 'vgg16_bn_ADMM/inference/conv4_2/bn',
    'features.30': 'vgg16_bn_ADMM/inference/conv4_3',
    'features.31': 'vgg16_bn_ADMM/inference/conv4_3/bn',
    'features.34': 'vgg16_bn_ADMM/inference/conv5_1',
    'features.35': 'vgg16_bn_ADMM/inference/conv5_1/bn',
    'features.37': 'vgg16_bn_ADMM/inference/conv5_2',
    'features.38': 'vgg16_bn_ADMM/inference/conv5_2/bn',
    'features.40': 'vgg16_bn_ADMM/inference/conv5_3',
    'features.41': 'vgg16_bn_ADMM/inference/conv5_3/bn',
    'classifier': 'vgg16_bn_ADMM/inference/fc1',
}

ResNet18 = {
    'conv1': 'ResNet18/inference/conv1/weight',
    'bn1': 'ResNet18/inference/conv1/bn',

    'layer1.0.conv1': 'ResNet18/inference/conv2_1/weight_1',
    'layer1.0.bn1': 'ResNet18/inference/conv2_1/bn_1',
    'layer1.0.conv2': 'ResNet18/inference/conv2_1/weight_2',
    'layer1.0.bn2': 'ResNet18/inference/conv2_1/bn_2',
    'layer1.1.conv1': 'ResNet18/inference/conv2_2/weight_1',
    'layer1.1.bn1': 'ResNet18/inference/conv2_2/bn_1',
    'layer1.1.conv2': 'ResNet18/inference/conv2_2/weight_2',
    'layer1.1.bn2': 'ResNet18/inference/conv2_2/bn_2',

    'layer2.0.conv1': 'ResNet18/inference/conv3_1/weight_1',
    'layer2.0.bn1': 'ResNet18/inference/conv3_1/bn_1',
    'layer2.0.conv2': 'ResNet18/inference/conv3_1/weight_2',
    'layer2.0.bn2': 'ResNet18/inference/conv3_1/bn_2',
    'layer2.0.shortcut.0': 'ResNet18/inference/conv3_1/weight_shortcut',
    'layer2.0.shortcut.1': 'ResNet18/inference/conv3_1/bn_shortcut',
    'layer2.1.conv1': 'ResNet18/inference/conv3_2/weight_1',
    'layer2.1.bn1': 'ResNet18/inference/conv3_2/bn_1',
    'layer2.1.conv2': 'ResNet18/inference/conv3_2/weight_2',
    'layer2.1.bn2': 'ResNet18/inference/conv3_2/bn_2',

    'layer3.0.conv1': 'ResNet18/inference/conv4_1/weight_1',
    'layer3.0.bn1': 'ResNet18/inference/conv4_1/bn_1',
    'layer3.0.conv2': 'ResNet18/inference/conv4_1/weight_2',
    'layer3.0.bn2': 'ResNet18/inference/conv4_1/bn_2',
    'layer3.0.shortcut.0': 'ResNet18/inference/conv4_1/weight_shortcut',
    'layer3.0.shortcut.1': 'ResNet18/inference/conv4_1/bn_shortcut',
    'layer3.1.conv1': 'ResNet18/inference/conv4_2/weight_1',
    'layer3.1.bn1': 'ResNet18/inference/conv4_2/bn_1',
    'layer3.1.conv2': 'ResNet18/inference/conv4_2/weight_2',
    'layer3.1.bn2': 'ResNet18/inference/conv4_2/bn_2',

    'layer4.0.conv1': 'ResNet18/inference/conv5_1/weight_1',
    'layer4.0.bn1': 'ResNet18/inference/conv5_1/bn_1',
    'layer4.0.conv2': 'ResNet18/inference/conv5_1/weight_2',
    'layer4.0.bn2': 'ResNet18/inference/conv5_1/bn_2',
    'layer4.0.shortcut.0': 'ResNet18/inference/conv5_1/weight_shortcut',
    'layer4.0.shortcut.1': 'ResNet18/inference/conv5_1/bn_shortcut',
    'layer4.1.conv1': 'ResNet18/inference/conv5_2/weight_1',
    'layer4.1.bn1': 'ResNet18/inference/conv5_2/bn_1',
    'layer4.1.conv2': 'ResNet18/inference/conv5_2/weight_2',
    'layer4.1.bn2': 'ResNet18/inference/conv5_2/bn_2',

    'linear': 'ResNet18/inference/fc'
}

ResNet50 = {
    # Input layer
    'module.conv1': 'ResNet50/inference/conv1/weight',
    'module.bn1': 'ResNet50/inference/conv1/bn',

    # Layer-1
    'module.layer1.0.conv1': 'ResNet50/inference/conv2_1/weight_1',
    'module.layer1.0.bn1': 'ResNet50/inference/conv2_1/bn_1',
    'module.layer1.0.conv2': 'ResNet50/inference/conv2_1/weight_2',
    'module.layer1.0.bn2': 'ResNet50/inference/conv2_1/bn_2',
    'module.layer1.0.conv3': 'ResNet50/inference/conv2_1/weight_3',
    'module.layer1.0.bn3': 'ResNet50/inference/conv2_1/bn_3',
    'module.layer1.0.downsample.0': 'ResNet50/inference/conv2_1/weight_shortcut',
    'module.layer1.0.downsample.1': 'ResNet50/inference/conv2_1/bn_shortcut',

    'module.layer1.1.conv1': 'ResNet50/inference/conv2_2/weight_1',
    'module.layer1.1.bn1': 'ResNet50/inference/conv2_2/bn_1',
    'module.layer1.1.conv2': 'ResNet50/inference/conv2_2/weight_2',
    'module.layer1.1.bn2': 'ResNet50/inference/conv2_2/bn_2',
    'module.layer1.1.conv3': 'ResNet50/inference/conv2_2/weight_3',
    'module.layer1.1.bn3': 'ResNet50/inference/conv2_2/bn_3',

    'module.layer1.2.conv1': 'ResNet50/inference/conv2_3/weight_1',
    'module.layer1.2.bn1': 'ResNet50/inference/conv2_3/bn_1',
    'module.layer1.2.conv2': 'ResNet50/inference/conv2_3/weight_2',
    'module.layer1.2.bn2': 'ResNet50/inference/conv2_3/bn_2',
    'module.layer1.2.conv3': 'ResNet50/inference/conv2_3/weight_3',
    'module.layer1.2.bn3': 'ResNet50/inference/conv2_3/bn_3',

    # Layer-2
    'module.layer2.0.conv1': 'ResNet50/inference/conv3_1/weight_1',
    'module.layer2.0.bn1': 'ResNet50/inference/conv3_1/bn_1',
    'module.layer2.0.conv2': 'ResNet50/inference/conv3_1/weight_2',
    'module.layer2.0.bn2': 'ResNet50/inference/conv3_1/bn_2',
    'module.layer2.0.conv3': 'ResNet50/inference/conv3_1/weight_3',
    'module.layer2.0.bn3': 'ResNet50/inference/conv3_1/bn_3',
    'module.layer2.0.downsample.0': 'ResNet50/inference/conv3_1/weight_shortcut',
    'module.layer2.0.downsample.1': 'ResNet50/inference/conv3_1/bn_shortcut',

    'module.layer2.1.conv1': 'ResNet50/inference/conv3_2/weight_1',
    'module.layer2.1.bn1': 'ResNet50/inference/conv3_2/bn_1',
    'module.layer2.1.conv2': 'ResNet50/inference/conv3_2/weight_2',
    'module.layer2.1.bn2': 'ResNet50/inference/conv3_2/bn_2',
    'module.layer2.1.conv3': 'ResNet50/inference/conv3_2/weight_3',
    'module.layer2.1.bn3': 'ResNet50/inference/conv3_2/bn_3',

    'module.layer2.2.conv1': 'ResNet50/inference/conv3_3/weight_1',
    'module.layer2.2.bn1': 'ResNet50/inference/conv3_3/bn_1',
    'module.layer2.2.conv2': 'ResNet50/inference/conv3_3/weight_2',
    'module.layer2.2.bn2': 'ResNet50/inference/conv3_3/bn_2',
    'module.layer2.2.conv3': 'ResNet50/inference/conv3_3/weight_3',
    'module.layer2.2.bn3': 'ResNet50/inference/conv3_3/bn_3',

    'module.layer2.3.conv1': 'ResNet50/inference/conv3_4/weight_1',
    'module.layer2.3.bn1': 'ResNet50/inference/conv3_4/bn_1',
    'module.layer2.3.conv2': 'ResNet50/inference/conv3_4/weight_2',
    'module.layer2.3.bn2': 'ResNet50/inference/conv3_4/bn_2',
    'module.layer2.3.conv3': 'ResNet50/inference/conv3_4/weight_3',
    'module.layer2.3.bn3': 'ResNet50/inference/conv3_4/bn_3',

    # Layer-3
    'module.layer3.0.conv1': 'ResNet50/inference/conv4_1/weight_1',
    'module.layer3.0.bn1': 'ResNet50/inference/conv4_1/bn_1',
    'module.layer3.0.conv2': 'ResNet50/inference/conv4_1/weight_2',
    'module.layer3.0.bn2': 'ResNet50/inference/conv4_1/bn_2',
    'module.layer3.0.conv3': 'ResNet50/inference/conv4_1/weight_3',
    'module.layer3.0.bn3': 'ResNet50/inference/conv4_1/bn_3',
    'module.layer3.0.downsample.0': 'ResNet50/inference/conv4_1/weight_shortcut',
    'module.layer3.0.downsample.1': 'ResNet50/inference/conv4_1/bn_shortcut',

    'module.layer3.1.conv1': 'ResNet50/inference/conv4_2/weight_1',
    'module.layer3.1.bn1': 'ResNet50/inference/conv4_2/bn_1',
    'module.layer3.1.conv2': 'ResNet50/inference/conv4_2/weight_2',
    'module.layer3.1.bn2': 'ResNet50/inference/conv4_2/bn_2',
    'module.layer3.1.conv3': 'ResNet50/inference/conv4_2/weight_3',
    'module.layer3.1.bn3': 'ResNet50/inference/conv4_2/bn_3',

    'module.layer3.2.conv1': 'ResNet50/inference/conv4_3/weight_1',
    'module.layer3.2.bn1': 'ResNet50/inference/conv4_3/bn_1',
    'module.layer3.2.conv2': 'ResNet50/inference/conv4_3/weight_2',
    'module.layer3.2.bn2': 'ResNet50/inference/conv4_3/bn_2',
    'module.layer3.2.conv3': 'ResNet50/inference/conv4_3/weight_3',
    'module.layer3.2.bn3': 'ResNet50/inference/conv4_3/bn_3',

    'module.layer3.3.conv1': 'ResNet50/inference/conv4_4/weight_1',
    'module.layer3.3.bn1': 'ResNet50/inference/conv4_4/bn_1',
    'module.layer3.3.conv2': 'ResNet50/inference/conv4_4/weight_2',
    'module.layer3.3.bn2': 'ResNet50/inference/conv4_4/bn_2',
    'module.layer3.3.conv3': 'ResNet50/inference/conv4_4/weight_3',
    'module.layer3.3.bn3': 'ResNet50/inference/conv4_4/bn_3',

    'module.layer3.4.conv1': 'ResNet50/inference/conv4_5/weight_1',
    'module.layer3.4.bn1': 'ResNet50/inference/conv4_5/bn_1',
    'module.layer3.4.conv2': 'ResNet50/inference/conv4_5/weight_2',
    'module.layer3.4.bn2': 'ResNet50/inference/conv4_5/bn_2',
    'module.layer3.4.conv3': 'ResNet50/inference/conv4_5/weight_3',
    'module.layer3.4.bn3': 'ResNet50/inference/conv4_5/bn_3',

    'module.layer3.5.conv1': 'ResNet50/inference/conv4_6/weight_1',
    'module.layer3.5.bn1': 'ResNet50/inference/conv4_6/bn_1',
    'module.layer3.5.conv2': 'ResNet50/inference/conv4_6/weight_2',
    'module.layer3.5.bn2': 'ResNet50/inference/conv4_6/bn_2',
    'module.layer3.5.conv3': 'ResNet50/inference/conv4_6/weight_3',
    'module.layer3.5.bn3': 'ResNet50/inference/conv4_6/bn_3',

    # Layer-4
    'module.layer4.0.conv1': 'ResNet50/inference/conv5_1/weight_1',
    'module.layer4.0.bn1': 'ResNet50/inference/conv5_1/bn_1',
    'module.layer4.0.conv2': 'ResNet50/inference/conv5_1/weight_2',
    'module.layer4.0.bn2': 'ResNet50/inference/conv5_1/bn_2',
    'module.layer4.0.conv3': 'ResNet50/inference/conv5_1/weight_3',
    'module.layer4.0.bn3': 'ResNet50/inference/conv5_1/bn_3',
    'module.layer4.0.downsample.0': 'ResNet50/inference/conv5_1/weight_shortcut',
    'module.layer4.0.downsample.1': 'ResNet50/inference/conv5_1/bn_shortcut',

    'module.layer4.1.conv1': 'ResNet50/inference/conv5_2/weight_1',
    'module.layer4.1.bn1': 'ResNet50/inference/conv5_2/bn_1',
    'module.layer4.1.conv2': 'ResNet50/inference/conv5_2/weight_2',
    'module.layer4.1.bn2': 'ResNet50/inference/conv5_2/bn_2',
    'module.layer4.1.conv3': 'ResNet50/inference/conv5_2/weight_3',
    'module.layer4.1.bn3': 'ResNet50/inference/conv5_2/bn_3',

    'module.layer4.2.conv1': 'ResNet50/inference/conv5_3/weight_1',
    'module.layer4.2.bn1': 'ResNet50/inference/conv5_3/bn_1',
    'module.layer4.2.conv2': 'ResNet50/inference/conv5_3/weight_2',
    'module.layer4.2.bn2': 'ResNet50/inference/conv5_3/bn_2',
    'module.layer4.2.conv3': 'ResNet50/inference/conv5_3/weight_3',
    'module.layer4.2.bn3': 'ResNet50/inference/conv5_3/bn_3',

    # Layer FC
    'module.fc': 'ResNet50/inference/fc'
}

WRN_28_4 = {
    'conv1': 'WRN_28_4/inference/conv1/weight',

    'block1.layer.0.bn1': 'WRN_28_4/inference/conv2_1/bn_1',
    'block1.layer.0.conv1': 'WRN_28_4/inference/conv2_1/weight_1',
    'block1.layer.0.bn2': 'WRN_28_4/inference/conv2_1/bn_2',
    'block1.layer.0.conv2': 'WRN_28_4/inference/conv2_1/weight_2',
    'block1.layer.0.convShortcut': 'WRN_28_4/inference/conv2_1/weight_shortcut',
    'block1.layer.1.bn1': 'WRN_28_4/inference/conv2_2/bn_1',
    'block1.layer.1.conv1': 'WRN_28_4/inference/conv2_2/weight_1',
    'block1.layer.1.bn2': 'WRN_28_4/inference/conv2_2/bn_2',
    'block1.layer.1.conv2': 'WRN_28_4/inference/conv2_2/weight_2',
    'block1.layer.2.bn1': 'WRN_28_4/inference/conv2_3/bn_1',
    'block1.layer.2.conv1': 'WRN_28_4/inference/conv2_3/weight_1',
    'block1.layer.2.bn2': 'WRN_28_4/inference/conv2_3/bn_2',
    'block1.layer.2.conv2': 'WRN_28_4/inference/conv2_3/weight_2',
    'block1.layer.3.bn1': 'WRN_28_4/inference/conv2_4/bn_1',
    'block1.layer.3.conv1': 'WRN_28_4/inference/conv2_4/weight_1',
    'block1.layer.3.bn2': 'WRN_28_4/inference/conv2_4/bn_2',
    'block1.layer.3.conv2': 'WRN_28_4/inference/conv2_4/weight_2',

    'block2.layer.0.bn1': 'WRN_28_4/inference/conv3_1/bn_1',
    'block2.layer.0.conv1': 'WRN_28_4/inference/conv3_1/weight_1',
    'block2.layer.0.bn2': 'WRN_28_4/inference/conv3_1/bn_2',
    'block2.layer.0.conv2': 'WRN_28_4/inference/conv3_1/weight_2',
    'block2.layer.0.convShortcut': 'WRN_28_4/inference/conv3_1/weight_shortcut',
    'block2.layer.1.bn1': 'WRN_28_4/inference/conv3_2/bn_1',
    'block2.layer.1.conv1': 'WRN_28_4/inference/conv3_2/weight_1',
    'block2.layer.1.bn2': 'WRN_28_4/inference/conv3_2/bn_2',
    'block2.layer.1.conv2': 'WRN_28_4/inference/conv3_2/weight_2',
    'block2.layer.2.bn1': 'WRN_28_4/inference/conv3_3/bn_1',
    'block2.layer.2.conv1': 'WRN_28_4/inference/conv3_3/weight_1',
    'block2.layer.2.bn2': 'WRN_28_4/inference/conv3_3/bn_2',
    'block2.layer.2.conv2': 'WRN_28_4/inference/conv3_3/weight_2',
    'block2.layer.3.bn1': 'WRN_28_4/inference/conv3_4/bn_1',
    'block2.layer.3.conv1': 'WRN_28_4/inference/conv3_4/weight_1',
    'block2.layer.3.bn2': 'WRN_28_4/inference/conv3_4/bn_2',
    'block2.layer.3.conv2': 'WRN_28_4/inference/conv3_4/weight_2',

    'block3.layer.0.bn1': 'WRN_28_4/inference/conv4_1/bn_1',
    'block3.layer.0.conv1': 'WRN_28_4/inference/conv4_1/weight_1',
    'block3.layer.0.bn2': 'WRN_28_4/inference/conv4_1/bn_2',
    'block3.layer.0.conv2': 'WRN_28_4/inference/conv4_1/weight_2',
    'block3.layer.0.convShortcut': 'WRN_28_4/inference/conv4_1/weight_shortcut',
    'block3.layer.1.bn1': 'WRN_28_4/inference/conv4_2/bn_1',
    'block3.layer.1.conv1': 'WRN_28_4/inference/conv4_2/weight_1',
    'block3.layer.1.bn2': 'WRN_28_4/inference/conv4_2/bn_2',
    'block3.layer.1.conv2': 'WRN_28_4/inference/conv4_2/weight_2',
    'block3.layer.2.bn1': 'WRN_28_4/inference/conv4_3/bn_1',
    'block3.layer.2.conv1': 'WRN_28_4/inference/conv4_3/weight_1',
    'block3.layer.2.bn2': 'WRN_28_4/inference/conv4_3/bn_2',
    'block3.layer.2.conv2': 'WRN_28_4/inference/conv4_3/weight_2',
    'block3.layer.3.bn1': 'WRN_28_4/inference/conv4_4/bn_1',
    'block3.layer.3.conv1': 'WRN_28_4/inference/conv4_4/weight_1',
    'block3.layer.3.bn2': 'WRN_28_4/inference/conv4_4/bn_2',
    'block3.layer.3.conv2': 'WRN_28_4/inference/conv4_4/weight_2',

    'bn1': 'WRN_28_4/inference/fc/bn',
    'fc': 'WRN_28_4/inference/fc'
}


def match_var(torch_v_name, torch_dict, model_name):

    if torch_v_name.find('module.basic_model') != -1:
        torch_lname = '.'.join(torch_v_name.split('.')[2:-1])
    else:
        torch_lname = '.'.join(torch_v_name.split('.')[:-1])

    v_name = torch_v_name.split('.')[-1]

    torch_v = torch_dict[torch_v_name]
    out_v = torch_v.numpy()

    if model_name == 'vgg16_bn_Hydra':
        torch_tf_dict = VGG16_Hydra
    elif model_name == 'vgg16_bn_ADMM':
        torch_tf_dict = VGG16_ADMM
    elif model_name == 'ResNet18':
        torch_tf_dict = ResNet18
    elif model_name == 'ResNet50':
        torch_tf_dict = ResNet50
    elif model_name == 'WRN_28_4':
        torch_tf_dict = WRN_28_4
    else:
        raise NameError(f'{model_name} is not supported in torch_2_tf.')

    tf_lname = torch_tf_dict[torch_lname]

    is_found = True

    if v_name == 'weight':
        if tf_lname.find('/bn') != -1:
            out_lname = '/'.join([tf_lname, 'gamma:0'])
        else:
            if out_v.shape.__len__() == 4:
                out_v = np.transpose(out_v, (2,3,1,0))
            elif out_v.shape.__len__() == 2:
                out_v = np.transpose(out_v, (1,0))

            if tf_lname.find('weight') == -1:
                out_lname = '/'.join([tf_lname, 'weight:0'])
            else:
                out_lname = tf_lname+':0'

    elif v_name == 'bias':
        if tf_lname.find('/bn') != -1:
            out_lname = '/'.join([tf_lname, 'beta:0'])
        else:
            if tf_lname.find('bias') == -1:
                out_lname = '/'.join([tf_lname, 'bias:0'])
            else:
                out_lname = tf_lname+':0'

    elif v_name == 'running_mean':
        if tf_lname.find('/bn') != -1:
            out_lname = '/'.join([tf_lname, 'moving_mean:0'])
        else:
            raise NameError(f'"{torch_v_name}" not exists in TF model {model_name}.')

    elif v_name == 'running_var':
        if tf_lname.find('/bn') != -1:
            out_lname = '/'.join([tf_lname, 'moving_variance:0'])
        else:
            raise NameError(f'"{torch_v_name}" not exists in TF model {model_name}.')

    else:
        is_found = False
        out_lname = None

    out_tf_var = {out_lname: [torch_v_name, out_v]}

    return out_tf_var, is_found

