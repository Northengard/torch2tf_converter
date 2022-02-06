import torch


def has_children(module):
    return len(list(module.children())) > 0


def get_conv_bn_names(module):
    conv_bn_names = list()
    modules = list(module.named_modules())
    for mod_id in range(len(modules) - 1):
        if has_children(modules[mod_id][1]):
            continue
        if ('conv' in str(modules[mod_id][1]).split('(')[0].lower()) and \
                ('batchnorm' in str(modules[mod_id + 1][1]).split('(')[0].lower()):
            conv_bn_names.append([modules[mod_id][0], modules[mod_id + 1][0]])
    return conv_bn_names


def fuse(model):
    conv_bn_to_fuze = get_conv_bn_names(model)
    for layer_names in conv_bn_to_fuze:
        torch.quantization.fuse_modules(model, modules_to_fuse=layer_names, inplace=True)
