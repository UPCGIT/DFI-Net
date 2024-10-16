from .NEWdlst import NEWdlst


def get_model(model_name, dataset_name, patch_size):
    # example: model_name='cnn3d', dataset_name='pu'
    if model_name == 'NEWdlst':
        model = NEWdlst(dataset_name, patch_size)
    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model


