import copy


def get_model(model_dict):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    model = model(name=name, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            #TODO ADd models
        }[name]
    except:
        raise NotImplementedError("Model {} not available".format(name))