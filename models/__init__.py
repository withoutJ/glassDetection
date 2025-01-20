import copy

from models.glass_detection_with_depth import glass_detection_with_depth


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
            "glass_detection_with_depth": glass_detection_with_depth
        }[name]
    except:
        raise NotImplementedError("Model {} not available".format(name))