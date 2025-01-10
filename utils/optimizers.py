import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("gsd-sde")

name2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}

def get_optimizer(opt):
    if opt["training"]["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD

    else:
        opt_name = opt["training"]["optimizer"]["name"]
        if opt_name not in name2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        logger.info("Using {} optimizer".format(opt_name))
        return name2opt[opt_name]