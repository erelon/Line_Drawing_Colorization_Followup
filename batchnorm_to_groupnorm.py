import torch
from torch import nn

from models import siggraph17_L


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


if __name__ == '__main__':

    model = siggraph17_L(128, norm_layer="bn", pretrained_path="model_e0_batch_19000.pt")
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Get current bn layer
            bn = get_layer(model, name)
            # Create new gn layer
            gn = nn.GroupNorm(1, bn.num_features)
            # Assign gn
            print("Swapping {} with {}".format(bn, gn))

            set_layer(model, name, gn)

    # torch.save(model.state_dict(), f"model_e{0}_batch_{19000}_gn.pt")
