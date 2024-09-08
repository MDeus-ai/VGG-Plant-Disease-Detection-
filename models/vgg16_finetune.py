#
import torch
from torchsummary import summary
from torchvision import models

# Function used for freezing and unfreezing VGG16 convolutional and fc layers
def vgg_tunner(unfreeze_features, unfreeze_classifier, weights_path, index=None, indexing=False):

    # Loading The Model And The Model Weights
    model = models.vgg16()
    model.load_state_dict(torch.load(weights_path))

    # Initially All Model Parameters Frozen
    for parameters in model.parameters():
        parameters.requires_grad = False

    # Change Softmax layer To 8 Output Classes
    # model.classifier[6] = nn.Linear(4096, out_features=8)

    if unfreeze_features and not unfreeze_classifier:
        if not indexing:
            for params in model.features.parameters():
                params.requires_grad = True
        elif indexing:
            for layer in index:
                for params in model.features[layer].parameters():
                    params.requires_grad = True

    if unfreeze_classifier and not unfreeze_features:
        if not indexing:
            for params in model.classifier.parameters():
                params.requires_grad = True
        elif indexing:
            for layer in index:
                for params in model.classifier[layer].parameters():
                    params.requires_grad = True


    print(model)
    print(summary(model))
    return model

