#
from torchvision.models import vgg19
import torch
import datetime
from utils.dataloaders import get_trainDataLoader, get_validDataLoader
import torch.optim as optim
from vgg16_finetune import vgg_tunner

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f'Training On {device}')

vgg16_8 = vgg_tunner(unfreeze_classifier=True, unfreeze_features=False)
vgg16_8 = vgg16_8.to(device=device)






def loop(model, loss_fn, num_epochs, optimizer, train_dataloader, test_dataloader, learning_rate):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total = 0


        model.train()
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model.forward(features)

            optimizer.zero_grad()

            loss_fn(outputs, labels)
            loss_fn.backward()
            optimizer.step()

            running_loss = + loss_fn.item()

            _, preds = torch.max(outputs, 1)
            correct_preds = + torch.sum(preds==labels.data)
            total += labels.size(0)

            epoch_accuracy = correct_preds.double() / total



        # Statistics
        time = datetime.datetime.now()
        print(f'{time} {epoch+1}/{num_epochs} Loss: {running_loss/len(train_dataloader)}')






