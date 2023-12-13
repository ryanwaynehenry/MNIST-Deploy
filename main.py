import pytorch_lightning
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
from PIL import Image

BATCH_SIZE = 256
WORKER_SIZE = 4
USE_EXISTING_MODEL = True
RUN_SINGLE_OUTPUT = True


class CNNClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_predictions = []
        self.test_loss = []
        self.test_labels = []
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 32, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return self.fc2(out)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        accuracy = self.calculate_accuracy(outputs, labels)

        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('crossEntropy', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        accuracy = self.calculate_accuracy(outputs, labels)

        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('crossEntropy', loss, on_step=True, on_epoch=True)


    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        predicted_labels = torch.argmax(outputs, dim=1)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        accuracy = self.calculate_accuracy(outputs, labels)
        self.test_loss.append(loss)
        self.test_labels.append(labels)
        self.test_predictions.append(predicted_labels)

        self.log('crossEntropy', loss, on_step=True, on_epoch=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return {'predictions': predicted_labels, 'labels': labels, 'test_loss': loss, 'test_accuracy': accuracy}

    def on_test_epoch_end(self):

        all_predictions = torch.cat(self.test_predictions, dim=0)
        all_labels = torch.cat(self.test_labels, dim=0)

        self.test_predictions = all_predictions
        self.test_labels = all_labels

        # Save predictions and labels
        torch.save({'predictions': all_predictions, 'labels': all_labels}, 'test_results.pt')
    def calculate_accuracy(self, y_pred, y_act):
        y_pred = torch.argmax(y_pred, dim=1)

        correct_pred = (y_pred == y_act).float()
        accuracy = correct_pred.sum() / len(correct_pred)

        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def prepare_data(self) -> None:
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        mnist_train = datasets.MNIST(os.getcwd(), train=True, transform=transform)
        mnist_test = datasets.MNIST(os.getcwd(), train=True, transform=transform)
        mnist_train, mnist_val = random_split(mnist_train, [0.8, 0.2])

        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_SIZE,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER_SIZE,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER_SIZE)

def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def visualize_filters(model):
    filters = model.layer1[0].weight.data.clone()
    #filters = model.layer2[0].weight.data.clone()
    n_filters = filters.shape[0]
    plt.figure(figsize=(8, 6))
    for i in range(n_filters):
        plt.subplot(4, 4, i + 1)
        #plt.subplot(4, 4, i + 1)
        plt.imshow(filters[i, 0, :, :], cmap='gray')
        plt.axis('off')
    plt.show()

def visualize_feature_maps(model, dataloader):
    # Function to visualize feature maps
    def forward_hook(module, input, output):
        # Store the output for visualization
        model.feature_maps.append(output)

    # Register hooks to all convolutional layers
    hooks = []
    for layer in [model.layer1, model.layer2]:
        hooks.append(layer[0].register_forward_hook(forward_hook))

    # Get a batch of data
    images, _ = next(iter(dataloader))
    model.feature_maps = []
    with torch.no_grad():
        model(images[:1])  # Pass one image through the network

    # Plot feature maps
    for layer_maps in model.feature_maps:
        n_maps = layer_maps.shape[1]
        plt.figure(figsize=(8, 6))
        for i in range(n_maps):
            if n_maps <= 16:
                plt.subplot(4, 4, i + 1)
            else:
                plt.subplot(6, 6, i + 1)
            plt.imshow(layer_maps[0, i, :, :].cpu(), cmap='gray')
            plt.axis('off')
        plt.show()

    # Remove hooks (to avoid memory leaks)
    for hook in hooks:
        hook.remove()

def load_and_transform_image(image_path):
    transform = transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image  = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    modelWeightsFile = 'model_weights.pth'

    if RUN_SINGLE_OUTPUT:
        image_path = 'path'
        model = CNNClassifier()
        model.load_state_dict(torch.load(modelWeightsFile))
        model.eval()

        image_tensor = load_and_transform_image(image_path)
        with torch.no_grad():
            output = model(image_tensor)
            predicted_label = torch.argmax(output, dim=1)
            print(f"Predicted Label: {predicted_label.item()}")


    early_stop_callback = pytorch_lightning.callbacks.EarlyStopping(
        monitor='crossEntropy', patience=2, strict=False, verbose=True, mode='min')

    model = CNNClassifier()
    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=10, callbacks=[early_stop_callback])

    if not USE_EXISTING_MODEL:
        trainer.fit(model)
        torch.save(model.state_dict(), modelWeightsFile)

    else:
        model.load_state_dict(torch.load(modelWeightsFile))
        model.eval()

    trainer.test(model)

    visualize_filters(model)
    model.setup()
    visualize_feature_maps(model, model.train_dataloader())

    #test_labels = model.test_labels.cpu()
    #test_predictions = model.test_predictions.cpu()
    #cm = confusion_matrix(test_labels, test_predictions)
    #classes = list(range(10))
    #plot_confusion_matrix(cm, classes)



