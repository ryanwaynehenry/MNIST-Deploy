import pytorch_lightning
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer
import os
from PIL import Image
from flask import Flask, jsonify, request, render_template, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

BATCH_SIZE = 256
WORKER_SIZE = 4
USE_EXISTING_MODEL = True
RUN_SINGLE_OUTPUT = True

UPLOAD_FOLDER = 'static' + os.sep + 'uploads'
#UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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



@app.route('/upload', methods=['GET', 'POST'])
def load_and_transform_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        origImage = Image.open(filepath)
        image = origImage.convert('L')
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        image = image.resize((28, 28))
        tensor_image = transform(image).unsqueeze(0)
        predicted_label = run_single_image(tensor_image)
        image_url = os.path.join('uploads', filename)
        image_url = image_url.replace('\\', '/')
        print(filepath)

        return render_template('result.html', image_url=image_url, label=predicted_label.item())

def run_single_image(tensor_image):

    with torch.no_grad():
        output = model(tensor_image)
        predicted_label = torch.argmax(output, dim=1)
        return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    modelWeightsFile = 'model_weights.pth'

    model = CNNClassifier()
    model.load_state_dict(torch.load(modelWeightsFile))
    model.eval()

    if RUN_SINGLE_OUTPUT:
        app.run(debug=True)


