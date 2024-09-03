import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ResNet34Custom(nn.Module):
    def __init__(self, num_freeze_layers=None):
        super(ResNet34Custom, self).__init__()
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet34.children())[:-1])
        if num_freeze_layers is not None:
            self.freeze_layers(freeze_layers=num_freeze_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        return x

    def freeze_layers(self, freeze_layers):
        # Freeze the early layers up to freeze_layers
        for layer in list(self.features.children())[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False


class ResNet50Custom(nn.Module):
    def __init__(self, output_size=512, num_freeze_layers=None):
        super(ResNet50Custom, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.fc = nn.Linear(2048, out_features=output_size)
        if num_freeze_layers:
            self.freeze_layers(freeze_layers=num_freeze_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    def freeze_layers(self, freeze_layers):
        # Freeze the early layers up to freeze_layers
        for layer in list(self.features.children())[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False


class ResNet18Custom(nn.Module):
    def __init__(self, output_size=512, num_freeze_layers=None):
        super(ResNet18Custom, self).__init__()
        resnet50 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        if num_freeze_layers:
            self.freeze_layers(freeze_layers=num_freeze_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        return x

    def freeze_layers(self, freeze_layers):
        # Freeze the early layers up to freeze_layers
        for layer in list(self.features.children())[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

class VGG19Custom(nn.Module):
    def __init__(self, output_size):
        super(VGG19Custom, self).__init__()
        vgg19 = models.vgg19(weights=True)

        # Extract features from the model (excluding the last fully connected layer)
        # print('len current model', len(list(vgg19.children())[:]))
        # print('len last layer', len(list(vgg19.children())[-1]))
        self.features = nn.Sequential(*list(vgg19.children())[:-1])
        self.features2 = list(vgg19.children())[-1][:5]
        # self.features = nn.Sequential(*(list(vgg19.children())[:-1] + list(list(vgg19.children())[-1][:4])))
        # print('current size ', self.features)
        self.fc = nn.Linear(4096, out_features=output_size)
        # list(vgg19.children())[-1][:4]

    def forward(self, x):
        # Forward pass through the features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.features2(x)
        x = self.fc(x)
        return x


class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(OneDimensionalCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x1 = self.relu(x1)
        x1 = self.adaptive_pool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc(x1)
        return x1


class OneDimensionalCNNVClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNNVClassifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        # remove the effect of the padding
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        x2 = self.relu(x1)
        x3 = self.adaptive_pool(x2)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.fc(x3)
        return x3, x1


class FCNetClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=.1)
        self.fc2 = nn.Linear(512, 256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=.1)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        output = self.softmax(x)
        return output


class OneDimensionalCNNSmall(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, feature_size):
        super(OneDimensionalCNNSmall, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(out_channels * ((input_size - kernel_size + 1) // 2), feature_size)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x1 = self.conv(x)
        if list_length is not None:
            for item_idx in range(x.shape[0]):
                x1[item_idx, :, list_length[item_idx]:] = 0
        return x1


class FCNet(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, feature_size)

    def forward(self, out, skip=False):
        out = out.to(torch.float32)
        if not skip:
            out = out.view(out.size(0), -1)
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        return out


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_size=224, latent_dim=200):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.latents_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(256 * (self.input_size // 8) * (self.input_size // 8), latent_dim)
        self.fc2 = nn.Linear(256 * (self.input_size // 8) * (self.input_size // 8), latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256 * (input_size // 8) * (input_size // 8))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = z.view(z.size(0), 256, (self.input_size // 8), (self.input_size // 8))
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        return torch.sigmoid(self.deconv3(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
