import os

os.system('pip install ../input/effnet-and-sklearn/scikit_learn-0.22.1-cp37-cp37m-manylinux1_x86_64.whl')
# os.system('pip install efficientnet_pytorch')

import torch
# from __future__ import division, print_function
import numpy as np
import torch.nn as nn
from sklearn.utils.linear_assignment_ import linear_assignment
import os
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from efficientnet_pytorch import EfficientNet
import cv2
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.model_selection import train_test_split
import plotly.express as px
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.transform import rescale
from sklearn.decomposition import PCA
import pickle
import gc


def getImage(path):
    img = plt.imread(path)
    img = rescale(img, (0.25, 0.25, 1))
    return OffsetImage(img)


def save_tsne(args, save_path, perplexities, model, train_loader, limit=50):
    train_inds, val_inds = train_test_split(
        np.arange(train_loader.dataset.dataDF.shape[0]),
        train_size=limit,
        stratify=train_loader.dataset.dataDF.diagnosis
    )
    backup_df = train_loader.dataset.dataDF.copy()
    train_loader.dataset.dataDF = train_loader.dataset.dataDF.iloc[train_inds, :]
    train_loader.dataset.return_image_name = True

    activations = torch.zeros([len(train_loader.dataset), args.n_clusters]).to(args.device)
    targets = list()
    image_names = list()
    model_output = args.output_vector_length
    for x, labels, idxs, names in train_loader:
        feat_1000, feat_1280 = model(x.to(args.device))

        if model_output == 1280:
            feat = feat_1280
        elif model_output == 1000:
            feat = feat_1000

        activations[idxs, :] = feat
        targets.extend(labels.cpu().tolist())
        image_names.extend(names)

        gc.collect()

    colors = np.ones([activations.shape[0], 3])
    colors[:, 0] = targets
    colors[:, 0] /= max(targets)

    from matplotlib.colors import hsv_to_rgb

    for perplexity in perplexities:
        x = TSNE().fit_transform(activations.detach().cpu().numpy())
        fig = px.scatter(
            x=x[:, 0],
            y=x[:, 1],
            hover_name=image_names,
            color=targets,
            color_continuous_scale=px.colors.sequential.Jet)
        fig.write_html(str(save_path / f'perplexity_{perplexity}.html'))

        fig, ax = plt.subplots(figsize=(20, 15))
        for i in range(x.shape[0]):
            ab = AnnotationBbox(getImage(image_names[i]), (x[i, 0], x[i, 1]), frameon=False)
            ax.add_artist(ab)

        scatter = plt.scatter(x[:, 0], x[:, 1], c=targets, cmap='jet', label=targets)

        plt.legend(*scatter.legend_elements())

        plt.savefig(save_path / f'perplexity_{perplexity}.jpg')
        plt.close()
    train_loader.dataset.dataDF = backup_df
    del activations
    gc.collect()


def save_pca(args, save_path, perplexities, model, train_loader, limit=50):
    train_inds, val_inds = train_test_split(
        np.arange(train_loader.dataset.dataDF.shape[0]),
        train_size=limit,
        stratify=train_loader.dataset.dataDF.diagnosis
    )
    backup_df = train_loader.dataset.dataDF.copy()
    train_loader.dataset.dataDF = train_loader.dataset.dataDF.iloc[train_inds, :]
    train_loader.dataset.return_image_name = True

    activations = torch.zeros([len(train_loader.dataset), args.n_clusters]).cpu()
    targets = list()
    image_names = list()
    model_output = args.output_vector_length
    print(activations.device)
    for x, labels, idxs, names in train_loader:
        feat_1000, feat_1280 = model(x.to(args.device))

        if model_output == 1280:
            feat = feat_1280
        elif model_output == 1000:
            feat = feat_1000

        activations[idxs, :] = feat.detach().cpu()
        targets.extend(labels.cpu().tolist())
        image_names.extend(names)
        del feat_1000, feat_1280, x
        gc.collect()

    colors = np.ones([activations.shape[0], 3])
    colors[:, 0] = targets
    colors[:, 0] /= max(targets)

    pca = PCA(n_components=2)
    activations = pca.fit_transform(activations.detach().cpu())
    fig = px.scatter(
        x=activations[:, 0],
        y=activations[:, 1],
        hover_name=image_names,
        color=targets,
        color_continuous_scale=px.colors.sequential.Jet)
    fig.write_html(str(save_path / f'pca.html'))

    fig, ax = plt.subplots(figsize=(20, 15))
    for i in range(activations.shape[0]):
        ab = AnnotationBbox(getImage(image_names[i]), (activations[i, 0], activations[i, 1]), frameon=False)
        ax.add_artist(ab)

    mins = activations.min(0)
    maxs = activations.max(0)

    plt.xlim([mins[0], maxs[0]])
    plt.ylim([mins[1], maxs[1]])
    # scatter = plt.scatter(activations[:, 0], activations[:, 1], c=targets, cmap='jet', label=targets)

    # plt.legend(*scatter.legend_elements())

    with open(save_path / 'pca_fig.pkl', 'wb') as f:
        pickle.dump({'fig': fig}, f)

    plt.savefig(save_path / f'pca.jpg')
    plt.close()
    train_loader.dataset.dataDF = backup_df


def feat2prob(feat, center, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(feat.unsqueeze(1) - center, 2), 2) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length):
        self.count = length
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        # self.count = 0

    def update(self, val, n=1):
        self.val = val
        # self.sum += val * n
        # self.count += n
        # self.avg = self.sum / self.count
        self.avg += self.val / self.count


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


# ramps
# Functions for ramping hyperparameters up or down

# Each function takes the current training step or epoch, and the
# ramp length in the same format, and returns a multiplier between
# 0 and 1.


class RetinaDataset(Dataset):
    def __init__(self, master_dir, dataset_name, transforms=None, dataDF=None, num_transforms=None, pi_augments=None):
        super().__init__()
        self.master_dir = Path(master_dir)
        self.dataset_name = dataset_name
        if dataDF is None:
            self.dataDF = pd.read_csv(self.master_dir / f'{dataset_name}.csv')
        else:
            self.dataDF = dataDF

        self.transforms = transforms
        self.num_transforms = num_transforms
        self.return_original = False
        self.pi_augments = pi_augments
        self.return_image_name = False

    def __getitem__(self, item):
        path = Path(self.master_dir) / self.dataset_name / (self.dataDF.iloc[item, 0] + '.png')
        if self.dataset_name == 'test':
            path = Path('../input/aptos2019-blindness-detection/test_images') / (self.dataDF.iloc[item, 0] + '.png')

        if not path.exists():
            raise FileNotFoundError(f'No such path {path}.\n\n\n')

        img = cv2.imread(str(path))  # Image.open(path)
        label = self.dataDF.iloc[item, -1]

        # ######################################################################### What about the metadata?

        if self.transforms is not None:
            img_transformed = self.transforms(img)
        if self.num_transforms is not None:
            if self.num_transforms != 2:
                raise ValueError("self.num_transforms could be either None or 2")
            img_transformed = self.transforms(img_transformed)

        if self.dataset_name == 'test':
            return img_transformed, -1, item

        if self.return_original:
            if self.pi_augments is not None:
                img = self.pi_augments(img)
            return img_transformed, label, item, img

        if self.return_image_name:
            return img_transformed, label, item, str(path)  # .parts[-1]

        return img_transformed, label, item

    def __len__(self):
        return self.dataDF.shape[0]


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


class Model(nn.Module):

    def __init__(self, efficient_net_type='efficientnet-b0', num_classes=None):
        super().__init__()
        self.efficient_net_type = efficient_net_type

        self.efficient_net = EfficientNet.from_pretrained(self.efficient_net_type)
        if num_classes is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = nn.Linear(1000, num_classes)

        self.last = nn.Identity()  # ######################################### This will become the PCA layer
        self.activations = list()
        self.mode = 'classification'
        self.set_hooks()

    def forward_activations(self, x):
        self.activations = list()  # self.activations[0]: 1x1280
        gc.collect()
        x = self.efficient_net(x)  # 1x1000
        #
        #         self.activations[0] = self.last(self.activations[0])
        x = self.last(x)

        return x, self.activations[0]

    def forward_classification(self, x):
        x = self.efficient_net(x)
        x = self.output_linear(x)

        return x

    def forward(self, x):
        if self.mode == 'classification':
            return self.forward_classification(x)
        elif self.mode == 'activations':
            return self.forward_activations(x)
        else:
            raise NotImplemented('Model recieved unknown mode')

    def get_image_size(self):
        return self.efficient_net.get_image_size(self.efficient_net_type)

    def set_hooks(self):
        def hook(module, inp, out):
            self.activations.append(out.squeeze())

        self.efficient_net._modules['_avg_pooling'].register_forward_hook(hook)


def get_current_time_as_string():
    return datetime.datetime.now().strftime("%B_%d_%Y_%I_%M%p")


class VGG(nn.Module):
    cfg = {
        '4+2': [64, 'M', 128, 'M', 256, 'M', 256, 'M'],
        '5+1': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    }

    # these two 6-layer variants of VGG are following https://arxiv.org/abs/1711.10125 and https://github.com/GT-RIPL/L2C/blob/master/models/vgg.py
    def __init__(self, n_layer, out_dim=10, in_channels=3, img_sz=32):
        super(VGG, self).__init__()
        self.conv_func = nn.Conv2d
        self.features = self._make_layers(VGG.cfg[n_layer], in_channels)
        if n_layer == '4+2':
            self.feat_map_sz = img_sz // 16
            feat_dim = 256 * (self.feat_map_sz ** 2)
            self.last = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.BatchNorm1d(feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 2, out_dim)
            )
            self.last.in_features = feat_dim
        elif n_layer == '5+1':
            self.feat_map_sz = img_sz // 32
            self.last = nn.Linear(512 * (self.feat_map_sz ** 2), out_dim)

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.last(x)
        return x, out
