import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import cv2
import os
os.system('pip install efficientnet_pytorch')

import datetime
import pickle
from PIL import Image


def get_current_time_as_string():
    return datetime.datetime.now().strftime("%B_%d_%Y_%I_%M%p")


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
        self.activations = list()
        x = self.efficient_net(x)

        self.activations[0] = self.last(self.activations[0])

        return x, self.activations

    def forward_classification(self, x):
        x = self.efficient_net(x)
        x = self.output_linear(x)

        self.activations = list()  # No need for this right now.....
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


class MelanomaDataset(Dataset):
    def __init__(self, master_dir, dataset_name, transforms=None, dataDF=None):
        super().__init__()
        self.master_dir = Path(master_dir)
        self.dataset_name = dataset_name
        if dataDF is None:
            self.dataDF = pd.read_csv(self.master_dir / f'{dataset_name}.csv')
        else:
            self.dataDF = dataDF

        self.transforms = transforms

    def __getitem__(self, item):
        path = Path(self.master_dir) / self.dataset_name / (self.dataDF.iloc[item, 0] + '.jpg')

        if not path.exists():
            raise FileNotFoundError(f'No such path {path}.\n\n\n')

        img = cv2.imread(str(path))  # Image.open(path)
        label = self.dataDF.iloc[item, -1]

        # ######################################################################### What about the metadata?

        if self.transforms is not None:
            img = self.transforms(img)

        if self.dataset_name == 'test':
            return img
        return img, label

    def __len__(self):
        return self.dataDF.shape[0]


def get_train_val_inds(trainvalDF, train_frac_patients):
    """
    Given dataframe trainvalDF, return a division to a trainDF and a valDf using patients such that no single patient
    will go to both groups.

    :param trainvalDF:
        pandas dataframe.
        must contain a 'patient_id' column
    :param train_frac_patients:
        float between 0 and 1.
        the size of the train group in fraction
    :return:
        indices in the dataframe of IMAGES.

        train_inds, val_inds
    """

    patient_ids = trainvalDF.patient_id.unique()
    patient_ids = np.random.permutation(patient_ids)

    number_of_patients_to_take = int(np.floor(patient_ids.shape[0] * train_frac_patients))
    train_patients = patient_ids[:number_of_patients_to_take]
    val_patients = patient_ids[number_of_patients_to_take:]

    train_inds = trainvalDF[
        trainvalDF.patient_id.isin(train_patients)
    ].index.values

    val_inds = trainvalDF[
        trainvalDF.patient_id.isin(val_patients)
    ].index.values

    return train_inds, val_inds


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        save_path,
        lr_schedular=None,
        epochs=1,
        device='cpu'
):
    print(f'saving to {save_path}\nTraining on {args.device}')

    train_epoch_losses = list()
    val_epoch_losses = list()
    batch_losses = list()

    loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCEWithLogitsLoss()
    loss_func = FocalLoss(logits=True).to(device)

    for epoch in range(epochs):
        model.train()
        this_epoch_loss = 0
        for batch_num, (imgs, labels) in enumerate(train_dataloader):
            output = model(imgs.to(device))
            loss = loss_func(output, labels.view(labels.shape[0], -1).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            this_epoch_loss += (loss.detach().item())

        this_epoch_loss /= len(train_dataloader)

        this_val_loss = 0
        model.eval()
        val_preds = list()
        val_labels = list()
        with torch.no_grad():
            for batch_num, (imgs, labels) in enumerate(val_dataloader):
                output = model(imgs.to(device))
                loss = loss_func(output, labels.view(labels.shape[0], -1).float().to(device))

                this_val_loss += (loss.detach().item())
                val_preds.extend(torch.sigmoid(output).detach().cpu().squeeze().tolist())
                val_labels.extend(labels.detach().cpu().squeeze().tolist())

        this_val_loss /= len(val_dataloader)

        train_epoch_losses.append(np.mean(this_epoch_loss))
        val_epoch_losses.append(np.mean(this_val_loss))
        # batch_losses.extend(this_epoch_loss)

        postfix = ''
        if epoch >= 1 and val_epoch_losses[-1] < np.min(val_epoch_losses[:-1]):
            torch.save(model.state_dict(), save_path / f'model_epoch_{epoch}_loss_{val_epoch_losses[-1]:0.4e}.pth')
            postfix = 'Save model.'
        print(f'Epoch {epoch}\t Train loss {train_epoch_losses[-1]:0.4e}\tVal loss {val_epoch_losses[-1]:0.4e}\t{postfix}')

        if lr_schedular is not None:
            lr_schedular.step(val_epoch_losses[-1])
        else:
            flag = False
            if epoch == 3:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
                flag = True
            if epoch > 4 and epoch % 3 == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
                flag = True
            if flag:
                print(f'Reducing learning rate to {optimizer.param_groups[0]["lr"]}')

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(train_epoch_losses)), train_epoch_losses)
    plt.plot(np.arange(len(val_epoch_losses)), val_epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Cross Entropy Loss')
    plt.legend(['Train Loss', 'Val Loss'])
    plt.grid('both')
    plt.savefig(save_path / 'losses.jpg')
    plt.close()

    torch.save(model.state_dict(), save_path / f'model_epoch_{epoch}_last.pth')
    f = open(save_path / 'training_history.pkl', 'wb')
    pickle.dump(
        {
            'Train loss': train_epoch_losses,
            'Validation loss': val_epoch_losses,
            'Batch loss': batch_losses
        },
        f
    )
    f.close()
    import pandas as pd
    val_preds = pd.DataFrame({'val_preds': val_preds, 'val_labels': val_labels})
    val_preds.to_csv('val_preds.csv', index=False)


def test(
        model,
        test_dataloader,
        device='cpu'
):
    model.eval()
    predictions = list()
    with torch.no_grad():
        for batch_num, imgs in enumerate(test_dataloader):
            output = model(imgs.to(device))

            predictions.extend(torch.sigmoid(output.detach().cpu().view(-1)).tolist())

    return predictions


# Hair augmentation


class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return Image.fromarray(img)

        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

        return Image.fromarray(img)

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'


class dummy:
    pass


torch.manual_seed(0)
np.random.seed(0)

parser = dummy()
parser.batch_size = 32
parser.train_size = 0.8
parser.epochs = 20
parser.num_workers = 2
parser.weight_decay = 5e-4
parser.lr_patience = 3
parser.model_type = 'efficientnet-b1'
parser.lr = 1e-4
parser.oversample = 10  # 1 is no oversampling.
args = parser
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Model(args.model_type, num_classes=1).to(args.device)

master_dir = '../input/jpeg-small-512/jpeg_small_512'
trainvalDF = pd.read_csv(f'{master_dir}/train.csv')
# trainvalDF = trainvalDF[:400]
# train_inds, val_inds = get_train_val_inds(trainvalDF, args.train_size)
# train_sampler = SubsetRandomSampler(train_inds)
# val_sampler = SubsetRandomSampler(val_inds)

from sklearn.model_selection import train_test_split

train_inds, val_inds = train_test_split(
    np.arange(trainvalDF.shape[0]),
    train_size=args.train_size,
    stratify=trainvalDF.target
)

trainDF = trainvalDF.iloc[train_inds, :]
valDF = trainvalDF.iloc[val_inds, :]
# valDF.to_csv('valDF.csv')

malignant_train_inds = trainDF.index[trainDF.target == 1].tolist()
malignant_train_inds = args.oversample * malignant_train_inds
benign_train_inds = trainDF.index[trainDF.target == 0].tolist()
trainDF = trainDF.loc[malignant_train_inds + benign_train_inds, :]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# mean = (0, 0, 0)
# std = (1, 1, 1)

trainval_augments = torchvision.transforms.Compose([
    AdvancedHairAugmentation(hairs_folder='/kaggle/input/melanoma-hairs/'),
    torchvision.transforms.Resize(model.get_image_size()),  # REMEMBER TO PLAY WITH THIS
    torchvision.transforms.CenterCrop(model.get_image_size()),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.RandomRotation(45),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),

])

test_augments = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(model.get_image_size()),  # REMEMBER TO PLAY WITH THIS
    torchvision.transforms.CenterCrop(model.get_image_size()),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),

])

# trainval_dataset = MelanomaDataset(master_dir, 'train', transforms=trainval_augments)
train_dataset = MelanomaDataset(master_dir, 'train', transforms=trainval_augments, dataDF=trainDF)
val_dataset = MelanomaDataset(master_dir, 'train', transforms=test_augments, dataDF=valDF)

# raise ValueError()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    #     sampler=train_sampler,
    num_workers=args.num_workers,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    #     sampler=val_sampler,
    num_workers=args.num_workers,
    shuffle=True
)

# train_means = np.zeros((0,3))

# for img_transformed, label in train_dataloader:
#     train_means = np.concatenate([
#         train_means,
#         img_transformed.view(2, 3, -1).mean(2).numpy()
#     ])


# all_rgbs = np.concatenate([
#     train_means,
#     # np.zeros([10,3]),
#     # eval_means
# ])

# indices = np.arange(all_rgbs.shape[0])
# import matplotlib.pyplot as plt
# plt.figure()

# cs = ['r','g','b']
# for i in range(3):
#     plt.scatter(indices, all_rgbs[:,i], c=cs[i], alpha=0.5)

# plt.plot([0, indices.max()], [0,0])
# plt.grid('both')
# plt.savefig('aaa.jpg')

# print(f'means: {all_rgbs.mean(0)} \n stds: {all_rgbs.std(0)}')

# # raise ValueError


save_path = Path('checkpoints') / get_current_time_as_string()
if not save_path.exists():
    save_path.mkdir(parents=True)
    print('created dir')

with open(save_path / 'args.txt', 'w') as f:
    f.write(str(args))

group_1 = list()
for key, value in model.efficient_net._modules.items():
    if key == '_fc':
        continue

    group_1.append({'params': value.parameters(), 'lr': args.lr * 0.1})

param_groups = [
    *group_1,
    {'params': model.efficient_net._modules['_fc'].parameters(), 'lr': args.lr},
    {'params': model.output_linear.parameters(), 'lr': args.lr},
]

optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=args.lr_patience)

train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    save_path,
    lr_schedular=lr_schedular,
    epochs=args.epochs,
    device=args.device
)
