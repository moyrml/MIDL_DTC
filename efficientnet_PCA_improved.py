import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from efficientnet_helper import *
from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoaderw
import torchvision

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from tqdm import tqdm
import os


class dummy:
    pass


parser = dummy()
parser.warmup_lr = 0.1
parser.lr = 0.1
parser.gamma = 0.5
parser.milestones = [20, 40, 60, 80]
parser.momentum = 0.9

parser.weight_decay = 1e-5
parser.warmup_epochs = 30
parser.epochs = 50
parser.rampup_length = 5
parser.rampup_coefficient = 10.0

parser.batch_size = 32
parser.update_interval = 10
parser.n_clusters = 5
parser.n_clusters = 1280

parser.DTC = 'PI'
parser.train_size = 0.8
parser.output_vector_length = 1280
model_output = parser.output_vector_length

parser.overunder_amount = [1, 1, 1, 1, 1]  # a list of ones = no resampling. >1 over-, < 1 under- sampling
#################################################################################################
# Watch out for this...
parser.equalize = False  # True
#################################################################################################

parser.seed = 0
parser.save_txt = True
parser.pretrained_DTC = '../input/v2ofirnew/pi_model.pth'
parser.dataset_root = '../input/retina-small-512'
parser.exp_root = './checkpoints/'
save_path = Path('checkpoints') / get_current_time_as_string()
if not save_path.exists():
    save_path.mkdir(parents=True)
parser.model_dir = save_path
parser.model_name = 'efficientnet'
parser.save_txt_name = 'result.txt'
parser.save_txt_path = parser.model_dir

parser.num_workers = 2
args = parser
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

print(f'Using {device}')

args.device = device
seed_torch(args.seed)

# runner_name = os.path.basename(__file__).split(".")[0]
# model_dir = args.exp_root + '{}/{}'.format(runner_name, args.DTC)
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# args.model_dir = model_dir + '/' + args.model_name + '.pth'
# args.save_txt_path = args.exp_root + '{}/{}/{}'.format(runner_name, args.DTC, args.save_txt_name)

model = Model('efficientnet-b1').to(args.device)
# model.last = nn.Linear(model_output, args.n_clusters).to(args.device)
######################################################### Remember this ############################################
model.load_state_dict(torch.load(args.pretrained_DTC, map_location=torch.device(args.device)), strict=False)
####################################################################################################################
model.mode = 'activations'

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# mean = [0.0850268 , 0.25153838, 0.47517351]
# std = [0.07048737, 0.0696396 , 0.15160605]

# mean = [0.66013826, 0.66338438, 0.67312815]
# std = [0.06188058, 0.061283, 0.05997333]

trainval_augments = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(model.get_image_size()),  # REMEMBER TO PLAY WITH THIS
    torchvision.transforms.CenterCrop(model.get_image_size()),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),

])

pi_augments = trainval_augments
# pi_augments = torchvision.transforms.Compose([
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.Resize(model.get_image_size()),  # REMEMBER TO PLAY WITH THIS
#     torchvision.transforms.CenterCrop(model.get_image_size()),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean, std),

# ])


test_augments = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(model.get_image_size()),  # REMEMBER TO PLAY WITH THIS
    torchvision.transforms.CenterCrop(model.get_image_size()),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),

])

trainval_dataframe = pd.read_csv(args.dataset_root + '/train.csv')


## Over/under sample
def overunder_helper(x):
    amount = args.overunder_amount.pop(0)

    if amount >= 1:
        inds = np.arange(x.shape[0]).tolist() * amount
    else:
        inds = np.arange(x.shape[0]).tolist()[:int(x.shape[0] * amount)]
    return x[inds]


if not args.equalize:
    sorted_indices = trainval_dataframe. \
        groupby('diagnosis'). \
        apply(lambda x: x.index.to_numpy()). \
        apply(np.random.permutation). \
        sort_index(). \
        apply(overunder_helper). \
        apply(pd.Series). \
        stack(). \
        values. \
        astype(int)

else:
    sorted_indices = trainval_dataframe. \
        groupby('diagnosis'). \
        apply(lambda x: x.index.to_numpy()). \
        apply(np.random.permutation). \
        sort_index()

    min_len = sorted_indices.apply(len).min()
    sorted_indices = sorted_indices.apply(
        lambda x: np.random.choice(x, min_len, replace=False)
    ).apply(pd.Series). \
        stack(). \
        values. \
        astype(int)

trainval_dataframe = trainval_dataframe.loc[sorted_indices]
trainval_dataframe = trainval_dataframe

# test_dataframe = pd.read_csv(args.dataset_root + '/test.csv')
test_dataframe = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

from sklearn.model_selection import train_test_split

if True:
    train_inds, val_inds = train_test_split(
        np.arange(trainval_dataframe.shape[0]),
        train_size=args.train_size,
        stratify=trainval_dataframe.diagnosis
    )

    trainDF = trainval_dataframe  # .loc[train_inds, :]
    evalDF = trainDF
    # evalDF = trainval_dataframe.loc[val_inds, :]

    train_dataset = RetinaDataset(
        args.dataset_root,
        'train',
        transforms=trainval_augments,
        dataDF=trainDF,
        #         num_transforms=2,
        pi_augments=pi_augments
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    eval_dataset = RetinaDataset(
        args.dataset_root,
        'train',
        transforms=trainval_augments,
        dataDF=trainDF,
        num_transforms=None,
        # pi_augments=pi_augments
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    test_dataset = RetinaDataset(
        args.dataset_root,
        'test',
        transforms=test_augments,
        dataDF=test_dataframe,
        num_transforms=None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

import gc

del test_dataframe, test_loader, train_loader,
gc.collect()
# save_tsne(args,save_path,[2,5,10,20,50], model, eval_loader, limit=100)
save_pca(args, save_path, [2, 5, 10, 20, 50], model, eval_loader, limit=960)
