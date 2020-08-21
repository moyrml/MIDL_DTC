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
from torch.utils.data import DataLoader
import torchvision

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from tqdm import tqdm
import os


def init_prob_kmeans(model, eval_loader, args):
    print('\n\nInitializing cluster centers\n')
    torch.manual_seed(0)
    model = model.to(device)

    model_output = args.output_vector_length
    # cluster parameter initiate
    # Apply f_thetas to the unlabelled data
    model.eval()
    targets = np.zeros(len(eval_loader.dataset))
    feats = np.zeros((len(eval_loader.dataset), model_output))

    BS = eval_loader.batch_size

    for _, (x, label, idx) in enumerate(eval_loader):
        x = x.to(device)
        feat_1000, feat_1280 = model(x)

        if model_output == 1280:
            feats[idx, :] = feat_1280.data.cpu().numpy()
        elif model_output == 1000:
            feats[idx, :] = feat_1000.data.cpu().numpy()

        targets[idx] = label.data.cpu().numpy()

    # evaluate clustering performance
    # Use PCA to reduce to K dimensions
    pca = PCA(n_components=args.n_clusters)
    feats = pca.fit_transform(feats)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(feats)
    acc, nmi, ari = cluster_acc(targets, y_pred), nmi_score(targets, y_pred), ari_score(targets, y_pred)
    print('Init acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    # Construct target distribution
    probs = feat2prob(torch.from_numpy(feats), torch.from_numpy(kmeans.cluster_centers_))
    return acc, nmi, ari, kmeans.cluster_centers_, probs, model_output


def warmup_train(model, train_loader, eva_loader, args):
    train_history = list()
    eval_history = dict(acc=list(),
                        nmi=list(),
                        ari=list())

    optimizer = SGD(model.parameters(), lr=args.warmup_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(args.warmup_epochs):
        loss_record = AverageMeter(len(train_loader))

        model.train()
        model_output = args.output_vector_length

        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat_1000, feat_1280 = model(x)

            if model_output == 1280:
                feat = feat_1280
            elif model_output == 1000:
                feat = feat_1000

            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\n\n\nWarmup_train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        train_history.append(loss_record.avg)

        test_acc, test_nmi, test_ari, probs = test(model, eva_loader, args, epoch)

        eval_history['acc'].append(test_acc)
        eval_history['nmi'].append(test_nmi)
        eval_history['ari'].append(test_ari)

    args.p_targets = target_distribution(probs)

    return train_history, eval_history


def Baseline_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_history = list()
    eval_history = dict(acc=list(),
                        nmi=list(),
                        ari=list())

    for epoch in range(args.epochs):
        loss_record = AverageMeter(len(train_loader))
        model.train()
        model_output = args.output_vector_length

        exp_lr_scheduler.step()
        # for batch_idx, ((x, _), label, idx) in enumerate(tqdm(train_loader)):
        #     x = x.to(device)
        #     _, feat = model(x)
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat_1000, feat_1280 = model(x)

            if model_output == 1280:
                feat = feat_1280
            elif model_output == 1000:
                feat_bar = feat_1000

            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('\n\n\nTrain Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        # _, _, _, probs = test(model, eva_loader, args, epoch)
        train_history.append(loss_record.avg)

        test_acc, test_nmi, test_ari, probs = test(model, eva_loader, args, epoch)

        eval_history['acc'].append(test_acc)
        eval_history['nmi'].append(test_nmi)
        eval_history['ari'].append(test_ari)

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

    return train_history, eval_history


def PI_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loader.dataset.return_original = True

    train_history = list()
    eval_history = dict(acc=list(),
                        nmi=list(),
                        ari=list())

    w = 0
    model_output = args.output_vector_length

    for epoch in range(args.epochs):
        loss_record = AverageMeter(len(train_loader))
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, (x, label, idx, x_bar) in enumerate(tqdm(train_loader)):
            # x is transformed img, x_bar is original img
            x, x_bar = x.to(device), x_bar.to(device)
            feat_1000, feat_1280 = model(x)
            feat_1000_bar, feat_1280_bar = model(x_bar)

            if model_output == 1280:
                feat = feat_1280
                feat_bar = feat_1280_bar
            elif model_output == 1000:
                feat = feat_1000
                feat_bar = feat_1000_bar

            prob = feat2prob(feat, model.center)
            prob_bar = feat2prob(feat_bar, model.center)
            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)
            loss = sharp_loss + w * consistency_loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('\n\n\nTrain Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        # _, _, _, probs = test(model, eva_loader, args, epoch)
        train_history.append(loss_record.avg)

        test_acc, test_nmi, test_ari, probs = test(model, eva_loader, args, epoch)

        eval_history['acc'].append(test_acc)
        eval_history['nmi'].append(test_nmi)
        eval_history['ari'].append(test_ari)

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)
    torch.save(model.state_dict(), args.model_dir / 'pi_model.pth')
    print("model saved to {}.".format(args.model_dir))
    train_loader.dataset.return_original = False
    eval_loader.dataset.return_original = False

    return train_history, eval_history


def TE_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_clusters).float().to(device)  # intermediate values
    z_ema = torch.zeros(ntrain, args.n_clusters).float().to(device)  # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_clusters).float().to(device)  # current outputs

    model_output = args.output_vector_length
    train_history = list()
    eval_history = dict(acc=list(),
                        nmi=list(),
                        ari=list())

    for epoch in range(args.epochs):
        loss_record = AverageMeter(len(train_loader))
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat_1000, feat_1280 = model(x)

            if model_output == 1280:
                feat = feat_1280
            elif model_output == 1000:
                feat = feat_1000

            prob = feat2prob(feat, model.center)
            z_epoch[idx, :] = prob
            prob_bar = z_ema[idx, :].clone().detach()

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)
            loss = sharp_loss + w * consistency_loss
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Z = alpha * Z + (1. - alpha) * z_epoch
        z_ema = Z * (1. / (1. - alpha ** (epoch + 1)))
        print('\n\n\nTrain Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        # _, _, _, probs = test(model, eva_loader, args, epoch)
        train_history.append(loss_record.avg)

        test_acc, test_nmi, test_ari, probs = test(model, eva_loader, args, epoch)

        eval_history['acc'].append(test_acc)
        eval_history['nmi'].append(test_nmi)
        eval_history['ari'].append(test_ari)

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(probs)
    torch.save(model.state_dict(), args.model_dir / 'pi_model.pth')
    print("model saved to {}.".format(args.model_dir))

    return train_history, eval_history


def TEP_train(model, train_loader, eva_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    w = 0
    alpha = 0.6
    ntrain = len(train_loader.dataset)
    Z = torch.zeros(ntrain, args.n_clusters).float().to(device)  # intermediate values
    z_bars = torch.zeros(ntrain, args.n_clusters).float().to(device)  # temporal outputs
    z_epoch = torch.zeros(ntrain, args.n_clusters).float().to(device)  # current outputs

    model_output = args.output_vector_length

    model_output = args.output_vector_length
    train_history = list()
    eval_history = dict(acc=list(),
                        nmi=list(),
                        ari=list())

    for epoch in range(args.epochs):
        loss_record = AverageMeter(len(train_loader))
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            feat_1000, feat_1280 = model(x)

            if model_output == 1280:
                feat = feat_1280
            elif model_output == 1000:
                feat = feat_1000
            prob = feat2prob(feat, model.center)
            loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\n\n\nTrain Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        # _, _, _, probs = test(model, eva_loader, args, epoch)

        test_acc, test_nmi, test_ari, probs = test(model, eva_loader, args, epoch)

        eval_history['acc'].append(test_acc)
        eval_history['nmi'].append(test_nmi)
        eval_history['ari'].append(test_ari)

        z_epoch = probs.float().to(device)
        Z = alpha * Z + (1. - alpha) * z_epoch
        z_bars = Z * (1. / (1. - alpha ** (epoch + 1)))

        if epoch % args.update_interval == 0:
            print('updating target ...')
            args.p_targets = target_distribution(z_bars).float().to(device)
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

    return train_history, eval_history


def test(model, test_loader, args, epoch='test'):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_clusters))
    probs = np.zeros((len(test_loader.dataset), args.n_clusters))

    model_output = args.output_vector_length

    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        #         _, feat = model(x)
        feat_1000, feat_1280 = model(x)

        if model_output == 1280:
            feat = feat_1280
        elif model_output == 1000:
            feat = feat_1000

        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets,
                                                                                                              preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)
    return acc, nmi, ari, probs


def test(model, test_loader, args, epoch='test', post_training=False):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_clusters))
    probs = np.zeros((len(test_loader.dataset), args.n_clusters))

    model_output = args.output_vector_length

    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        #         _, feat = model(x)
        feat_1000, feat_1280 = model(x)

        if model_output == 1280:
            feat = feat_1280
        elif model_output == 1000:
            feat = feat_1000

        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets,
                                                                                                              preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)

    if not post_training:
        return acc, nmi, ari, probs

    df = pd.DataFrame({'targets': targets, 'preds': preds})
    cluster_assignments = df.groupby(
        df.columns.to_list()
    ).size().unstack(1)

    cluster_translations = dict()

    while cluster_assignments.size > 0:
        if cluster_assignments.isna().all().all():
            cluster_translations[cluster_assignments.columns[0]] = cluster_assignments.index[0]
            max_loc = np.zeros((2, 1))
        else:
            max_loc = np.where((cluster_assignments == cluster_assignments.max().max()).values)
            cluster_translations[cluster_assignments.columns[max_loc[1][0]]] = cluster_assignments.index[max_loc[0][0]]

        cluster_assignments.drop(cluster_assignments.index[max_loc[0][0]], inplace=True)
        cluster_assignments.drop(cluster_assignments.columns[max_loc[1][0]], axis='columns', inplace=True)

    # cluster_max = cluster_assignments.idxmax(0) # this contains the cluster that has the max number of members
    # cluster_translations[cluster_max.iloc[0]] = cluster_max.index[0]

    return acc, nmi, ari, probs, cluster_translations


def make_predictions(model, test_loader, args, epoch='test'):
    model.eval()
    preds = list()

    model_output = args.output_vector_length

    for batch_idx, (x, _, idx) in enumerate(tqdm(test_loader)):
        x = x.to(device)

        feat_1000, feat_1280 = model(x)

        if model_output == 1280:
            feat = feat_1280
        elif model_output == 1000:
            feat = feat_1000

        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)

        preds.extend(pred.tolist())

    return preds


class dummy:
    pass


parser = dummy()
parser.warmup_lr = 0.1
parser.lr = 0.1
parser.gamma = 0.5
parser.milestones = [10, 20, 40, 60, 80]
parser.momentum = 0.9

parser.weight_decay = 1e-5
parser.warmup_epochs = 10
parser.epochs = 30
parser.rampup_length = 5
parser.rampup_coefficient = 10.0

parser.batch_size = 32
parser.update_interval = 10
parser.n_clusters = 5

parser.DTC = 'PI'
parser.train_size = 0.8
parser.output_vector_length = 1000

parser.overunder_amount = [1, 1, 1, 1, 1]  # [1,1,1,3,3] # a list of ones = no resampling. >1 over-, < 1 under- sampling
#################################################################################################
# Watch out for this...
parser.equalize = False
#################################################################################################

parser.seed = 0
parser.save_txt = True
parser.pretrained_model = '../input/pretrained-model-weights/model_epoch_19_last.pth'
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

######################################################### Remember this ############################################
model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device(args.device)), strict=False)
####################################################################################################################
init_feat_extractor = model
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

# Cluster centers- Unlabeled
init_acc, init_nmi, init_ari, init_centers, init_probs, model_output = init_prob_kmeans(init_feat_extractor,
                                                                                        eval_loader, args)

# Initial q - target distribution
args.p_targets = target_distribution(init_probs)

model = Model('efficientnet-b1').to(args.device)  ###################################################
model.last = nn.Linear(model_output, args.n_clusters).to(args.device)
model.mode = 'activations'

model.load_state_dict(init_feat_extractor.state_dict(), strict=False)  ############################################

model.center = nn.Parameter(torch.Tensor(args.n_clusters, args.n_clusters))
model.center.data = torch.tensor(init_centers).float().to(device)

print('\n\n--------------------Warmup training-----------------------\n')
warmup_train_history, warmup_eval_history = warmup_train(model, train_loader, eval_loader, args)

print(f'\n\n--------------------{args.DTC} training-----------------------\n')

if args.DTC == 'Baseline':
    train_history, eval_history = Baseline_train(model, train_loader, eval_loader, args)
elif args.DTC == 'PI':
    train_history, eval_history = PI_train(model, train_loader, eval_loader, args)
elif args.DTC == 'TE':
    train_history, eval_history = TE_train(model, train_loader, eval_loader, args)
elif args.DTC == 'TEP':
    train_history, eval_history = TEP_train(model, train_loader, eval_loader, args)

cluster_translations = {}
# acc, nmi, ari, _, cluster_translations = test(model, eval_loader, args, post_training=True)
# This is done to prevent nan craqshing while TE training
acc, nmi, ari, _ = test(model, eval_loader, args, post_training=False)
print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))

"""
Make Predictions on test data
"""
print(f'\n\n--------------------Testing-----------------------\n')

# keys_not_in_translation = [k not in cluster_translations.keys() for k in range(args.n_clusters)]
# def custom_formatwarning(msg, *args, **kwargs):
#     # ignore everything except the message
#     return str(msg) + '\n'

# warnings.formatwarning = custom_formatwarning
# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# if np.any(keys_not_in_translation):
#     warnings.warn(f'Clusters {np.where(keys_not_in_translation)[0]} have no labels assigned to them!', category=DeprecationWarning)

# test_dataframe['diagnosis'] = make_predictions(model, test_loader, args, epoch='test')
# # test_dataframe['diagnosis'] = test_dataframe.diagnosis.replace(cluster_translations)
# test_dataframe['diagnosis'] = test_dataframe['diagnosis'].astype(int)
# test_dataframe.to_csv('submission.csv', index=False)

"""
-------------------------------
"""

################################ Plot ###################################
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(train_history)), train_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss'])
plt.grid('both')
plt.savefig(save_path / 'losses.jpg')
plt.close()

plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(eval_history['acc'])), eval_history['acc'])
plt.plot(np.arange(len(eval_history['nmi'])), eval_history['nmi'])
plt.plot(np.arange(len(eval_history['ari'])), eval_history['ari'])
plt.xlabel('Epoch')
plt.ylim([0, 1])
plt.legend(['Acc', 'NMI', 'ARI'])
plt.grid('both')
plt.savefig(save_path / 'acc_nmi_ari.jpg')
plt.close()

################################ save stuff ###################################
import pickle

with open(save_path / 'training_history.pkl', 'wb') as f:
    pickle.dump(
        {
            'train history': train_history,
            'eval history': eval_history
        },
        f
    )

if args.save_txt:
    with open(args.save_txt_path / args.save_txt_name, 'a') as f:
        f.write("{:.4f}, {:.4f}, {:.4f}\n".format(acc, nmi, ari))

################################ tSNE ###################################
import gc

del init_feat_extractor, test_dataframe, test_loader, train_loader,
gc.collect()
save_tsne(args, save_path, [2, 5, 10, 20, 50], model, eval_loader, limit=100)
save_pca(args, save_path, [2, 5, 10, 20, 50], model, eval_loader, limit=960)
