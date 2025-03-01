import time
import logging
import argparse
import pandas as pd
from scipy.stats import gmean
from collections import defaultdict

import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboard_logger import Logger
from torch.utils.data import DataLoader

from utils import *
from datasets import AgeDB
from resnet import resnet50
from dist_loss import DistLoss
from loss_utils import get_label_distribution, get_batch_theoretical_labels

import os
os.environ["KMP_WARNINGS"] = "FALSE"


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

# Dist Loss
parser.add_argument('--bw_method', default=0.5, help='The method used to calculate the estimator bandwidth')
parser.add_argument('--min_label', default=0, help='The theoretical minimum of the label')
parser.add_argument('--max_label', default=120, help='The theoretical maximum of the label')
parser.add_argument('--step', default=1, help='The interval between discrete labels in the estimated distribution')
parser.add_argument('--unweighted', action='store_false', help='whether to use weighted loss')
parser.add_argument('--not_balanced_metric', action='store_true', help='do not use balanced metric')
parser.add_argument('--distribution_loss_term_weight', type=float, default=1.0, help='Weight for the distribution distance loss term')
parser.add_argument('--regularization_strength', type=float, default=0.1, help='Strength of regularization in soft_sort algorithm')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='inverse', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')
# two-stage training: RRT
parser.add_argument('--retrain_fc', action='store_true', default=False, help='whether to retrain last regression layer (regressor)')
parser.add_argument('--retrain_from', type=str, default='None')

# training/optimization related
parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='../../datasets/AgeDB', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=90, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--print_freq', type=int, default=2, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.start_epoch, args.best_loss = 0, 1e5


if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.retrain_fc:
    args.store_name += f'_retrain_fc'
if args.unweighted:
    args.store_name = f"{args.dataset}_{args.store_name}_{args.optimizer}_dist_weighted_{args.loss}_weight_{args.distribution_loss_term_weight}_{args.lr}_{args.batch_size}_{args.bw_method}_{args.epoch}_{args.retrain_from}"
else:
    args.store_name = f"{args.dataset}_{args.store_name}_{args.optimizer}_dist_{args.loss}_weight_{args.distribution_loss_term_weight}_{args.lr}_{args.batch_size}_{args.bw_method}_{args.epoch}_{args.retrain_from}"

prepare_folders(args)

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, 'training.log')),
        logging.StreamHandler()
    ])
print = logging.info
print(f"Args: {args}")
print(f"Store name: {args.store_name}")

tb_logger = Logger(logdir=os.path.join(args.store_root, args.store_name), flush_secs=2)


def main():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')
    print(f"File (.csv): {args.dataset}.csv")
    df = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    
    # Training label distribution estimation and pseudo-labels generation
    density = get_label_distribution(train_labels, args.bw_method, args.min_label, args.max_label, args.step)
    batch_theoretical_labels = get_batch_theoretical_labels(density, args.batch_size, args.min_label, step=args.step)
    batch_theoretical_labels = torch.tensor(batch_theoretical_labels, dtype=torch.float32).reshape(-1,1).cuda()

    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size, split='train',
                          reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, img_size=args.img_size, split='val')
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test, img_size=args.img_size, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")

    # Model
    print('=====> Building model...')
    model = resnet50()
    model = torch.nn.DataParallel(model).cuda()

    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, train_labels=train_labels, prefix='Test')
        return

    if args.retrain_fc:
        assert args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None, [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epoch): 
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, batch_theoretical_labels, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean, mean_MSE, mean_L1 = validate(val_loader, model, train_labels=train_labels)

        if args.not_balanced_metric:
            loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        else:
            loss_metric = mean_MSE if args.loss == 'mse' else mean_L1
            
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best Loss: {args.best_loss:.3f}")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

        tb_logger.log_value('train_loss', train_loss, epoch)
        tb_logger.log_value('val_loss_mse', val_loss_mse, epoch)
        tb_logger.log_value('val_loss_l1', val_loss_l1, epoch)
        tb_logger.log_value('val_loss_gmean', val_loss_gmean, epoch)

    # test with best checkpoint
    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    test_loss_mse, test_loss_l1, test_loss_gmean, mean_MSE, mean_L1 = validate(test_loader, model, train_labels=train_labels, prefix='Test')
    print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")

def train(train_loader, batch_theoretical_labels, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    loss_fn = DistLoss(
        loss_fn=args.loss,
        loss_weight=args.distribution_loss_term_weight,
        regularization_strength=args.regularization_strength
        )
     
    optimizer.zero_grad() 

    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, weights = \
            inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), weights.cuda(non_blocking=True)
        outputs = model(inputs)

        if not args.unweighted:
            loss = loss_fn(outputs.type(torch.double), targets.type(torch.double), batch_theoretical_labels.type(torch.double))
        else:
            loss = loss_fn(outputs.type(torch.double), targets.type(torch.double), batch_theoretical_labels.type(torch.double), weights.type(torch.double))
        loss.backward()
        optimizer.step()         
        optimizer.zero_grad()    
        
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0)) 

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % (args.print_freq) == 0:
            progress.display(idx)

    return losses.avg

def validate(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(inputs)

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())

            loss_mse = criterion_mse(outputs, targets)
            loss_l1 = criterion_l1(outputs, targets)
            loss_all = criterion_gmean(outputs, targets)
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)

        mean_MSE, mean_L1 = balanced_metrics(np.hstack(preds), np.hstack(labels))
        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        shot_dict_balanced = shot_metrics_balanced(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print('-' * 40)
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
              f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")
        print('=' * 40)
        print(f" * bMSE {mean_MSE:.3f}\tbMAE {mean_L1:.3f}")
        print('-' * 40)
        print(f" * Many: bMSE {shot_dict_balanced['many']['mse']:.3f}\t"
              f"bMAE {shot_dict_balanced['many']['l1']:.3f}\tG-Mean {shot_dict_balanced['many']['gmean']:.3f}")
        print(f" * Median: bMSE {shot_dict_balanced['median']['mse']:.3f}\t"
              f"bMAE {shot_dict_balanced['median']['l1']:.3f}\tG-Mean {shot_dict_balanced['median']['gmean']:.3f}")
        print(f" * Low: bMSE {shot_dict_balanced['low']['mse']:.3f}\t"
              f"bMAE {shot_dict_balanced['low']['l1']:.3f}\tG-Mean {shot_dict_balanced['low']['gmean']:.3f}")

        save_path = None
        if not args.evaluate and prefix=='Test':
            save_path = f'{args.store_root}/{args.store_name}/results.npz'
        elif args.evaluate:
            save_path = f"{os.path.join(os.path.dirname(args.resume), 'results.npz')}"
        if save_path != None:
            np.savez(save_path, pred=preds, label=labels)
            print(f'Results are saved in {save_path}')
            
    return losses_mse.avg, losses_l1.avg, loss_gmean, mean_MSE, mean_L1

def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict

def balanced_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    mse_per_class, l1_per_class = [], []
    for l in np.unique(labels):
        mse_per_class.append(np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.mean(np.abs(preds[labels == l] - labels[labels == l])))

    mean_mse = sum(mse_per_class) / len(mse_per_class)
    mean_l1 = sum(l1_per_class) / len(l1_per_class)
    return mean_mse, mean_l1

def shot_metrics_balanced(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.mean(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])


        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / len(many_shot_mse)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / len(many_shot_l1)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / len(median_shot_mse)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / len(median_shot_l1)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / len(low_shot_mse)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / len(low_shot_l1)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


if __name__ == '__main__':
    main()
