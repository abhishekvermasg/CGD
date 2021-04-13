import argparse

import pandas as pd
import torch
from thop import profile, clever_format
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import process_sop_data
from pytorch_metric_learning import losses, miners, distances, reducers, testers

from model import Model, set_bn_eval
from utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler

# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
# torch.autograd.profiler.profile(True)
# torch.autograd.profiler.emit_nvtx(True)

def train(net, optim, loss_type=None):
    net.train()
    # fix bn on backbone network
    net.apply(set_bn_eval)
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels in data_bar:
        inputs, labels = inputs.cuda(), labels.cuda()
        features, classes = net(inputs)
        if loss_type == 'arcface':
            class_loss = class_criterion(features, labels)
            # print(class_loss)
        else:
            class_loss = class_criterion(classes, labels)
        feature_loss = feature_criterion(features, labels)
        loss = class_loss + feature_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                inputs, labels = inputs.cuda(), labels.cuda()
                features, classes = net(inputs)
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        # compute recall metric
        if data_name == 'isc':
            acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids,
                              eval_dict['gallery']['features'], gallery_data_set.labels)
        else:
            acc_list = recall(eval_dict['test']['features'], test_data_set.labels, recall_ids)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)
    print()
    return acc_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CGD')
    parser.add_argument('--data_path', default='/home/data', type=str, help='datasets path')
    parser.add_argument('--data_name', default='car', type=str, choices=['car', 'cub', 'sop', 'isc'],
                        help='dataset name')
    parser.add_argument('--cuda', default=True, type=bool, help='gpu')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnext50'],
                        help='backbone network type')
    parser.add_argument('--gd_config', default='SG', type=str,
                        choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'],
                        help='global descriptors config')
    parser.add_argument('--feature_dim', default=512, type=int, help='feature dim') # 1536
    parser.add_argument('--smoothing', default=0.1, type=float, help='smoothing value for label smoothing')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='temperature scaling used in softmax cross-entropy loss')
    parser.add_argument('--size', default=224, type=int, help='size of square image')
    parser.add_argument('--margin', default=0.1, type=float, help='margin of m for triplet loss')
    parser.add_argument('--recalls', default='10,25,50', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=32, type=int, help='train batch size') # 128
    parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
    parser.add_argument('--df_path', default='../input/shopee-clean/train_90.csv',\
     type=str, help='train df path')
    parser.add_argument('--class_loss', default='ce', type=str, help='loss function for class labels')
    parser.add_argument('--feature_loss', default='bhtl', type=str, help='loss function for features')
    parser.add_argument('--data_dir', default='../input/shopee-product-matching/train_images/',\
     type=str, help='train images folder')
    parser.add_argument('--warmup_epochs', default=2, type=int, help='number of warmup epochs')
    parser.add_argument('--workers', default=2, type=int, help='number of workers')


    opt = parser.parse_args()
    # args parse
    data_path, data_name, backbone_type = opt.data_path, opt.data_name, opt.backbone_type
    gd_config, feature_dim, smoothing, temperature = opt.gd_config, opt.feature_dim, opt.smoothing, opt.temperature
    gpu = opt.cuda
    margin, recalls, batch_size = opt.margin, [int(k) for k in opt.recalls.split(',')], opt.batch_size
    num_epochs = opt.num_epochs
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}'.format(data_name, backbone_type, gd_config, feature_dim,
                                                        smoothing, temperature, margin, batch_size)

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []

    process_sop_data(opt.data_dir, opt.df_path)

    # dataset loader
    train_data_set = ImageReader(data_path, data_name, 'train')
    train_sample = MPerClassSampler(train_data_set.labels, batch_size)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=opt.workers, pin_memory=True)
    test_data_set = ImageReader(data_path, data_name, 'test')
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)
    eval_dict = {'test': {'data_loader': test_data_loader}}

    # model setup, model profile, optimizer config and loss definition
    model = Model(backbone_type, gd_config, feature_dim, num_classes=len(train_data_set.class_to_idx)).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).cuda(),), verbose=False)
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    if opt.class_loss == 'arcface':
        class_criterion = losses.ArcFaceLoss(num_classes=len(train_data_set.class_to_idx), \
            embedding_size=512) # , reducer=reducers.ThresholdReducer(low=0.1)
    elif opt.class_loss == 'contra':
        distance = distances.CosineSimilarity()
        class_criterion = losses.ContrastiveLoss(distance=distance)
    elif opt.class_loss == 'multi':
        class_criterion = losses.MultiSimilarityLoss()
    else: 
        class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=smoothing, temperature=temperature)
    if opt.class_loss == 'arcface':
        print('Using ArcFace')
        optimizer = Adam([{'params': model.parameters()}, {'params': class_criterion.parameters()}], lr=1e-4)
    else:
        optimizer = Adam(model.parameters(), lr=1e-4)
    # lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.8 * num_epochs)], gamma=0.1)
    # lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.8 * num_epochs)], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    feature_criterion = BatchHardTripletLoss(margin=margin)

    best_recall = 0.0
    for epoch in range(1, num_epochs + 1):
        if epoch < opt.warmup_epochs + 1:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / num_epochs
        train_loss, train_accuracy = train(model, optimizer, loss_type='arcface')
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        rank = test(model, recalls)
        lr_scheduler.step()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(epoch), index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['test_images'] = test_data_set.images
            data_base['test_labels'] = test_data_set.labels
            data_base['test_features'] = eval_dict['test']['features']
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(epoch))
            torch.save(data_base, 'results/{}_data_base.pth'.format(epoch))