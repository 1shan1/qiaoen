# -*- coding: utf-8 -*-

import sys, os, argparse, time

import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.externals import joblib
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, hopenet, utils
import torch.utils.model_zoo as model_zoo


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=100, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.0001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
                        default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.',
                        default='', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
                        default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default='', type=str)
    parser.add_argument('--direction', default=False, type=bool, help='if training with direction labels.')
    parser.add_argument('--flip', dest='flip', type=bool, default=True, help='flip input images or not. ')
    parser.add_argument('--test_txt', dest='test_txt', type=str, default='', help='path of test dataset.')
    parser.add_argument('--crop', dest='crop', type=bool, default=False)
    parser.add_argument('--test_dt', dest='test_dt', type=str, default='')

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    # b = [model.conv1, model.bn1, model.fc_finetune]
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model_dict = model.state_dict()

        index = 0
        for k, v in model_dict.iteritems():
            if k in saved_state_dict and index < 260:
                model_dict[k] = saved_state_dict[k]

            index += 1

        model.load_state_dict(model_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(140),
                                          transforms.RandomCrop(128), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'DRIVER_TEST':
        pose_dataset = datasets.DRIVER_TEST(args.data_dir, args.filename_list, transformations, args.direction,
                                            args.flip, crop=bool(args.crop))
    else:
        print 'Error: not a valid dataset name'
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    model.cuda(gpu)

    criterion = nn.CrossEntropyLoss(size_average=True).cuda(gpu)
    reg_criterion = nn.MSELoss(size_average=True).cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                 lr=args.lr)

    if args.test_txt != '':
        test_dataset = datasets.Driver_Train_TXT(args.test_txt, transformations)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=2)
    if args.test_dt != '':
        dt_pose_dataset = None
        test_dataset = datasets.Driver_Train_TXT(args.test_dt, transformations)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=2)

    print 'Ready to train network.'
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:, 0]).cuda(gpu)
            label_pitch = Variable(labels[:, 1]).cuda(gpu)
            label_roll = Variable(labels[:, 2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:, 0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:, 1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:, 2]).cuda(gpu)

            if args.direction:
                label_direction = Variable(cont_labels[:, 3]).cuda(gpu)
                label_direction = label_direction.type(torch.LongTensor).cuda(gpu)

                # Forward pass
                yaw, pitch, roll, direction = model(images, idx_tensor)

                # Cross entropy loss
                loss_yaw = criterion(yaw, label_yaw)
                loss_pitch = criterion(pitch, label_pitch)
                loss_roll = criterion(roll, label_roll)

                # MSE loss
                yaw_predicted = softmax(yaw)
                pitch_predicted = softmax(pitch)
                roll_predicted = softmax(roll)

                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
                loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
                loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

                # Total loss
                loss_yaw += alpha * loss_reg_yaw
                loss_pitch += alpha * loss_reg_pitch
                loss_roll += alpha * loss_reg_roll

                # Direction loss
                loss_direction = criterion(direction, label_direction)
                # loss_direction += 0.1 * (loss_yaw + loss_pitch + loss_roll)

                loss_seq = [loss_yaw, loss_pitch, loss_roll, loss_direction]
                grad_seq = [torch.ones(1).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer.zero_grad()
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer.step()

                if (i + 1) % args.batch_size == 0:
                    print 'Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f, Direction %.4f' \
                          % (epoch + 1, num_epochs, i + 1, len(pose_dataset) // batch_size, loss_yaw.data[0],
                             loss_pitch.data[0], loss_roll.data[0], loss_direction.data[0])
            else:
                # Forward pass
                yaw, pitch, roll, direction = model(images, idx_tensor)

                # Cross entropy loss
                loss_yaw = criterion(yaw, label_yaw)
                loss_pitch = criterion(pitch, label_pitch)
                loss_roll = criterion(roll, label_roll)

                # MSE loss
                yaw_predicted = softmax(yaw)
                pitch_predicted = softmax(pitch)
                roll_predicted = softmax(roll)

                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
                roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

                loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
                loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
                loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

                # Total loss
                loss_yaw += alpha * loss_reg_yaw
                loss_pitch += alpha * loss_reg_pitch
                loss_roll += alpha * loss_reg_roll

                loss_seq = [loss_yaw, loss_pitch, loss_roll]
                grad_seq = [torch.ones(1).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer.zero_grad()
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer.step()

                if (i + 1) % args.batch_size == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                           % (epoch + 1, num_epochs, i + 1, len(pose_dataset) // batch_size, loss_yaw.data[0],
                              loss_pitch.data[0], loss_roll.data[0]))

        # Save models at numbered epochs.
        if epoch % 3 == 0 and epoch < num_epochs:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
                       'output/snapshots/' + args.output_string + '_epoch_' + str(epoch + 1) + '.pth')

        if args.test_txt != '':
            # Test driver_test dataset's accuracy

            clf = joblib.load('../driver_data.m')

            if epoch % 2 == 0 and epoch < num_epochs:
                print 'Test driver_test dataset...'

                error_direction_dt = 0
                error_direction_nn = 0
                total_direction = len(test_dataset)
                for i, (images, name, direction_label, item) in enumerate(test_loader):
                    parent_dir = os.path.basename(os.path.dirname(name[0]))
                    images = Variable(images).cuda(gpu)

                    yaw, pitch, roll, direction = model(images, idx_tensor)

                    # Continuous predictions
                    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
                    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
                    roll_predicted = utils.softmax_temperature(roll.data, 1)

                    idx_tensor_test = [idx for idx in xrange(66)]
                    idx_tensor_test = torch.FloatTensor(idx_tensor_test).cuda(gpu)

                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor_test, 1).cpu() * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor_test, 1).cpu() * 3 - 99
                    roll_predicted = torch.sum(roll_predicted * idx_tensor_test, 1).cpu() * 3 - 99

                    prediction = int(clf.predict(np.asarray(
                        [[float(yaw_predicted[0]), float(pitch_predicted[0]), float(roll_predicted[0])]]))
                    )
                    if prediction != int(direction_label[0]):
                        error_direction_dt += 1

                    direction = softmax(direction)
                    _, direciton_bpred = torch.max(direction.data, 1)
                    direction_value_bpred = int(direciton_bpred.cpu())

                    if int(direction_value_bpred) != int(direction_label[0]):
                        error_direction_nn += 1

                print 'direction error_dt/total: %d/%d, accuracy_dt: %.4f .' % (
                    error_direction_dt, total_direction,
                    (float)(total_direction - error_direction_dt) / total_direction)

                print 'direction error_nn/total: %d/%d, accuracy_nn: %.4f .' % (
                    error_direction_nn, total_direction,
                    (float)(total_direction - error_direction_nn) / total_direction)

        if args.test_dt != '':
            idx_tensor_test = [idx for idx in xrange(66)]
            idx_tensor_test = torch.FloatTensor(idx_tensor_test).cuda(gpu)

            if epoch % 2 == 0 and epoch < num_epochs:
                print 'Test driver_test dataset...'

                from decision_tree_demo import tree_demo

                ori_img_txt = '/media/f/face_angle_dataset/driver_train_ori.txt'

                if dt_pose_dataset is None:
                    dt_pose_dataset = datasets.Driver_Train_TXT(ori_img_txt, transformations)
                    dt_loader = torch.utils.data.DataLoader(dataset=dt_pose_dataset,
                                                            batch_size=1,
                                                            num_workers=2)

                train_data = []
                target_data = []

                for i, (images, name, direction_label, item) in enumerate(dt_loader):
                    parent_dir = os.path.basename(os.path.dirname(name[0]))

                    images = Variable(images).cuda(gpu)

                    yaw, pitch, roll, direction = model(images, idx_tensor)

                    # Binned predictions
                    _, yaw_bpred = torch.max(yaw.data, 1)
                    _, pitch_bpred = torch.max(pitch.data, 1)
                    _, roll_bpred = torch.max(roll.data, 1)

                    # Continuous predictions
                    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
                    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
                    roll_predicted = utils.softmax_temperature(roll.data, 1)

                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor_test, 1).cpu() * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor_test, 1).cpu() * 3 - 99
                    roll_predicted = torch.sum(roll_predicted * idx_tensor_test, 1).cpu() * 3 - 99

                    train_data.append([float(yaw_predicted[0]), float(pitch_predicted[0]), float(roll_predicted[0])])
                    target_data.append([int(direction_label[0])])

                clf = tree_demo(np.asarray(train_data), np.asarray(target_data))

                error_direction_dt = 0
                total_direction = len(test_dataset)
                for i, (images, name, direction_label, item) in enumerate(test_loader):
                    parent_dir = os.path.basename(os.path.dirname(name[0]))
                    images = Variable(images).cuda(gpu)

                    yaw, pitch, roll, direction = model(images, idx_tensor)

                    # Continuous predictions
                    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
                    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
                    roll_predicted = utils.softmax_temperature(roll.data, 1)

                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor_test, 1).cpu() * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor_test, 1).cpu() * 3 - 99
                    roll_predicted = torch.sum(roll_predicted * idx_tensor_test, 1).cpu() * 3 - 99

                    prediction = int(clf.predict(np.asarray(
                        [[float(yaw_predicted[0]), float(pitch_predicted[0]), float(roll_predicted[0])]]))
                    )

                    if prediction != int(direction_label[0]):
                        error_direction_dt += 1

                print 'direction error_dt/total: %d/%d, accuracy_dt: %.4f .' % (
                    error_direction_dt, total_direction,
                    (float)(total_direction - error_direction_dt) / total_direction)
