# -*- coding: utf-8 -*-
import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import random

import datasets, hopenet, utils

directions_id = ['kan-dang', 'nei-shi-jing', 'qian-fang', 'yi-biao-pan', 'you-a', 'you-b', 'zuo-a', 'zuo-b']


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
                        default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list',
                        help='Path to text file containing relative paths for every example.',
                        default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Driver_Test', type=str)
    parser.add_argument('--test_txt', dest='test_txt', type=str, default='', help='path of test dataset.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id

    snapshot_path = args.snapshot

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print 'Loading snapshot.'
    # Load snapshot
    save_state_dict = torch.load(snapshot_path)
    model.load_state_dict(save_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(140),
                                          transforms.CenterCrop(128), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Driver_Test':
        print args.data_dir, args.filename_list
        pose_dataset = datasets.Driver_Test(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'By_TXT':
        pose_dataset = datasets.Driver_Train_TXT(args.filename_list, transformations)

    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=2)

    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    l1loss = torch.nn.L1Loss(size_average=False)

    softmax = nn.Softmax().cuda(gpu)

    x_list = []
    yaw_predicted_list = []
    pitch_predicted_list = []
    roll_predicted_list = []
    out_lines = []

    error_num = 0
    total_num = 0

    error_direction = 0
    total_direction = len(pose_dataset)

    test_result_dicts = {'key': [], 'error_num': [], 'total_num': []}

    driver_train_dicts = {}

    pre_dir = ''

    if args.test_txt == '':
        angles_output_txt = '/media/f/face_angle_dataset/car_driver_frame0313_cut/driver_test_0323.txt'
        if os.path.exists(angles_output_txt):
            os.remove(angles_output_txt)

        for i, (images, name, direction_label, item) in enumerate(test_loader):
            parent_dir = os.path.basename(os.path.dirname(name[0]))

            if not driver_train_dicts.has_key(parent_dir):
                driver_train_dicts[parent_dir] = [[], [], []]
                if i != 0:
                    test_result_dicts['key'].append(pre_dir)
                    test_result_dicts['error_num'].append(error_num)
                    test_result_dicts['total_num'].append(total_num)
                    total_num = 0
                    error_num = 0

            images = Variable(images).cuda(gpu)
            # total += len(pose_dataset)

            yaw, pitch, roll, direction = model(images, Variable(idx_tensor))

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            direction = softmax(direction)

            _, direciton_bpred = torch.max(direction.data, 1)
            direction_value_bpred = int(direciton_bpred[0])

            if int(direction_value_bpred) != int(direction_label[0]):
                error_direction += 1
                print direction_value_bpred, direction_label[0]

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

            x_list.append(i + 1)
            yaw_predicted_list.append(yaw_predicted)
            pitch_predicted_list.append(pitch_predicted)
            roll_predicted_list.append(roll_predicted)

            for j in range(len(yaw_predicted)):
                # out_lines.append(
                #     name[j] + ' ' + str(yaw_predicted[j]) + ' ' + str(pitch_predicted[j]) + ' ' + str(
                #         roll_predicted[j]) + ' ' + str(directions_id.index(parent_dir)) + '\n')
                out_lines.append(
                    name[j] + ' ' + str(yaw_predicted[j]) + ' ' + str(pitch_predicted[j]) + ' ' + str(
                        roll_predicted[j]) + ' ' + str(direction_label[j]) + '\n')

            total_num += 1

            # print 'direction: %s, yaw: %f, pitch: %f, roll: %f, name: %s' % (
            #     parent_dir, yaw_predicted, pitch_predicted, roll_predicted, name[0])

            # Save first image in batch with pose cube or axis.
            if args.save_viz:
                name = name[0]
                # cv2_img = cv2.imread(os.path.join(args.data_dir, name))
                cv2_img = cv2.imread(os.path.join(name))
                if args.batch_size == 1:
                    error_num += 1

                    error_string = 'y %.0f, p %.0f, r %.0f, %s' % (
                        torch.sum(yaw_predicted),
                        torch.sum(pitch_predicted),
                        torch.sum(roll_predicted)
                    )

                    cv2.putText(cv2_img, error_string, (10, cv2_img.shape[0] - 30), fontFace=1, fontScale=1,
                                color=(0, 0, 255), thickness=2)

                utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx=112, tdy=112,
                                size=100)

                dst_dir = '/media/d/face_angle/deep-head-pose/code_dir/output/driver_test/' + parent_dir
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)

                final_name = os.path.join(dst_dir, os.path.basename(name))

                # cv2.imshow('img', cv2_img)
                # cv2.waitKey(0)
                # if torch.abs(yaw_predicted)[0] <= 3 and torch.abs(pitch_predicted)[0] <= 3 \
                #         and torch.abs(roll_predicted)[0] < 3:
                # cv2.imwrite(final_name, cv2_img)

            print i, len(pose_dataset)
            if i == len(pose_dataset) - 2:
                test_result_dicts['key'].append(parent_dir)
                test_result_dicts['error_num'].append(error_num)
                test_result_dicts['total_num'].append(total_num)

            pre_dir = parent_dir

        random.shuffle(out_lines)

        print 'direction error/total: %d/%d, accuracy: %.4f .' % (
            error_direction, total_direction, (float)(total_direction - error_direction) / total_direction)

        with open(angles_output_txt, 'a+') as out_file:
            out_file.writelines(''.join(out_lines))

        out_file.close()

        error_list = np.asarray(test_result_dicts['error_num'])
        total_list = np.asarray(test_result_dicts['total_num'])
        accuracy = np.true_divide((total_list - error_list), total_list)

        print test_result_dicts, accuracy

        # for keys, values in test_result_dicts.iteritems():
        #     print keys, values
        #     for i, key in enumerate(keys):
        #         print 'key: %s, error_num/total_num: %f/%f, accuracy: %f.' % (
        #             values[0][i], values[1][i], values[2][i],
        #             values[1][i] / (float)(values[2][i]))
        #
        # print 'test_result_dicts: ', test_result_dicts
        #
        # print 'yaw_mean: %f, pitch_mean: %f, roll_mean: %f.' % (yaw_mean, pitch_mean, roll_mean)
        # print 'error_num/total_num : %d/%d, accuracy: %f' % (
        #     error_num, total_num, (float)(total_num - error_num) / total_num)

        # outfile = os.path.join('output/driver_test', os.path.basename(args.data_dir) + '.txt')
        # with open(outfile, 'a+') as out_stream:
        #     out_stream.writelines(os.path.basename(args.data_dir) + ' yaw_mean: %f, pitch_mean: %f, roll_mean: %f.\n' % (
        #         yaw_mean, pitch_mean, roll_mean))
        # out_stream.close()
        #
        # 创建图
        # plt.figure('Angle distribute')
        # plt.subplot()
        #
        # axis = plt.gca()
        # # 设置x轴, y轴
        # axis.set_xlabel('x_neishijing')
        # axis.set_ylabel('y_qianfang')
        #
        # axis.scatter(x_neishijing_list, y_neishijing_list, s=10, color='b')
        # axis.scatter(x_qianfang_list, y_qianfang_list, s=10, color='r')
        #
        # plt.show()
    else:
        test_dataset = datasets.Driver_Train_TXT(args.test_txt, transformations)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=2)

        print 'Test driver_test dataset...'
        print 'Show error predicted...'

        error_direction = 0
        total_direction = len(test_dataset)
        output_dir = '/media/f/face_angle_dataset/car_driver_test_error/car_driver_0313_error'

        save_error_img = True

        for i, (images, name, direction_label, item) in enumerate(test_loader):
            parent_dir = os.path.basename(os.path.dirname(name[0]))
            images = Variable(images).cuda(gpu)

            yaw, pitch, roll, direction = model(images, idx_tensor)

            direction = softmax(direction)
            _, direciton_bpred = torch.max(direction.data, 1)
            direction_value_bpred = int(direciton_bpred.cpu())

            if int(direction_value_bpred) != int(direction_label[0]):
                print 'raw/predict: %s/%s' % (parent_dir, directions_id[(int(direction_value_bpred))])

                error_direction += 1
                if save_error_img is True:
                    output_direction_dir = os.path.join(output_dir, parent_dir)

                    if not os.path.exists(output_direction_dir):
                        os.mkdir(output_direction_dir)

                    shutil.copy(name[0], os.path.join(output_direction_dir,
                                                      os.path.basename(name[0])[:-4] + '_'
                                                      + directions_id[direction_value_bpred] + '.jpg'))

        print 'direction error/total: %d/%d, accuracy: %.4f .' % (
            error_direction, total_direction, (float)(total_direction - error_direction) / total_direction)
