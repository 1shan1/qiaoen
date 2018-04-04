# -*- coding: utf-8 -*-

import random
import os
import argparse

import numpy as np
import shutil
import torch
import cv2

# yaw, pitch, roll
import torchvision
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision.transforms import transforms

import hopenet, datasets, utils

kan_dang_mean = [13.647625, -28.518246, -15.024275]
nei_shi_jing_mean = [23.282516, 11.826946, 5.225193]
qian_fang_mean = [-0.738090, -1.609898, -0.024009]
yi_biao_pan_mean = [-1.480816, -20.946795, -0.052280]
you_a_mean = [34.444702, -1.909611, -4.990962]
you_b_mean = [43.878971, -14.892829, -9.016929]  # raw: 38.54
zuo_a_mean = [-40.730156, -3.381142, 4.972855]
zuo_b_mean = [-53.785542, -17.713676, 2.538277]

directions_dict = {'kan-dang': kan_dang_mean, 'nei-shi-jing': nei_shi_jing_mean, 'qian-fang': qian_fang_mean,
                   'yi-biao-pan': yi_biao_pan_mean, 'you-a': you_a_mean, 'you-b': you_b_mean, 'zuo-a': zuo_a_mean,
                   'zuo-b': zuo_b_mean}

directions_id = ['kan-dang', 'nei-shi-jing', 'qian-fang', 'yi-biao-pan', 'you-a', 'you-b', 'zuo-a', 'zuo-b']


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


def cal_min_distance(euler_angels):
    min_key = 0
    min_value = 10000000

    kan_dang_dis = 0
    you_a_dis = 0

    for key, value in directions_dict.iteritems():
        if key == 'kan-dang':
            kan_dang_dis = np.sqrt(np.sum(np.square(np.asarray(euler_angels) - np.asarray(value)))) / len(euler_angels)
        elif key == 'you-a':
            you_a_dis = np.sqrt(np.sum(np.square(np.asarray(euler_angels) - np.asarray(value)))) / len(euler_angels)
        elif key == 'nei-shi-jing':
            nei_shi_jing_dis = np.sqrt(np.sum(np.square(np.asarray(euler_angels) - np.asarray(value)))) / len(
                euler_angels)
        elif key == 'qian-fang':
            qian_fang_dis = np.sqrt(np.sum(np.square(np.asarray(euler_angels) - np.asarray(value)))) / len(euler_angels)

        distance_value = np.sqrt(np.sum(np.square(np.asarray(euler_angels) - np.asarray(value)))) / len(euler_angels)
        if distance_value < min_value:
            min_value = distance_value
            min_key = key

            # print 'min_key: %s, min_value: %f' % (min_key, min_value)
    return min_key, min_value


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
    parser.add_argument('--calculate_mean', dest='calculate_mean', default=False,
                        help='calculate 8 means(3-D) of 8 directions.')
    parser.add_argument('--cal_accuracy', dest='cal_accuracy', default=False,
                        help='calculate every direction\'s accuracy by euler distance.')
    parser.add_argument('--dt_accuracy', dest='dt_accuracy', default=False)
    parser.add_argument('--direction', dest='direction', default=False)

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

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    driver_train_dicts = {}  # calculating means for 8 directions

    if args.cal_accuracy is True:
        error_num = 0
        total_num = 0
        pre_dir = ''
        test_result_dicts = {'key': [], 'error_num': [], 'total_num': []}

    out_lines = []

    # for test accuracy
    error_direction = 0
    total_direction = len(pose_dataset)

    if bool(args.dt_accuracy):
        ori_img_txt = '/media/f/face_angle_dataset/driver_train_filter.txt'

        dt_pose_dataset = datasets.DRIVER_TEST(args.data_dir, ori_img_txt, transformations, args.direction)
        dt_loader = torch.utils.data.DataLoader(dataset=dt_pose_dataset,
                                                batch_size=1,
                                                num_workers=2)
        train_data = []
        target_data = []

        for i, (images, labels, cont_labels, name) in enumerate(dt_loader):
            direction_label = cont_labels[0, 3]  # torch.Size() (1, 4)

            parent_dir = os.path.basename(os.path.dirname(name[0]))

            images = Variable(images).cuda(gpu)

            yaw, pitch, roll, direction = model(images, Variable(idx_tensor))

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

            if str(roll_predicted[0]) == 'nan':
                roll_predicted[0] = 0

            train_data.append([float(yaw_predicted[0]), float(pitch_predicted[0]), float(roll_predicted[0])])
            target_data.append([int(direction_label)])

        from decision_tree_demo import tree_demo

        clf = tree_demo(train_data, target_data)

        from sklearn.externals import joblib

        joblib.dump(clf, './../driver_data.m')

        # 根据决策树过滤训练集
        dt_out_lines = []

    for i, (images, name, direction_label, item) in enumerate(test_loader):
        parent_dir = os.path.basename(os.path.dirname(name[0]))

        if not driver_train_dicts.has_key(parent_dir):
            driver_train_dicts[parent_dir] = [[], [], []]

            # if args.cal_accuracy:
            #     if i != 0:
            #         test_result_dicts['key'].append(pre_dir)
            #         test_result_dicts['error_num'].append(error_num)
            #         test_result_dicts['total_num'].append(total_num)
            #         total_num = 0
            #         error_num = 0

        images = Variable(images).cuda(gpu)

        yaw, pitch, roll, direction = model(images, Variable(idx_tensor))

        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

        if bool(args.dt_accuracy) is True:
            prediction = int(clf.predict(np.asarray(
                [[float(yaw_predicted[0]), float(pitch_predicted[0]), float(roll_predicted[0])]]))
            )

            # print prediction, ' ', int(direction_label[0])

            if prediction == int(direction_label[0]):
                for j in range(len(yaw_predicted)):
                    dt_out_lines.append(
                        name[j] + ' ' + str(yaw_predicted[j]) + ' ' + str(pitch_predicted[j]) + ' ' + \
                        str(roll_predicted[j]) + ' ' + str(direction_label[j]) + '\n')
            else:
                error_direction += 1

        if bool(args.calculate_mean) is True:
            driver_train_dicts[parent_dir][0].append(yaw_predicted)
            driver_train_dicts[parent_dir][1].append(pitch_predicted)
            driver_train_dicts[parent_dir][2].append(roll_predicted)
            print i, len(pose_dataset)

        if bool(args.cal_accuracy) is True and args.batch_size == 1:
            min_key, min_value = cal_min_distance([yaw_predicted, pitch_predicted, roll_predicted])  # 计算预测方向，最小距离

            if min_key == parent_dir:
                out_lines.append(item[0] + '\n')
                print item[0], min_key
            else:
                error_direction += 1

                output_dir = '/media/f/face_angle_dataset/car_driver_frame_cut_error_euler'
                output_direction_dir = os.path.join(output_dir, parent_dir)

                # if not os.path.exists(output_direction_dir):
                #     os.mkdir(output_direction_dir)

                # shutil.copy(name[0], os.path.join(output_direction_dir,
                #                                   os.path.basename(name[0])[:-4] + '_'
                #                                   + min_key + '.jpg'))

        # Save first image in batch with pose cube or axis.
        if args.save_viz:
            name = name[0]
            # cv2_img = cv2.imread(os.path.join(args.data_dir, name))
            cv2_img = cv2.imread(os.path.join(name))
            if args.batch_size == 1:
                min_key, min_value, kan_dang_dis, you_a_dis, nei_shi_jing_dis, qian_fang_dis = cal_min_distance(
                    [yaw_predicted, pitch_predicted, roll_predicted])

                src_key = os.path.basename(args.data_dir)

                # if min_key == os.path.basename(parent_dir):
                #     continue
                print 'final_min_key: %s, final_min_value: %f' % (min_key, min_value)

                # print 'direction error/total: %d/%d, accuracy: %.4f .' % (
                #     error_direction, total_direction, (float)(total_direction - error_direction) / total_direction)

    if bool(args.calculate_mean) is True:
        for key, value in driver_train_dicts.iteritems():
            # calculate 8-means
            yaw_mean = np.mean(value[0])
            pitch_mean = np.mean(value[1])
            roll_mean = np.mean(value[2])
            print 'direction: %s, yaw_mean: %f, pitch_mean: %f, roll' \
                  '_mean: %f.' % (key, yaw_mean, pitch_mean, roll_mean)

            out_txt = '/media/f/face_angle_dataset/driver_train_filter.txt'
            if os.path.exists(out_txt):
                os.remove(out_txt)

            with open(out_txt, 'a+') as out_file:
                out_file.writelines(''.join(out_lines))

            out_file.close()

    if bool(args.dt_accuracy):
        dt_out_txt = '/media/f/face_angle_dataset/driver_train_ori_angle.txt'

        if os.path.exists(dt_out_txt):
            os.remove(dt_out_txt)

        with open(dt_out_txt, 'a+') as out_file:
            out_file.writelines(''.join(dt_out_lines))

        out_file.close()
        print 'direction error/total: %d/%d, accuracy: %.4f .' % (
            error_direction, total_direction, (float)(total_direction - error_direction) / total_direction)
