#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:17:16 2018

@author: wufangyu
"""

import cv2
import os
from PIL import Image
import numpy as np
from sogou_utils.file_utils import mkdir
from sogou_utils.image_utils import get_numpy_form_img_dir_with_compare
from sogou_utils.video_utils import get_image_from_video
import logging
import json
from skimage import morphology
from scipy import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
class input_evaluation_video(object):
    video_img_list = []
    video_img_list_len = 0

    video_img_state_list = []
    video_keydown_list = []
    video_keyup_list = []
    video_fps = 0
    video_path = ""
    video_key_point_list = []
    state_block_list = []
    screen_pos = [0, 0, 0, 0]
    keyboard_pos = [0, 0, 0, 0]

    have_get_screen_pos = False
    have_get_keyboard_pos = False

    img_size_width = None
    img_size_hight = None

    is_key_down = 0
    last_red_point_num = -1
    red_reduce_num = 0
    is_rgb_channel = -1

    def __init__(self, video_path, need_rgb_to_gray, img_size_width, img_size_hight):
        if not os.path.exists(video_path):
            logger.error("path : %s not exists" % (video_path))
            return

        json_video_info = get_image_from_video(video_path)

        video_img_dir = json_video_info['image_path']
        self.video_fps = json_video_info['fps']

        self.img_size_width = img_size_width
        self.img_size_hight = img_size_hight

        compare_func = lambda x, y: -1 if int(x.split('.')[0]) < int(y.split('.')[0]) else 1
        self.video_img_list, labels = get_numpy_form_img_dir_with_compare(video_img_dir, compare_func, need_rgb_to_gray,
                                                                          img_size_width, img_size_hight)
        self.video_img_list_len = len(self.video_img_list)

        screen_area_rate = 0.0

        i = 0
        for img_np in self.video_img_list:

            i += 1
            if i < 1012:
                continue
            if not self.have_get_screen_pos:
                _, self.screen_pos = self.get_pos(img_np, 'screen')
                logger.debug("Screen_pos : %s" % (self.screen_pos))
                if self.screen_pos != None:
                    screen_area_rate = float(
                        (self.screen_pos[1] - self.screen_pos[0]) * (self.screen_pos[3] - self.screen_pos[2])) / float(
                        (self.img_size_width * self.img_size_hight))
                    if screen_area_rate > 0.2:
                        self.have_get_screen_pos = True

            _, self.edit_pos = self.get_pos(img_np, 'edit')
            logger.debug("Edit_pos : %s" % (self.edit_pos))
            
            if not self.have_get_screen_pos:
                self.screen_pos = self.extract_screen_contours_blue(img_np)
                logger.debug("Screen_pos : %s" % (self.screen_pos))
                if self.screen_pos != None:
                    screen_area_rate = float(
                        (self.screen_pos[1] - self.screen_pos[0]) * (self.screen_pos[3] - self.screen_pos[2])) / float(
                        (self.img_size_width * self.img_size_hight))
                    if screen_area_rate > 0.2:
                        self.have_get_screen_pos = True

            if not self.have_get_keyboard_pos and screen_area_rate > 0.3 and i / 2 > self.video_img_list_len / 3:
                self.edit_pos = self.extract_screen_contours_blue(img_np)
                keyboard_area_rate = float(
                    (self.edit_pos[1] - self.edit_pos[0]) * (self.edit_pos[3] - self.edit_pos[2])) / float(
                    (self.img_size_width * self.img_size_hight))
                logger.debug("Edit_pos : %s" % (self.edit_pos))
                if keyboard_area_rate / screen_area_rate < 0.8:
                    self.keyboard_pos = self.edit_pos[:]
                    self.keyboard_pos[2] = self.keyboard_pos[3]
                    self.keyboard_pos[3] = self.screen_pos[3]

                    self.edit_pos[3] -= (self.keyboard_pos[3] - self.keyboard_pos[2]) / 10
                    self.have_get_keyboard_pos = True
            
            if_get_pos, keyboard_pos = self.get_pos(img_np, 'keyboard')
            print(if_get_pos, keyboard_pos, i)
            if if_get_pos == 0:
                continue
            else:
                self.have_get_keyboard_pos = True
                self.keyboard_pos = keyboard_pos
                break

        if self.have_get_keyboard_pos:
            img_num = len(self.video_img_list)

            for i in range(img_num):
                if i % (img_num / 50) == 0:
                    logger.info("Get State: %.2f%%" % ((i * 100) / img_num))
                self.get_key_state(self.video_img_list[i])
                self.video_img_state_list.append(self.is_key_down)
                if self.video_img_state_list[i - 1] - self.is_key_down == 1:
                    self.video_keyup_list.append(i - self.red_reduce_num)

                if self.video_img_state_list[i - 1] - self.is_key_down == -1:
                    self.video_keydown_list.append(i)
        config_path = video_path.split('.' + video_path.split('.')[-1])[0] + '.json'
        with open(config_path, 'r') as config_file:
            video_info = json.load(config_file)

        with open(config_path, 'w') as config_file:
            video_info['keyup_list'] = self.video_keyup_list
            video_info['screen_pos'] = self.screen_pos
            video_info['keyboard_pos'] = self.keyboard_pos
            json_data = json.dumps(video_info)
            config_file.writelines(json_data)

    def extract_screen_contours_blue(self, numpy_image):
        binary_img = self.blue_to_binary(numpy_image)
        (_, contours, _) = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = sorted(contours, key=cv2.contourArea, reverse=True)
        contour_num = len(contour_list)

        area_all = (float)(self.img_size_hight * self.img_size_width)
        logger.info("Contour_num : %s" % (contour_num))
        for i in range(contour_num):
            contour_area_rate = cv2.contourArea(contour_list[i]) / area_all
            logger.debug("Contour_area_rate : %s" % (contour_area_rate))
            if contour_area_rate > 0.1 and contour_area_rate < 0.9:
                rect = cv2.minAreaRect(contour_list[i])
                box = np.int0(cv2.boxPoints(rect))

                horizontal_arr = [i[0] for i in box]
                vertical_arr = [i[1] for i in box]
                left_pos = min(horizontal_arr)
                right_pos = max(horizontal_arr)
                up_pos = min(vertical_arr)
                down_pos = max(vertical_arr)
                return [left_pos, right_pos, up_pos, down_pos]

    def blue_to_binary(self, numpy_image):
        r, g, b = cv2.split(numpy_image)
        if self.is_rgb_channel == -1:
            if b.sum() < r.sum():
                self.is_rgb_channel = False
            else:
                self.is_rgb_channel = True

        if not self.is_rgb_channel:
            r, g, b = b, g, r

        for x in range(b.shape[0]):
            for y in range(b.shape[1]):
                if (b[x, y] > 150 and g[x, y] < 100 and r[x, y] < 100):
                    b[x, y] = 0
                else:
                    b[x, y] = 255

        return b

    def get_pos(self, img_np, type):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        (gray_height, gray_width) = np.shape(gray)
        screen_thresh_gaussblur = cv2.GaussianBlur(gray, (9, 9), 0)  # sobel_filter
        x = cv2.Sobel(screen_thresh_gaussblur, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(screen_thresh_gaussblur, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        screen_sobel = cv2.addWeighted(absX, 2, absY, 2, 0)
        threshold, binary = cv2.threshold(screen_sobel, 127, 255, cv2.THRESH_BINARY)
        temp_binary = binary / 255
        mask = np.zeros((gray_height + 2, gray_width + 2), np.uint8)
        seed_point = 0, 0
        screen_height = 0
        cv2.floodFill(binary, mask, seed_point, 255, 0x08)  # flood_fill(background_filter)
        for i in range(gray_height):
            if (screen_height != 0):
                break
            for j in range(gray_width):
                if (binary[i, j] == 0):
                    screen_height = i
                    break
        binary = binary / 255
        for i in range(gray_height):  # keep_main_object
            for j in range(gray_width):
                if binary[i, j] == 1:
                    gray[i, j] = gray[0, 0]
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # houghlines
        edges = cv2.Canny(gray, 10, 50, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 400, gray_height / 50)
        tempImage = np.zeros((gray_height, gray_width, 3), dtype=np.uint8)
        a = len(lines)
        for i in range(a):
            for x1, y1, x2, y2 in lines[i]:
                if (x1 != x2):
                    k = (y1 - y2) / (x1 - x2)
                    b = y1 - k * x1
                    if (abs(k) < 0.3):  # extending_lines_and_filtering_angles
                        cv2.line(tempImage, (0, b), (gray_width, k * gray_width + b), (0, 255, 0), 1)
                if (y1 != y2):
                    k = (x1 - x2) / (y1 - y2)
                    b = x1 - k * y1
                    if (abs(k) < 0.3):
                        cv2.line(tempImage, (b, 0), (k * gray_height + b, gray_height), (0, 255, 0), 1)
        line_gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
        threshold, binary = cv2.threshold(line_gray, 127, 255, cv2.THRESH_BINARY)  # exact_line_skeleton
        kernel_size = gray_height / 80
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tempImage = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        tempImage = tempImage / 255
        tempImage = morphology.skeletonize(tempImage)
        conv_kernel = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        tempImage = signal.convolve2d(tempImage, conv_kernel, boundary='symm', mode='same')  # determine_point_pos
        point_pos = []
        row = 0
        temp_num = -gray_height / 25
        for i in range(gray_height)[::-1]:
            for j in range(gray_width):
                if (tempImage[i, j] > 6):
                    if (abs(i - temp_num) > gray_height / 25):
                        temp_num = i
                        row += 1
                    point_pos.append([row, i, j])
        keyboard_bottom = []
        keyboard_top = []
        if_get_pos = 1
        if row <= 5:
            if_get_pos = 0
            result = [0, 0, 0, 0]
        if (row > 5):
            for i in range(len(point_pos)):
                [x, y, z] = point_pos[i]
                if (x == 1):
                    keyboard_bottom.append(point_pos[i])
                if (x == 5):
                    keyboard_top.append(point_pos[i])
            pos = []
            pos.append(keyboard_top[0])
            pos.append(keyboard_bottom[len(keyboard_bottom) - 1])
            keyboard_pos_list = []
            for item in pos:
                for i in item:
                    keyboard_pos_list.append(i)
            if type == 'keyboard':
                result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], keyboard_pos_list[1]-(keyboard_pos_list[4]-keyboard_pos_list[1])/5, keyboard_pos_list[4]]
            if type == 'screen':
                result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], screen_height, keyboard_pos_list[4]]
            if type == 'edit':
                result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], screen_height, keyboard_pos_list[1]-(keyboard_pos_list[4]-keyboard_pos_list[1])/5]
            print(if_get_pos, result)
        return if_get_pos, result

    def get_key_state(self, orig_img):

        if self.edit_pos == [0, 0, 0, 0]:
            logger.error("Can not get edit pos")
            return -1
        edit_img = orig_img[self.edit_pos[2]:self.edit_pos[3], self.edit_pos[0]:self.edit_pos[1]]

        if self.is_rgb_channel:
            r, g, b = cv2.split(edit_img)
        else:
            r, g, b = cv2.split(edit_img)

        num = len(self.state_block_list)
        green_num = 0
        red_num = 0
        last_five_list = []

        if num <= 10:
            for x in xrange(0, g.shape[0], 5):
                for y in xrange(0, g.shape[1], 5):
                    state = 1
                    if (g[x, y] > 200 and r[x, y] < 200):
                        green_num += 1
                    elif (r[x, y] > 200 and g[x, y] < 200):
                        red_num += 1
                    else:
                        state = 0

                    last_five_list.append(state)
                    if len(last_five_list) > 5:
                        last_five_list.pop(0)

                    if sum(last_five_list) >= 3:
                        self.state_block_list.append([x, y])
            logger.debug("State_block_list : %s" % (self.state_block_list))
        else:
            for i in range(num):
                x = self.state_block_list[i][0]
                y = self.state_block_list[i][1]
                if (g[x, y] > 200 and r[x, y] < 200):
                    green_num += 1
                if (r[x, y] > 200 and g[x, y] < 200):
                    red_num += 1

        if self.last_red_point_num == -1:
            if red_num > green_num:
                self.is_key_down = 1
            else:
                self.is_key_down = 0
        else:
            if self.is_key_down == 0:
                if red_num > green_num:
                    self.is_key_down = 1
            else:
                if red_num < green_num:
                    self.is_key_down = 0
                elif self.last_red_point_num - red_num > self.last_red_point_num / 10:
                    self.red_reduce_num += 1
                else:
                    self.red_reduce_num = 0

        self.last_red_point_num = red_num

    def extrct_by_pos(self, extract_pos, img_dir_path, is_to_numpy=False, size=[28, 28]):
        if img_dir_path != "":
            mkdir(img_dir_path)

        extrct_img_list = []

        img_num = len(self.video_img_list)

        for i in xrange(img_num):
            orig_img = self.video_img_list[i]

            extrct_img = orig_img[extract_pos[2]:extract_pos[3], extract_pos[0]:extract_pos[1]]

            extrct_img_list.append(Image.fromarray(extrct_img))

            if img_dir_path != "":
                cv2.imwrite(os.path.join(img_dir_path,
                                         os.path.splitext(os.path.basename(self.video_path))[0] + '_' + str(
                                             i + 1) + '.jpeg'), extrct_img)

        if is_to_numpy:
            extrct_img_num = len(extrct_img_list)
            data = np.empty((extrct_img_num, 1, size[1], size[0]), dtype="float32")
            for i in range(extrct_img_num):
                img = extrct_img_list[i].resize(size, Image.ANTIALIAS).convert('L')
                arr = np.asarray(img, dtype="float32")
                data[i, :, :, :] = arr

            return data

    def extract_pinyin(self, img_dir_path, is_to_numpy=False):
        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return
        pinyin_pos = self.keyboard_pos[:]
        logger.debug("pinyin_pos : %s" % (pinyin_pos))

        pinyin_pos[1] -= (self.keyboard_pos[1] - self.keyboard_pos[0]) / 2
        logger.debug("pinyin_pos : %s" % (pinyin_pos))

        pinyin_pos[2] -= (self.keyboard_pos[3] - self.keyboard_pos[2]) / 10
        pinyin_pos[3] = self.keyboard_pos[2]
        logger.debug("pinyin_pos : %s" % (pinyin_pos))
        return self.extrct_by_pos(pinyin_pos, img_dir_path, is_to_numpy, [100, 12])

    def extract_cand(self, img_dir_path, is_to_numpy=False):
        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return
        cand_pos = self.keyboard_pos[:]

        cand_pos[3] = self.keyboard_pos[2] + (self.keyboard_pos[3] - self.keyboard_pos[2]) / 25 * 4
        # cv2.imwrite("D:\\cat2.jpg", )

        return self.extrct_by_pos(cand_pos, img_dir_path, is_to_numpy, [100, 12])

    def extract_symbol(self, img_dir_path, is_to_numpy=False):
        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return
        symbol_pos = self.keyboard_pos[:]

        symbol_pos[2] = self.keyboard_pos[2] + (self.keyboard_pos[3] - self.keyboard_pos[2]) / 5
        symbol_pos[3] = self.keyboard_pos[3] - (self.keyboard_pos[3] - self.keyboard_pos[2]) / 5
        symbol_pos[1] = self.keyboard_pos[0] + (self.keyboard_pos[1] - self.keyboard_pos[0]) / 5

        return self.extrct_by_pos(symbol_pos, img_dir_path, is_to_numpy, [40, 80])

    def extract_edit(self, img_dir_path, is_to_numpy=False):
        if self.edit_pos == [0, 0, 0, 0]:
            logger.error("Can not get edit pos")
            return

        eidt_half_pos = self.edit_pos[:]
        eidt_half_pos[3] -= (self.edit_pos[3] - self.edit_pos[2]) / 2
        return self.extrct_by_pos(eidt_half_pos, img_dir_path, is_to_numpy, [200, 75])

    def extract_screen(self, img_dir_path, is_to_numpy=False):

        if not self.have_get_screen_pos:
            logger.error("Can not get screen pos")
            return

        return self.extrct_by_pos(self.screen_pos, img_dir_path, is_to_numpy)

    def extract_keyboard(self, img_dir_path, is_to_numpy=False):

        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return

        return self.extrct_by_pos(self.keyboard_pos, img_dir_path, is_to_numpy)

    def get_keyup_list(self):
        return self.video_keyup_list

'''


class input_evaluation_video(object):
    video_img_list = []
    video_img_list_len = 0

    video_img_state_list = []
    video_keydown_list = []
    video_keyup_list = []
    video_fps = 0
    video_path = ""
    video_key_point_list = []
    state_block_list = []
    screen_pos = [0, 0, 0, 0]
    keyboard_pos = [0, 0, 0, 0]

    have_get_screen_pos = False
    have_get_keyboard_pos = False

    img_size_width = None
    img_size_hight = None

    is_key_down = 0
    last_red_point_num = -1
    red_reduce_num = 0
    is_rgb_channel = -1

    def __init__(self, video_path, need_rgb_to_gray, img_size_width, img_size_hight):
        if not os.path.exists(video_path):
            logger.error("path : %s not exists" % (video_path))
            return

        json_video_info = get_image_from_video(video_path)

        video_img_dir = json_video_info['image_path']
        self.video_fps = json_video_info['fps']

        self.img_size_width = img_size_width
        self.img_size_hight = img_size_hight

        compare_func = lambda x, y: -1 if int(x.split('.')[0]) < int(y.split('.')[0]) else 1
        self.video_img_list, labels = get_numpy_form_img_dir_with_compare(video_img_dir, compare_func, need_rgb_to_gray,
                                                                          img_size_width, img_size_hight)
        self.video_img_list_len = len(self.video_img_list)

        screen_area_rate = 0.0

        i = 0
        for img_np in self.video_img_list:

            i += 1
            if i < 1518:
                continue
            if not self.have_get_screen_pos:
                _, self.screen_pos = self.get_pos(img_np, 'screen')
                logger.debug("Screen_pos : %s" % (self.screen_pos))
                if self.screen_pos != None:
                    screen_area_rate = float(
                        (self.screen_pos[1] - self.screen_pos[0]) * (self.screen_pos[3] - self.screen_pos[2])) / float(
                        (self.img_size_width * self.img_size_hight))
                    if screen_area_rate > 0.2:
                        self.have_get_screen_pos = True

            _, self.edit_pos = self.get_pos(img_np, 'edit')
            logger.debug("Edit_pos : %s" % (self.edit_pos))
            '''
            if not self.have_get_screen_pos:
                self.screen_pos = self.extract_screen_contours_blue(img_np)
                logger.debug("Screen_pos : %s" % (self.screen_pos))
                if self.screen_pos != None:
                    screen_area_rate = float(
                        (self.screen_pos[1] - self.screen_pos[0]) * (self.screen_pos[3] - self.screen_pos[2])) / float(
                        (self.img_size_width * self.img_size_hight))
                    if screen_area_rate > 0.2:
                        self.have_get_screen_pos = True

            if not self.have_get_keyboard_pos and screen_area_rate > 0.3 and i / 2 > self.video_img_list_len / 3:
                self.edit_pos = self.extract_screen_contours_blue(img_np)
                keyboard_area_rate = float(
                    (self.edit_pos[1] - self.edit_pos[0]) * (self.edit_pos[3] - self.edit_pos[2])) / float(
                    (self.img_size_width * self.img_size_hight))
                logger.debug("Edit_pos : %s" % (self.edit_pos))
                if keyboard_area_rate / screen_area_rate < 0.8:
                    self.keyboard_pos = self.edit_pos[:]
                    self.keyboard_pos[2] = self.keyboard_pos[3]
                    self.keyboard_pos[3] = self.screen_pos[3]

                    self.edit_pos[3] -= (self.keyboard_pos[3] - self.keyboard_pos[2]) / 10
                    self.have_get_keyboard_pos = True
            '''
            if_get_pos, keyboard_pos = self.get_pos(img_np, 'keyboard')
            print(if_get_pos, keyboard_pos, i)
            if if_get_pos == 0:
                continue
            else:
                self.have_get_keyboard_pos = True
                self.keyboard_pos = keyboard_pos
                break

        if self.have_get_keyboard_pos:
            img_num = len(self.video_img_list)

            for i in range(img_num):
                if i % (img_num / 50) == 0:
                    logger.info("Get State: %.2f%%" % ((i * 100) / img_num))
                self.get_key_state(self.video_img_list[i])
                self.video_img_state_list.append(self.is_key_down)
                if self.video_img_state_list[i - 1] - self.is_key_down == 1:
                    self.video_keyup_list.append(i - self.red_reduce_num)

                if self.video_img_state_list[i - 1] - self.is_key_down == -1:
                    self.video_keydown_list.append(i)
        config_path = video_path.split('.' + video_path.split('.')[-1])[0] + '.json'
        with open(config_path, 'r') as config_file:
            video_info = json.load(config_file)

        with open(config_path, 'w') as config_file:
            video_info['keyup_list'] = self.video_keyup_list
            video_info['screen_pos'] = self.screen_pos
            video_info['keyboard_pos'] = self.keyboard_pos
            json_data = json.dumps(video_info)
            config_file.writelines(json_data)

    def extract_screen_contours_blue(self, numpy_image):
        binary_img = self.blue_to_binary(numpy_image)
        (_, contours, _) = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = sorted(contours, key=cv2.contourArea, reverse=True)
        contour_num = len(contour_list)

        area_all = (float)(self.img_size_hight * self.img_size_width)
        logger.info("Contour_num : %s" % (contour_num))
        for i in range(contour_num):
            contour_area_rate = cv2.contourArea(contour_list[i]) / area_all
            logger.debug("Contour_area_rate : %s" % (contour_area_rate))
            if contour_area_rate > 0.1 and contour_area_rate < 0.9:
                rect = cv2.minAreaRect(contour_list[i])
                box = np.int0(cv2.boxPoints(rect))

                horizontal_arr = [i[0] for i in box]
                vertical_arr = [i[1] for i in box]
                left_pos = min(horizontal_arr)
                right_pos = max(horizontal_arr)
                up_pos = min(vertical_arr)
                down_pos = max(vertical_arr)
                return [left_pos, right_pos, up_pos, down_pos]

    def blue_to_binary(self, numpy_image):
        r, g, b = cv2.split(numpy_image)
        if self.is_rgb_channel == -1:
            if b.sum() < r.sum():
                self.is_rgb_channel = False
            else:
                self.is_rgb_channel = True

        if not self.is_rgb_channel:
            r, g, b = b, g, r

        for x in range(b.shape[0]):
            for y in range(b.shape[1]):
                if (b[x, y] > 150 and g[x, y] < 100 and r[x, y] < 100):
                    b[x, y] = 0
                else:
                    b[x, y] = 255

        return b

    def get_pos(self, img_np, type):
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        (gray_height, gray_width) = np.shape(gray)
        screen_thresh_gaussblur = cv2.GaussianBlur(gray, (9, 9), 0)  # sobel_filter
        x = cv2.Sobel(screen_thresh_gaussblur, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(screen_thresh_gaussblur, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        screen_sobel = cv2.addWeighted(absX, 2, absY, 2, 0)
        threshold, binary = cv2.threshold(screen_sobel, 127, 255, cv2.THRESH_BINARY)
        temp_binary = binary / 255
        mask = np.zeros((gray_height + 2, gray_width + 2), np.uint8)
        seed_point = 0, 0
        screen_height = 0
        cv2.floodFill(binary, mask, seed_point, 255, 0x08)  # flood_fill(background_filter)
        for i in range(gray_height):
            if (screen_height != 0):
                break
            for j in range(gray_width):
                if (binary[i, j] == 0):
                    screen_height = i
                    break
        # binary = binary / 255
        # for i in range(gray_height):# keep_main_object
        #    for j in range(gray_width):
        #        if binary[i, j] == 1:
        #            gray[i, j] = gray[0, 0]
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # houghlines
        edges = cv2.Canny(gray, 10, 50, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 400, gray_height / 50)
        tempImage = np.zeros((gray_height, gray_width, 3), dtype=np.uint8)
        a = len(lines)
        for i in range(a):
            for x1, y1, x2, y2 in lines[i]:
                if (x1 != x2):
                    k = (y1 - y2) / (x1 - x2)
                    b = y1 - k * x1
                    if (abs(k) < 0.3):  # extending_lines_and_filtering_angles
                        cv2.line(tempImage, (0, b), (gray_width, k * gray_width + b), (0, 255, 0), 1)
                if (y1 != y2):
                    k = (x1 - x2) / (y1 - y2)
                    b = x1 - k * y1
                    if (abs(k) < 0.3):
                        cv2.line(tempImage, (b, 0), (k * gray_height + b, gray_height), (0, 255, 0), 1)
        line_gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
        threshold, binary = cv2.threshold(line_gray, 127, 255, cv2.THRESH_BINARY)  # exact_line_skeleton
        kernel_size = gray_height / 80
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tempImage = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        tempImage = tempImage / 255
        tempImage = morphology.skeletonize(tempImage)
        conv_kernel = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        tempImage = signal.convolve2d(tempImage, conv_kernel, boundary='symm', mode='same')  # determine_point_pos
        point_pos = []
        row = 0
        temp_num = -gray_height / 25
        for i in range(gray_height)[::-1]:
            for j in range(gray_width):
                if (tempImage[i, j] > 6):
                    if (abs(i - temp_num) > gray_height / 25):
                        temp_num = i
                        row += 1
                    point_pos.append([row, i, j])
        keyboard_bottom = []
        keyboard_top = []
        if_get_pos = 1
        if row <= 5:
            if_get_pos = 0
            result = [0, 0, 0, 0]
        if (row > 5):
            for i in range(len(point_pos)):
                [x, y, z] = point_pos[i]
                if (x == 1):
                    keyboard_bottom.append(point_pos[i])
                if (x == 6):
                    keyboard_top.append(point_pos[i])
            pos = []
            pos.append(keyboard_top[0])
            pos.append(keyboard_bottom[len(keyboard_bottom) - 1])
            keyboard_pos_list = []
            for item in pos:
                for i in item:
                    keyboard_pos_list.append(i)
            if type == 'keyboard':
                # result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], keyboard_pos_list[1]-(keyboard_pos_list[4]-keyboard_pos_list[1])/5, keyboard_pos_list[4]]
                result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5],
                          keyboard_pos_list[1], keyboard_pos_list[4]]
            if type == 'screen':
                result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], screen_height, keyboard_pos_list[4]]
            if type == 'edit':
                # result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], screen_height, keyboard_pos_list[1]-(keyboard_pos_list[4]-keyboard_pos_list[1])/5]
                result = [keyboard_pos_list[2] - 5, keyboard_pos_list[5], screen_height, keyboard_pos_list[1]]
            print(if_get_pos, result)
        return if_get_pos, result

    def get_key_state(self, orig_img):

        if self.edit_pos == [0, 0, 0, 0]:
            logger.error("Can not get edit pos")
            return -1
        edit_img = orig_img[self.edit_pos[2]:self.edit_pos[3], self.edit_pos[0]:self.edit_pos[1]]

        if self.is_rgb_channel:
            r, g, b = cv2.split(edit_img)
        else:
            r, g, b = cv2.split(edit_img)

        num = len(self.state_block_list)
        green_num = 0
        red_num = 0
        last_five_list = []

        if num <= 10:
            for x in xrange(0, g.shape[0], 5):
                for y in xrange(0, g.shape[1], 5):
                    state = 1
                    if (g[x, y] > 200 and r[x, y] < 200):
                        green_num += 1
                    elif (r[x, y] > 200 and g[x, y] < 200):
                        red_num += 1
                    else:
                        state = 0

                    last_five_list.append(state)
                    if len(last_five_list) > 5:
                        last_five_list.pop(0)

                    if sum(last_five_list) >= 3:
                        self.state_block_list.append([x, y])
            logger.debug("State_block_list : %s" % (self.state_block_list))
        else:
            for i in range(num):
                x = self.state_block_list[i][0]
                y = self.state_block_list[i][1]
                if (g[x, y] > 200 and r[x, y] < 200):
                    green_num += 1
                if (r[x, y] > 200 and g[x, y] < 200):
                    red_num += 1

        if self.last_red_point_num == -1:
            if red_num > green_num:
                self.is_key_down = 1
            else:
                self.is_key_down = 0
        else:
            if self.is_key_down == 0:
                if red_num > green_num:
                    self.is_key_down = 1
            else:
                if red_num < green_num:
                    self.is_key_down = 0
                elif self.last_red_point_num - red_num > self.last_red_point_num / 10:
                    self.red_reduce_num += 1
                else:
                    self.red_reduce_num = 0

        self.last_red_point_num = red_num

    def extrct_by_pos(self, extract_pos, img_dir_path, is_to_numpy=False, size=[28, 28]):
        if img_dir_path != "":
            mkdir(img_dir_path)

        extrct_img_list = []

        img_num = len(self.video_img_list)

        for i in xrange(img_num):
            orig_img = self.video_img_list[i]

            extrct_img = orig_img[extract_pos[2]:extract_pos[3], extract_pos[0]:extract_pos[1]]

            extrct_img_list.append(Image.fromarray(extrct_img))

            if img_dir_path != "":
                cv2.imwrite(os.path.join(img_dir_path,
                                         os.path.splitext(os.path.basename(self.video_path))[0] + '_' + str(
                                             i + 1) + '.jpeg'), extrct_img)

        if is_to_numpy:
            extrct_img_num = len(extrct_img_list)
            data = np.empty((extrct_img_num, 1, size[1], size[0]), dtype="float32")
            for i in range(extrct_img_num):
                img = extrct_img_list[i].resize(size, Image.ANTIALIAS).convert('L')
                arr = np.asarray(img, dtype="float32")
                data[i, :, :, :] = arr

            return data

    def extract_pinyin(self, img_dir_path, is_to_numpy=False):
        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return
        pinyin_pos = self.keyboard_pos[:]
        logger.debug("pinyin_pos : %s" % (pinyin_pos))

        pinyin_pos[1] -= (self.keyboard_pos[1] - self.keyboard_pos[0]) / 2
        logger.debug("pinyin_pos : %s" % (pinyin_pos))

        pinyin_pos[2] -= (self.keyboard_pos[3] - self.keyboard_pos[2]) / 10
        pinyin_pos[3] = self.keyboard_pos[2]
        logger.debug("pinyin_pos : %s" % (pinyin_pos))
        return self.extrct_by_pos(pinyin_pos, img_dir_path, is_to_numpy, [100, 12])

    def extract_cand(self, img_dir_path, is_to_numpy=False):
        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return
        cand_pos = self.keyboard_pos[:]

        cand_pos[3] = self.keyboard_pos[2] + (self.keyboard_pos[3] - self.keyboard_pos[2]) / 25 * 4
        # cv2.imwrite("D:\\cat2.jpg", )

        return self.extrct_by_pos(cand_pos, img_dir_path, is_to_numpy, [100, 12])

    def extract_symbol(self, img_dir_path, is_to_numpy=False):
        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return
        symbol_pos = self.keyboard_pos[:]

        symbol_pos[2] = self.keyboard_pos[2] + (self.keyboard_pos[3] - self.keyboard_pos[2]) / 5
        symbol_pos[3] = self.keyboard_pos[3] - (self.keyboard_pos[3] - self.keyboard_pos[2]) / 5
        symbol_pos[1] = self.keyboard_pos[0] + (self.keyboard_pos[1] - self.keyboard_pos[0]) / 5

        return self.extrct_by_pos(symbol_pos, img_dir_path, is_to_numpy, [40, 80])

    def extract_edit(self, img_dir_path, is_to_numpy=False):
        if self.edit_pos == [0, 0, 0, 0]:
            logger.error("Can not get edit pos")
            return

        eidt_half_pos = self.edit_pos[:]
        eidt_half_pos[3] -= (self.edit_pos[3] - self.edit_pos[2]) / 2
        return self.extrct_by_pos(eidt_half_pos, img_dir_path, is_to_numpy, [200, 75])

    def extract_screen(self, img_dir_path, is_to_numpy=False):

        if not self.have_get_screen_pos:
            logger.error("Can not get screen pos")
            return

        return self.extrct_by_pos(self.screen_pos, img_dir_path, is_to_numpy)

    def extract_keyboard(self, img_dir_path, is_to_numpy=False):

        if self.keyboard_pos == [0, 0, 0, 0]:
            logger.error("Can not get keyboard pos")
            return

        return self.extrct_by_pos(self.keyboard_pos, img_dir_path, is_to_numpy)

    def get_keyup_list(self):
        return self.video_keyup_list
