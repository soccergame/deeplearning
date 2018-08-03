#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 21:02:27 2018

@author: wufangyu
"""
import json
import os
import datetime
from evaluation.input_evaluation_video import input_evaluation_video
from sogou_ai.siamese_cnn import load_model_with_contrastive_loss
import logging
'''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def vaild_key_up_list_9(input_list):#key_list should be written in config.json
    is_vaild_key_up_list = [0]
    for item_list in input_list:
        for item in item_list:
            if str(item).isdigit() :
                is_vaild_key_up_list.append(1)
            else:
                is_vaild_key_up_list.append(0)
        is_vaild_key_up_list.append(0)
    is_vaild_key_up_list.append(0)
    bool_is_vaild_key_up_list = [False] * len(is_vaild_key_up_list)
    for item in range(len(is_vaild_key_up_list)):
        if is_vaild_key_up_list[item]:
            bool_is_vaild_key_up_list[item] = True
    return bool_is_vaild_key_up_list


def vaild_key_up_list_26(input_list):#key_list should be written in config.json
    is_vaild_key_up_list = [0]
    exchange_list = {'abc': 2, 'def':3,'ghi':4,'jkl':5,'mno':6,'pqrs':7,'tuv':8,'wxyz':9,'&':'&'}

    for item_list in input_list:
        for item in item_list:
            for word in exchange_list:
                if item in word:
                    is_vaild_key_up_list.append(1)
                    #if str(exchange_list[word]).isdigit() :
                    #    is_vaild_key_up_list.append(1)

                    #else:
                    #    is_vaild_key_up_list.append(0)
        is_vaild_key_up_list.append(0)
    is_vaild_key_up_list.append(0)
    bool_is_vaild_key_up_list = [False] * len(is_vaild_key_up_list)
    for item in range(len(is_vaild_key_up_list)):
        if is_vaild_key_up_list[item]:
            bool_is_vaild_key_up_list[item] = True
    return bool_is_vaild_key_up_list


def whether_vaild_key_up_list(input_list_file, keyboard_type):
    input_list = []
    with open(input_list_file) as input_file:
        for line in input_file.readlines():
            input_list.append(line.split()[0])
    if keyboard_type == 26:
        is_vaild_key_up_list = vaild_key_up_list_26(input_list)
    elif keyboard_type == 9:
        is_vaild_key_up_list = vaild_key_up_list_9(input_list)
    else:
        logger.error("wrong keyboard type, only 9 or 26 permit")
        return None
    return input_list, is_vaild_key_up_list


def get_change_list_by_siamese(np_list, keyup_list, is_vaild_key_up_list, model_name):

    model = load_model_with_contrastive_loss(model_name)

    result_key_up_list = []
    result_change_list = []

    keyup_list_len = len(keyup_list)
    keyup_list.append(len(np_list) - 1)
    for i in range(keyup_list_len - 1):

        if not is_vaild_key_up_list[i]:
           continue

        pred_sub = {}
        pred_sub_tag= 0
        pred_sub_refer = 0.0
        for j in range(keyup_list[i]-15, keyup_list[i+1])[::-1]:
            pred = model.predict([np_list[j].reshape(1, 12, 100, 1), np_list[j - 1].reshape(1, 12, 100, 1)])
            pred_sub[j] = pred
        for m in range(keyup_list[i]-15,keyup_list[i]+15,1):

            if (pred_sub[m][0,0] > pred_sub_refer):
                pred_sub_refer = pred_sub[m][0,0]
                pred_sub_tag = m
        tag = pred_sub_tag
        for n in range(1,10,1):
            if ((pred_sub[pred_sub_tag+n][0,0]-pred_sub[pred_sub_tag+n+1][0,0])<0.3):
                tag=pred_sub_tag+n
                break
        #tag = max(pred_sub.items(), key=lambda x:  x[1])[0]
        logger.debug("result_change_list : %s" % (pred_sub[tag]))

        result_key_up_list.append(keyup_list[i])
        result_change_list.append(tag)

    logger.debug("result_change_list : %s" % (result_change_list))
    logger.debug("result_key_up_list : %s" % (result_key_up_list))

    return result_key_up_list, result_change_list


def get_config(project_name):
    with open('config.json') as json_file:
        json_config = json.load(json_file)
        module_config = json_config[project_name]

    return module_config


def get_response_time(key_start_list, change_list, fps):

    response_time_list = list(map(lambda x: x[0] - x[1], zip(change_list, key_start_list)))

    for item in range(len(response_time_list)):
        response_time_list[item] = round(response_time_list[item] * 1000 / float(fps), 2)
    logger.debug('response_time_list : %s' % response_time_list)
    return response_time_list

def write_line(list_to_write, f, line_name):
    if len(list_to_write) < 1:
        logger.error('List : %s is Null' % line_name)
    else:
        f.writelines(line_name)
        for item in range(len(list_to_write)):
            f.writelines(',' + str(list_to_write[item]))
        f.writelines('\n')


def get_result(cand_model, pinyin_model, is_vaild_key_up_list, video_path, resultPath, input_name, input_version, input_list):
    config = get_config('sogou_input_response_time')
    img_size_width = config['img_size_width']
    img_size_hight = config['img_size_hight']
    input_dim = tuple(config['input_dim'])
    need_rgb_to_gray = config['need_rgb_to_gray']

    evaluation_video = input_evaluation_video(video_path, need_rgb_to_gray, img_size_width, img_size_hight)
    cand_data = evaluation_video.extract_cand('', True)
    cand_data /= 255
    pinyin_data = evaluation_video.extract_pinyin('', True)
    pinyin_data /= 255
    keyup_list = evaluation_video.get_keyup_list()
    cand_key_up_list, cand_change_list = get_change_list_by_siamese(cand_data, keyup_list, is_vaild_key_up_list,cand_model)
    pinyin_key_up_list, pinyin_change_list = get_change_list_by_siamese(pinyin_data, keyup_list, is_vaild_key_up_list,pinyin_model)

    pinyin_response_time_list = get_response_time(pinyin_key_up_list, pinyin_change_list, evaluation_video.video_fps)
    cand_response_time_list = get_response_time(cand_key_up_list, cand_change_list, evaluation_video.video_fps)

    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    today = datetime.date.today()
    result_name = today.strftime("%Y_%m_%d")
    videoname = video_path.split('/')[-1].split('.')[0]
    with open(resultPath + str(result_name) + '_' + videoname + '.json', 'w') as jsonFile:
        return_info = dict()
        result_info = []
        aver_result = []
        all_key_name = []
        record_count = 0
        cand_sum = []
        pinyin_sum = []
        key_count = []
        aver_result_info = []

        for item in range(len(input_list)):
            word_info = dict()
            key_info = dict()

            key_name = []
            cand = []
            pinyin = []
            start = []
            cand_change = []
            pinyin_change = []

            for key_item in range(len(input_list[item])):

                key_name.append(input_list[item][key_item])
                cand.append(cand_response_time_list[record_count])
                pinyin.append(pinyin_response_time_list[record_count])
                start.append(cand_key_up_list[record_count])
                cand_change.append(cand_change_list[record_count])
                pinyin_change.append(pinyin_change_list[record_count])

                if input_list[item][key_item] not in all_key_name:
                    all_key_name.append(input_list[item][key_item])
                    cand_sum.append(cand_response_time_list[record_count])
                    pinyin_sum.append(pinyin_response_time_list[record_count])
                    key_count.append(1)
                else:
                    key_index = all_key_name.index(input_list[item][key_item])
                    cand_sum[key_index] += cand_response_time_list[record_count]
                    pinyin_sum[key_index] += pinyin_response_time_list[record_count]
                    key_count[key_index] += 1
                record_count += 1

            key_info['key_name'] = key_name
            key_info['cand'] = cand
            key_info['pinyin'] = pinyin
            key_info['start'] = start
            key_info['cand_change'] = cand_change
            key_info['pinyin_change'] = pinyin_change
            word_info['word_list'] = input_list[item]
            word_info['word_result'] = key_info
            result_info.append(word_info)

        for key_item in range(len(all_key_name)):
            key_aver_result = dict()
            key_aver_result['key_name'] = all_key_name[key_item]
            key_aver_result['aver_cand'] = round(cand_sum[key_item] / key_count[key_item], 2)
            key_aver_result['aver_pinyin'] = round(pinyin_sum[key_item] / key_count[key_item], 2)
            aver_result_info.append(key_aver_result)

        return_info['input_name'] = input_name
        return_info['input_version'] = input_version
        return_info['aver_result_info'] = aver_result_info
        return_info['result_info'] = result_info

        json_data = json.dumps(return_info)
        jsonFile.writelines(json_data)

'''
import json
import os
import datetime
from input_evaluation_video import input_evaluation_video
from sogou_ai.siamese_cnn import load_model_with_contrastive_loss
import logging

logger = logging.getLogger(__name__)

'''
def vaild_key_up_list_9(input_list):#key_list should be written in config.json
    is_vaild_key_up_list = [0]
    for item_list in input_list:
        for item in item_list:
            if str(item).isdigit() or str(item)=='&' :
                is_vaild_key_up_list.append(1)
            else:
                is_vaild_key_up_list.append(0)
        is_vaild_key_up_list.append(0)
    is_vaild_key_up_list.append(0)
    bool_is_vaild_key_up_list = [False] * len(is_vaild_key_up_list)
    for item in range(len(is_vaild_key_up_list)):
        if is_vaild_key_up_list[item]:
            bool_is_vaild_key_up_list[item] = True
    return bool_is_vaild_key_up_list

def vaild_key_up_list_26(input_list):#key_list should be written in config.json
    is_vaild_key_up_list = [0]
    exchange_list = {'abc': 2, 'def':3,'ghi':4,'jkl':5,'mno':6,'pqrs':7,'tuv':8,'wxyz':9,'&':'&'}

    for item_list in input_list:
        for item in item_list:
            for word in exchange_list:
                if item in word:
                    is_vaild_key_up_list.append(1)
                    #if str(exchange_list[word]).isdigit()  :
                    #   is_vaild_key_up_list.append(1)

                    #else:
                     #   is_vaild_key_up_list.append(0)
        is_vaild_key_up_list.append(0)
    is_vaild_key_up_list.append(0)
    bool_is_vaild_key_up_list = [False] * len(is_vaild_key_up_list)
    for item in range(len(is_vaild_key_up_list)):
        if is_vaild_key_up_list[item]:
            bool_is_vaild_key_up_list[item] = True
    return bool_is_vaild_key_up_list


def whether_vaild_key_up_list(input_list_file, keyboard_type):
    input_list = []
    with open(input_list_file) as input_file:
        for line in input_file.readlines():
            input_list.append(line.split()[0])
    if keyboard_type == 26:
        is_vaild_key_up_list = vaild_key_up_list_26(input_list)
    elif keyboard_type == 9:
        is_vaild_key_up_list = vaild_key_up_list_9(input_list)
    else:
        logger.error("wrong keyboard type, only 9 or 26 permit")
        return None
    return input_list, is_vaild_key_up_list


def get_change_list_by_siamese(np_list, keyup_list, is_vaild_key_up_list, model_name):

    model = load_model_with_contrastive_loss(model_name)

    result_key_up_list = []
    result_change_list = []

    keyup_list_len = len(keyup_list)
    keyup_list.append(len(np_list) - 1)
    for i in range(keyup_list_len - 1):

        if not is_vaild_key_up_list[i]:
           continue

        pred_sub = {}
        for j in range(keyup_list[i], keyup_list[i+1])[::-1]:
            pred = model.predict([np_list[j].reshape(1, 12, 100, 1), np_list[j - 1].reshape(1, 12, 100, 1)])
            pred_sub[j] = pred

        tag = max(pred_sub.items(), key=lambda x: x[1])[0]
        logger.debug("result_change_list : %s" % (pred_sub[tag]))

        result_key_up_list.append(keyup_list[i])
        result_change_list.append(tag)

    logger.debug("result_change_list : %s" % (result_change_list))
    logger.debug("result_key_up_list : %s" % (result_key_up_list))

    return result_key_up_list, result_change_list


def get_config(project_name):
    with open('config.json') as json_file:
        json_config = json.load(json_file)
        module_config = json_config[project_name]

    return module_config


def get_response_time(key_start_list, change_list, fps):

    response_time_list = list(map(lambda x: x[0] - x[1], zip(change_list, key_start_list)))

    for item in range(len(response_time_list)):
        response_time_list[item] = round(response_time_list[item] * 1000 / float(fps), 2)
    logger.debug('response_time_list : %s' % response_time_list)
    return response_time_list

def write_line(list_to_write, f, line_name):
    if len(list_to_write) < 1:
        logger.error('List : %s is Null' % line_name)
    else:
        f.writelines(line_name)
        for item in range(len(list_to_write)):
            f.writelines(',' + str(list_to_write[item]))
        f.writelines('\n')


def get_result(cand_model, pinyin_model, is_vaild_key_up_list, video_path, resultPath, input_name, input_version, input_list):
    config = get_config('sogou_input_response_time')
    img_size_width = config['img_size_width']
    img_size_hight = config['img_size_hight']
    input_dim = tuple(config['input_dim'])
    need_rgb_to_gray = config['need_rgb_to_gray']

    evaluation_video = input_evaluation_video(video_path, need_rgb_to_gray, img_size_width, img_size_hight)
    cand_data = evaluation_video.extract_cand('', True)
    cand_data /= 255
    pinyin_data = evaluation_video.extract_pinyin('', True)
    pinyin_data /= 255
    keyup_list = evaluation_video.get_keyup_list()
    cand_key_up_list, cand_change_list = get_change_list_by_siamese(cand_data, keyup_list, is_vaild_key_up_list,cand_model)
    pinyin_key_up_list, pinyin_change_list = get_change_list_by_siamese(pinyin_data, keyup_list, is_vaild_key_up_list,pinyin_model)

    pinyin_response_time_list = get_response_time(pinyin_key_up_list, pinyin_change_list, evaluation_video.video_fps)
    cand_response_time_list = get_response_time(cand_key_up_list, cand_change_list, evaluation_video.video_fps)

    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    today = datetime.date.today()
    result_name = today.strftime("%Y_%m_%d")
    videoname = video_path.split('/')[-1].split('.')[0]
    
    with open(resultPath + str(result_name)+'.csv','a') as f:
        f.writelines('video_name,'+str(videoname)+'\n')
        write_line(cand_response_time_list, f, 'cand')
        write_line(pinyin_response_time_list, f, 'pinyin')
        write_line(cand_key_up_list, f, 'start')
        write_line(cand_change_list, f, 'cand_change')
        #write_line(pinyin_key_up_list, f, 'start')
        write_line(pinyin_change_list, f, 'pinyin_change')
        f.writelines('\n')
    
    with open(resultPath + str(result_name) + '_' + videoname + '.json', 'w') as jsonFile:
        return_info = dict()
        result_info = []
        aver_result = []
        all_key_name = []
        record_count = 0
        cand_sum = []
        pinyin_sum = []
        key_count = []
        aver_result_info = []

        for item in range(len(input_list)):
            word_info = dict()
            key_info = dict()

            key_name = []
            cand = []
            pinyin = []
            start = []
            cand_change = []
            pinyin_change = []

            for key_item in range(len(input_list[item])):

                key_name.append(input_list[item][key_item])
                cand.append(cand_response_time_list[record_count])
                pinyin.append(pinyin_response_time_list[record_count])
                start.append(cand_key_up_list[record_count])
                cand_change.append(cand_change_list[record_count])
                pinyin_change.append(pinyin_change_list[record_count])

                if input_list[item][key_item] not in all_key_name:
                    all_key_name.append(input_list[item][key_item])
                    cand_sum.append(cand_response_time_list[record_count])
                    pinyin_sum.append(pinyin_response_time_list[record_count])
                    key_count.append(1)
                else:
                    key_index = all_key_name.index(input_list[item][key_item])
                    cand_sum[key_index] += cand_response_time_list[record_count]
                    pinyin_sum[key_index] += pinyin_response_time_list[record_count]
                    key_count[key_index] += 1
                record_count += 1

            key_info['key_name'] = key_name
            key_info['cand'] = cand
            key_info['pinyin'] = pinyin
            key_info['start'] = start
            key_info['cand_change'] = cand_change
            key_info['pinyin_change'] = pinyin_change
            word_info['word_list'] = input_list[item]
            word_info['word_result'] = key_info
            result_info.append(word_info)

        for key_item in range(len(all_key_name)):
            key_aver_result = dict()
            key_aver_result['key_name'] = all_key_name[key_item]
            key_aver_result['aver_cand'] = round(cand_sum[key_item] / key_count[key_item], 2)
            key_aver_result['aver_pinyin'] = round(pinyin_sum[key_item] / key_count[key_item], 2)
            aver_result_info.append(key_aver_result)

        return_info['input_name'] = input_name
        return_info['input_version'] = input_version
        return_info['aver_result_info'] = aver_result_info
        return_info['result_info'] = result_info

        json_data = json.dumps(return_info)
        jsonFile.writelines(json_data)
'''
def vaild_key_up_list_9(input_list):#key_list should be written in config.json
    is_vaild_key_up_list = [0]
    for item_list in input_list:
        for item in item_list:
            if str(item).isdigit() or str(item)=='&' :
                is_vaild_key_up_list.append(1)
            else:
                is_vaild_key_up_list.append(0)
        is_vaild_key_up_list.append(0)
    is_vaild_key_up_list.append(0)
    bool_is_vaild_key_up_list = [False] * len(is_vaild_key_up_list)
    for item in range(len(is_vaild_key_up_list)):
        if is_vaild_key_up_list[item]:
            bool_is_vaild_key_up_list[item] = True
    return bool_is_vaild_key_up_list


def vaild_key_up_list_26(input_list):#key_list should be written in config.json
    is_vaild_key_up_list = [0]
    exchange_list = {'abc': 2, 'def':3,'ghi':4,'jkl':5,'mno':6,'pqrs':7,'tuv':8,'wxyz':9,'&':'&'}

    for item_list in input_list:
        for item in item_list:
            for word in exchange_list:
                if item in word:
                    is_vaild_key_up_list.append(1)
                    #if str(exchange_list[word]).isdigit()  :
                    #   is_vaild_key_up_list.append(1)

                    #else:
                     #   is_vaild_key_up_list.append(0)
        is_vaild_key_up_list.append(0)
    is_vaild_key_up_list.append(0)
    bool_is_vaild_key_up_list = [False] * len(is_vaild_key_up_list)
    for item in range(len(is_vaild_key_up_list)):
        if is_vaild_key_up_list[item]:
            bool_is_vaild_key_up_list[item] = True
    return bool_is_vaild_key_up_list


def whether_vaild_key_up_list(input_list_file, keyboard_type):
    input_list = []
    with open(input_list_file) as input_file:
        for line in input_file.readlines():
            input_list.append(line.split()[0])
    if keyboard_type == 26:
        is_vaild_key_up_list = vaild_key_up_list_26(input_list)
    elif keyboard_type == 9:
        is_vaild_key_up_list = vaild_key_up_list_9(input_list)
    else:
        logger.error("wrong keyboard type, only 9 or 26 permit")
        return None
    return input_list, is_vaild_key_up_list


def get_change_list_by_siamese(np_list, keyup_list, is_vaild_key_up_list, model_name):

    model = load_model_with_contrastive_loss(model_name)

    result_key_up_list = []
    result_change_list = []

    keyup_list_len = len(keyup_list)
    keyup_list.append(len(np_list) - 1)
    for i in range(keyup_list_len - 1):

        if not is_vaild_key_up_list[i]:
           continue

        pred_sub = {}
        for j in range(keyup_list[i], keyup_list[i+1])[::-1]:
            pred = model.predict([np_list[j].reshape(1, 12, 100, 1), np_list[j - 1].reshape(1, 12, 100, 1)])
            pred_sub[j] = pred

        tag = max(pred_sub.items(), key=lambda x: x[1])[0]
        logger.debug("result_change_list : %s" % (pred_sub[tag]))

        result_key_up_list.append(keyup_list[i])
        result_change_list.append(tag)

    logger.debug("result_change_list : %s" % (result_change_list))
    logger.debug("result_key_up_list : %s" % (result_key_up_list))

    return result_key_up_list, result_change_list


def get_config(project_name):
    with open('config.json') as json_file:
        json_config = json.load(json_file)
        module_config = json_config[project_name]

    return module_config


def get_response_time(key_start_list, change_list, fps):

    response_time_list = list(map(lambda x: x[0] - x[1], zip(change_list, key_start_list)))

    for item in range(len(response_time_list)):
        response_time_list[item] = round(response_time_list[item] * 1000 / float(fps), 2)
    logger.debug('response_time_list : %s' % response_time_list)
    return response_time_list

def write_line(list_to_write, f, line_name):
    if len(list_to_write) < 1:
        logger.error('List : %s is Null' % line_name)
    else:
        f.writelines(line_name)
        for item in range(len(list_to_write)):
            f.writelines(',' + str(list_to_write[item]))
        f.writelines('\n')


def get_result(cand_model, pinyin_model, is_vaild_key_up_list, video_path, resultPath, input_name, input_version, input_list):
    config = get_config('sogou_input_response_time')
    img_size_width = config['img_size_width']
    img_size_hight = config['img_size_hight']
    input_dim = tuple(config['input_dim'])
    need_rgb_to_gray = config['need_rgb_to_gray']

    evaluation_video = input_evaluation_video(video_path, need_rgb_to_gray, img_size_width, img_size_hight)
    cand_data = evaluation_video.extract_cand('', True)
    cand_data /= 255
    pinyin_data = evaluation_video.extract_pinyin('', True)
    pinyin_data /= 255
    keyup_list = evaluation_video.get_keyup_list()
    cand_key_up_list, cand_change_list = get_change_list_by_siamese(cand_data, keyup_list, is_vaild_key_up_list,cand_model)
    pinyin_key_up_list, pinyin_change_list = get_change_list_by_siamese(pinyin_data, keyup_list, is_vaild_key_up_list,pinyin_model)

    pinyin_response_time_list = get_response_time(pinyin_key_up_list, pinyin_change_list, evaluation_video.video_fps)
    cand_response_time_list = get_response_time(cand_key_up_list, cand_change_list, evaluation_video.video_fps)

    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    #today = datetime.date.today()
    #result_name = today.strftime("%Y_%m_%d")
    videoname = video_path.split('/')[-1].split('.')[0]
    '''
    with open(resultPath + str(result_name)+'.csv','a') as f:
        f.writelines('video_name,'+str(videoname)+'\n')
        write_line(cand_response_time_list, f, 'cand')
        write_line(pinyin_response_time_list, f, 'pinyin')
        write_line(cand_key_up_list, f, 'start')
        write_line(cand_change_list, f, 'cand_change')
        #write_line(pinyin_key_up_list, f, 'start')
        write_line(pinyin_change_list, f, 'pinyin_change')
        f.writelines('\n')
    '''
    #with open(resultPath + str(result_name) + '_' + videoname + '.json', 'w') as jsonFile:
    with open(resultPath + videoname + '.json', 'w') as jsonFile:
        return_info = dict()
        result_info = []
        aver_result = []
        all_key_name = []
        record_count = 0
        cand_sum = []
        pinyin_sum = []
        key_count = []
        aver_result_info = []

        for item in range(len(input_list)):
            word_info = dict()
            key_info = dict()
            key_name = []
            cand = []
            pinyin = []
            start = []
            cand_change = []
            pinyin_change = []

            for key_item in range(len(input_list[item])):

                key_name.append(input_list[item][key_item])
                cand.append(cand_response_time_list[record_count])
                pinyin.append(pinyin_response_time_list[record_count])
                start.append(cand_key_up_list[record_count])
                cand_change.append(cand_change_list[record_count])
                pinyin_change.append(pinyin_change_list[record_count])

                if input_list[item][key_item] not in all_key_name:
                    all_key_name.append(input_list[item][key_item])
                    cand_sum.append(cand_response_time_list[record_count])
                    pinyin_sum.append(pinyin_response_time_list[record_count])
                    key_count.append(1)
                else:
                    key_index = all_key_name.index(input_list[item][key_item])
                    cand_sum[key_index] += cand_response_time_list[record_count]
                    pinyin_sum[key_index] += pinyin_response_time_list[record_count]
                    key_count[key_index] += 1
                record_count += 1

            key_info['key_name'] = key_name
            key_info['cand'] = cand
            key_info['pinyin'] = pinyin
            key_info['start'] = start
            key_info['cand_change'] = cand_change
            key_info['pinyin_change'] = pinyin_change
            word_info['word_list'] = input_list[item]
            word_info['word_result'] = key_info
            result_info.append(word_info)

        for key_item in range(len(all_key_name)):
            key_aver_result = dict()
            key_aver_result['key_name'] = all_key_name[key_item]
            key_aver_result['aver_cand'] = round(cand_sum[key_item] / key_count[key_item], 2)
            key_aver_result['aver_pinyin'] = round(pinyin_sum[key_item] / key_count[key_item], 2)
            aver_result_info.append(key_aver_result)

        return_info['input_name'] = input_name
        return_info['input_version'] = input_version
        return_info['aver_result_info'] = aver_result_info
        return_info['result_info'] = result_info

        json_data = json.dumps(return_info)
        jsonFile.writelines(json_data)
