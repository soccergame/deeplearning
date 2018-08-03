# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import datetime
import json
from django.shortcuts import render
import  os
from evaluation.models import MEDIA,TESTDATA,HISTORY_DATA
import logging
from input_keyboard_response_time import get_result,whether_vaild_key_up_list
# Create your views here.


#class MEDIA(models.Model):
#    video = models.FileField(upload_to='file')
#    name = models.CharField(max_length=50)

def details(request):
    resultPath = './result/'
    with open('video_info.json', 'r') as jsonFile:
        video_info = json.load(jsonFile)
        video_name = video_info['video_name']
    result_name = resultPath+video_name+'.json'

    with open(result_name, 'r')as result_file:
        result_json = json.load(result_file)
    context = dict()
    result_info = result_json['result_info']
    context['result_info'] = result_info
    return render(request, 'details_info.html', context)


def average_info(request):
    context = dict()
    context['history_data'] = HISTORY_DATA.objects.get(pk=1)
    if request.method =="POST":
        resultPath = './result/'
        video_name = '2018_08_02_08_23_46_0155'
        result_name = resultPath + video_name + '.json'
        with open(result_name, 'r')as result_file:
            result_json = json.load(result_file)
        result_info = result_json['aver_result_info']
        context['result_info'] = result_info
        context['input_name'] = result_json["input_name"]
        context['input_version'] = result_json["input_version"]
        context['remark'] = request.POST.get("remark")
        if ( request.FILES.get('video_upload') and request.FILES.get('test_data_upload')):
            test_video = MEDIA(video=request.FILES.get('video_upload'),name= request.FILES.get('video_upload').name)
            test_video.video.name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + test_video.name
            input_test_data = TESTDATA(test_data= request.FILES.get('test_data_upload'),
                                       name = request.FILES.get('test_data_upload').name)
            input_test_data.test_data.name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'_'+input_test_data.name
            with open('video_info.json', 'w') as jsonFile:
                return_info = dict()
                return_info['video_name'] = test_video.video.name.split('.')[0]
                json_data = json.dumps(return_info)
                jsonFile.writelines(json_data)
            VIDEO_NAME = test_video.video.name
            TEST_LIST= input_test_data.test_data.name
            test_video.save()
            input_test_data.save()

            #result_info = result_json['aver_result_info']
            #context['result_info'] = result_info
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            input_name = request.POST.get('input_name')
            input_version = request.POST.get('input_version')
            keyboard_type = 9
            input_list_dir = './media/testdata/'
            input_list_file = os.path.join(input_list_dir,TEST_LIST)
            #input_list_file = './test31'
            resultPath = './result/'
            video_dir = './media/file/'
            video_path = os.path.join(video_dir,VIDEO_NAME)
            #video_path = 'media/file/0713.MOV'
            cand_model = './model/siamese_cnn_cand.h5'
            pinyin_model = './model/siamese_cnn_pinyin.h5'
            input_list, is_vaild_key_up_list = whether_vaild_key_up_list(input_list_file,keyboard_type)
            get_result(cand_model, pinyin_model, is_vaild_key_up_list, video_path, resultPath, input_name, input_version, input_list)
            history_data = HISTORY_DATA(history_input_name = request.POST.get('input_name') ,
                                        history_input_version = request.POST.get('input_version'),
                                        history_json_path = resultPath,
                                        history_name = VIDEO_NAME.split('.')[0]+'.json',
                                        history_remark = request.POST.get('remark'))
            history_data.save()
            video_name = VIDEO_NAME.split('.')[0]
            result_name = resultPath+video_name + '.json'
            with open(result_name, 'r')as result_file:
                result_json = json.load(result_file)
            result_info = result_json['aver_result_info']
            context['result_info'] = result_info
            context['input_name'] = request.POST.get('input_name')
            context['input_version'] = request.POST.get('input_version')
            context['remark'] = request.POST.get('remark')
    return render(request, 'average_info.html', context)
