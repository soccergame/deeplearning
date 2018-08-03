# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class MEDIA(models.Model):
    video = models.FileField(upload_to = 'file')
    name = models.CharField(max_length = 50)

class TESTDATA(models.Model):
    test_data = models.FileField(upload_to = 'testdata')
    name = models.CharField(max_length = 50)

class HISTORY_DATA(models.Model):
    history_input_name = models.TextField()
    history_input_version = models.TextField()
    history_json_path = models.CharField(max_length = 100)
    history_name = models.CharField(max_length= 50 )
    history_remark = models.TextField()