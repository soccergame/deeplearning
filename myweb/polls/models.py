# -*- coding: utf-8 -*-

from __future__ import unicode_literals
#from django import models
from django.db import models
import datetime
from django.utils import timezone

# Create your models here.
#@python_2_unicode_compatible # 当你想支持python2版本的时候才需要这个装饰器
class Question(models.Model):
    # ...
    def __unique__(self):   # 在python2版本中使用的是__unique__
        return self.question_text

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)
#@python_2_unicode_compatible
class Choice(models.Model):
    # ...
    def __unique__(self):
        return self.choice_text
# class Choice(models.Model):
#     question=models.ForeignKey(Question,on_delete=models.CASCADE)
#     choice_text = models.CharField(max_length=200)
#     votes = models.IntegerField(default=0)