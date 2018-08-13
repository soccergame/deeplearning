# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import HttpResponse

from django.shortcuts import render
from cmdb import models
# Create your views here.
user_list=[{"user":"jack","pwd":"abc"},{"user":"tom","pwd":"ABC"}]
def index(request):
    if request.method=="POST":
        username=request.POST.get("username", None)
        passwprd=request.POST.get("password",None)
        #添加数据到数据库
        models.UserInfo.objects.create(user=username,pwd=passwprd)
        # tmp={"user":username,"pwd":passwprd}
        # user_list.append(tmp)
        #print(username,passwprd)
    #return HttpResponse("hello world!")
    #从数据库中读取所有数据
    user_list=models.UserInfo.objects.all()
    return render(request, "index.html",{"data":user_list})