from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

#from os import rename, listdir

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
import sys
import os
import sqlite3
sys.path.insert(0,'D:/lajos/coding/done_projects/builderschallenge/simple-file-upload/uploads/core/tensorflow/models/research/object_detection')

import film_search_django


def home(request):
    documents = Document.objects.all()
    return render(request, 'core/wordable.html', { 'documents': documents })

def model_form_upload(request):
    #global filename
    #global file_id
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        filename = request.FILES['document']
        #print(str(filename))
        path = 'D:/lajos/coding/done_projects/builderschallenge/simple-file-upload/media/documents/'
        if form.is_valid():
            form.save()
            documents = Document.objects.all()
            if len(documents) == 0:
                file_id = 1
            else : file_id = documents[len(documents)-1].id
            #print(documents.reverse()[0].id)
            #print(filename)
            print(path + str(filename))
            print(path + str(file_id) + '.mp4')
            video_title = request.POST['title']
            video_description = request.POST['description']
            film_search_django.wordable(path + str(filename), str(file_id), video_title)
            os.rename(path + str(filename), path + str(file_id) + '.mp4')
            #video_title = request.POST['title']
            #video_description = request.POST['description']
            #print(request.FILES)
            #print(str(video_title))
            #print(str(video_description))
            return redirect('home')
    else:
        form = DocumentForm()
        
    
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })


def details(request, pk):
    document = Document.objects.get(pk=pk)
    return render(request, 'core/details.html', { 'document': document })

def search(request):
    conn = sqlite3.connect('D:/lajos/coding/done_projects/builderschallenge/simple-file-upload/video.db')
    curs = conn.cursor()
    
    qs = Document.objects.all()
    vs = [[0 for j in range(2)] for i in range(0)]
    ts = list()
    cnt = -1
    
    q = request.GET.get('q', '') # GET request의 인자중에 q 값이 있으면 가져오고, 없으면 빈 문자열 넣기
    if q: # q가 있으면
        qs = qs.filter(title__icontains=q) # 제목에 q가 포함되어 있는 레코드만 필터링
        
    
    a = curs.execute("SELECT * FROM vd WHERE object == '%s'" % (q))
    
    print(q)
    
    rows = a.fetchall()

    row_counter = 0

            #출력
    for row in rows:
        l = row[0]
        m = row[1]
        n = row[2]
        #print('asdf')

        #print("비디오 번호 : %s  물체 번호 : %s  등장 시간 : %sm %.3lfs\n" % (l, m, int(n / 60), n % 60))
        row_counter += 1
        
        line = list()
        
        if cnt == -1 or vs[cnt][0] != l:
            line.append(l)
            line.append(n)
            cnt+=1
            vs.append(line)
        
    print(vs)


    return render(request, 'core/search.html' , {'search' : qs, 'q' : q, 'vs' : vs, 'ts' : ts, })

def search_wordable(request):
    conn = sqlite3.connect('D:/lajos/coding/done_projects/builderschallenge/simple-file-upload/video.db')
    curs = conn.cursor()
    
    qs = Document.objects.all()
    vs = [[0 for j in range(2)] for i in range(0)]
    ts = list()
    cnt = -1
    
    q = request.GET.get('q', '') # GET request의 인자중에 q 값이 있으면 가져오고, 없으면 빈 문자열 넣기
    if q: # q가 있으면
        qs = qs.filter(title__icontains=q) # 제목에 q가 포함되어 있는 레코드만 필터링
        
    
    a = curs.execute("SELECT * FROM vd WHERE object == '%s'" % (q))
    
    print(q)
    
    rows = a.fetchall()

    row_counter = 0

            #출력
    for row in rows:
        l = row[0]
        m = row[1]
        n = row[2]
        #print('asdf')

        #print("비디오 번호 : %s  물체 번호 : %s  등장 시간 : %sm %.3lfs\n" % (l, m, int(n / 60), n % 60))
        row_counter += 1
        
        line = list()
        
        if cnt == -1 or vs[cnt][0] != l:
            line.append(l)
            line.append(n)
            cnt+=1
            vs.append(line)
        
    print(vs)


    return render(request, 'core/wordable.html' , {'search' : qs, 'q' : q, 'vs' : vs, 'ts' : ts, })