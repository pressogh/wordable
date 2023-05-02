# Wordable

## 서비스 소개
### 1️. 서비스 목적
사용자가 동영상을 업로드하면 **동영상을 분석하여 자동으로 태그를 달아 검색에 사용하는 동영상 플랫폼**
### 2. 주요 기능
- 동영상을 업로드하면 오브젝트 디텍션을 이용하여 동영상에 등장한 물체를 찾은 뒤 시간 등의 정보와 함께 데이터베이스에 저장됩니다.
- 위에서 데이터베이스에 저장된 정보를 이용하여 검색 및 세부 검색에 사용합니다.
### 3. ETC
PyQt를 이용하여 GUI로 어플리케이션을 제작하다가 개발 중에 웹 사이트에 접목시키면 좋을 것 같다고 판단하여 웹 사이트로 방향을 바꾸었습니다.

## 사용 기술
### Frontend
- Django Template
- Bootstrap
- ~~PyQt5~~
### Backend
- Python(version 3.6)
- Django

## 실행 방법
#### 1. Clone Repository
```bash
git clone https://github.com/pressogh/wordable.git
```
#### 2. Install Packages
```bash
pip install -r requirements.txt
```
#### 3. Start Server
```bash
python manage.py runserver
```

## 포스터
<img src="https://user-images.githubusercontent.com/50871137/235599441-bb3bfece-ea61-43b4-b04d-c93c6140881b.png" width="40%" height="30%" title="포스터 1" alt="포스터 1"></img>
<img src="https://user-images.githubusercontent.com/50871137/235599459-8b0ac39b-4602-41fb-9b2f-7025720cf765.png" width="40%" height="30%" title="포스터 2" alt="포스터 2"></img>

## 성과
[**제 1회 한국코드페어에서 과학기술정보통신부장관상**](https://incheonedu-my.sharepoint.com/personal/user1205_o365_ice_go_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fuser1205%5Fo365%5Fice%5Fgo%5Fkr%2FDocuments%2FSW%EB%A7%88%EC%97%90%EC%8A%A4%ED%8A%B8%EB%A1%9C%2F%ED%95%9C%EA%B5%AD%EC%BD%94%EB%93%9C%ED%8E%98%EC%96%B4%20%EB%B9%8C%EB%8D%94%EC%8A%A4%20%EC%B1%8C%EB%A6%B0%EC%A7%80%20%EA%B8%88%EC%83%81%20%EC%9D%B4%EA%B0%95%ED%98%81%2Epng&parent=%2Fpersonal%2Fuser1205%5Fo365%5Fice%5Fgo%5Fkr%2FDocuments%2FSW%EB%A7%88%EC%97%90%EC%8A%A4%ED%8A%B8%EB%A1%9C&ga=1, "상장")을 수상했습니다.
