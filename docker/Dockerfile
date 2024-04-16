FROM gcr.io/kaggle-gpu-images/python:latest

LABEL maintainer "hiroi"

ENV DEBIAN_FRONTEND=noninteractive3
# 一般ユーザーで実行
RUN useradd -m user

RUN pip install isort
# 他のモジュールをインストールする時はrequirements.txtに記載
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# 一般ユーザをCWDのユーザに変更
COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]