FROM nginx:latest
FROM nvcr.io/nvidia/pytorch:22.12-py3

# アプリケーションを配置するディレクトリを確保する
RUN mkdir /app
COPY . /app

### nginxの設定 ###
# contentsを配置するディレクトリを作成する
COPY nginx/work /home/work
RUN mkdir -p /var/log/nginx/log \
    && mkdir /home/www \
    && mkdir /home/www/contents

# ビルド環境で構築されたdistディレクトリをnignxの該当のディレクトリに配置する
COPY dist/ /home/www/contents

# nginx.confファイルを配置する
WORKDIR /home/work
RUN rm -f /etc/nginx/conf.d/*.conf \
    && rm -f /etc/nginx/nginx.conf \
    && cp -i *.conf /etc/nginx
###############

### uwsgiの設定 ###
# データベースの接続情報を環境変数に設定
# ENV DATABASE_USER=qbtrxpqgcpedpc\
#     DATABASE_HOST=ec2-54-159-138-67.compute-1.amazonaws.com\
#     DATABASE_PASSWORD=d6c41d81d1c2b79954f034bc1c9b70590fab583b1ed4360d311365da54ab7146\
#     DATABASE_NAME=ddahgpnmmp4u5s

# python本体とパッケージのインストール
WORKDIR /app
RUN apt-get update &&\
    apt-get install -y python3 python3-pip libpq-dev &&\
    pip3 install uwsgi &&\
    pip3 install -r requirements.txt
###############

# サービスの軌道を管理するプログラムをインストール
RUN apt-get install -y supervisor

# nginx.confのlistenポートをherokuによって設定される環境変数PORTに変更し、サービスを起動。
CMD sed -i -e "s/ENV_PORT/$PORT/g" /etc/nginx/nginx.conf &&\
    supervisord -c supervisord.conf
