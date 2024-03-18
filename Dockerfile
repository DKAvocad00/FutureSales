FROM ubuntu:20.04

RUN apt-get -y update
RUN apt-get -y install nginx

FROM python:3.10

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
