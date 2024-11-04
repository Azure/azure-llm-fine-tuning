#PTCA image
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu124-py310-torch241:biweekly.202410.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN python -m nltk.downloader punkt
RUN MAX_JOBS=4 pip install flash-attn==2.6.3 --no-build-isolation