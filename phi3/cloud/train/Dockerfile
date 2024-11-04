FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu124-py310-torch241:biweekly.202410.2

USER root

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip
RUN apt-get install -y openssh-server openssh-client

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN MAX_JOBS=4 pip install flash-attn==2.6.3 --no-build-isolation
