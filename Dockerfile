FROM ubuntu:18.04

RUN apt-get update -yy
RUN apt-get install -yy \
  software-properties-common \
  apt-transport-https \
  ca-certificates \
  gnupg \
  wget

# Add cmake
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

RUN add-apt-repository ppa:intel-opencl/intel-opencl

RUN apt-get update -yy
RUN apt-get install -yy \
  g++-8 \
  cmake \
  opencl-headers \
  clinfo \
  ocl-icd-opencl-dev \
  intel-opencl-icd \
  nvidia-opencl-dev

RUN ln -sf /usr/bin/g++-8 /usr/bin/g++

WORKDIR /root
ENTRYPOINT bash
