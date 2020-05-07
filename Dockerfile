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
  nvidia-opencl-dev \
  xorg-dev \
  alien

RUN ln -sf /usr/bin/g++-8 /usr/bin/g++
RUN ln -sf /usr/bin/gcc-8 /usr/bin/gcc

# Install intel cpu opencl runtime
RUN export RUNTIME_URL="http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz" \
  && export TAR=$(basename ${RUNTIME_URL}) \
  && export DIR=$(basename ${RUNTIME_URL} .tgz) \
  && wget -q ${RUNTIME_URL} \
  && tar -xf ${TAR} \
  && for i in ${DIR}/rpm/*.rpm; do alien --to-deb $i; done \
  && rm -rf ${DIR} ${TAR} \
  && dpkg -i *.deb \
  && rm *.deb

RUN echo "/opt/intel/opencl-1.2-6.4.0.25/lib64/libintelocl.so" > /etc/OpenCL/vendors/intel_cpu.icd

ENV OCL_INC /opt/intel/opencl/include
ENV OCL_LIB /opt/intel/opencl-1.2-6.4.0.25/lib64
ENV LD_LIBRARY_PATH $OCL_LIB:$LD_LIBRARY_PATH

WORKDIR /root
ENTRYPOINT bash
