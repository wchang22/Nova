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
  clang-9 \
  cmake \
  opencl-headers \
  clinfo \
  ocl-icd-opencl-dev \
  intel-opencl-icd \
  nvidia-opencl-dev \
  xorg-dev \
  alien \
  zip

RUN ln -sf /usr/bin/g++-8 /usr/bin/g++
RUN ln -sf /usr/bin/gcc-8 /usr/bin/gcc
RUN ln -sf /usr/bin/clang-9 /usr/bin/clang
RUN ln -sf /usr/bin/clang++-9 /usr/bin/clang++

WORKDIR /tmp

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

RUN wget https://github.com/KhronosGroup/SPIRV-LLVM-Translator/releases/download/v9.0.0-1/SPIRV-LLVM-Translator-v9.0.0-1-linux-Release.zip
RUN unzip SPIRV-LLVM-Translator-v9.0.0-1-linux-Release.zip -d spirv-llvm-translator
RUN install -Dm755 spirv-llvm-translator/lib/libLLVMSPIRVLib.so.9 /usr/lib
RUN install -Dm755 spirv-llvm-translator/bin/llvm-spirv /usr/bin

ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

WORKDIR /root
ENTRYPOINT bash
