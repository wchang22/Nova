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
  clang-9 \
  cmake \
  opencl-headers \
  clinfo \
  ocl-icd-opencl-dev \
  intel-opencl-icd \
  nvidia-opencl-dev \
  xorg-dev \
  libarchive-tools \
  zip \
  numactl

RUN ln -sf /usr/bin/clang-9 /usr/bin/clang
RUN ln -sf /usr/bin/clang++-9 /usr/bin/clang++

WORKDIR /tmp

# Install intel cpu opencl runtime
ARG cpu_runtime_pkg=l_opencl_p_18.1.0.013
RUN wget "http://registrationcenter-download.intel.com/akdlm/irc_nas/13793/${cpu_runtime_pkg}.tgz"
RUN tar xf "${cpu_runtime_pkg}.tgz"
WORKDIR "${cpu_runtime_pkg}"
RUN rm rpm/intel-openclrt-pset-*.rpm && for i in rpm/*.rpm; do bsdtar -xf "$i"; done

RUN echo /opt/intel/opencl-runtime/linux/compiler/lib/intel64_lin/libintelocl.so > /etc/OpenCL/vendors/intel_cpu.icd
RUN mkdir -p /opt/intel/opencl-runtime
RUN cp -r opt/intel/opencl_*/* /opt/intel/opencl-runtime
RUN rm -rf /tmp/"${cpu_runtime_pkg}"*

ARG OCL_LIB=/opt/intel/opencl-runtime/linux/compiler/lib/intel64_lin/
ENV LD_LIBRARY_PATH $OCL_LIB:$LD_LIBRARY_PATH

# Install SPIRV-LLVM-Translator for compiling C++ for OpenCL to SPIR-V
WORKDIR /tmp
ARG translator_pkg=SPIRV-LLVM-Translator-v9.0.0-1-linux-Release
RUN wget "https://github.com/KhronosGroup/SPIRV-LLVM-Translator/releases/download/v9.0.0-1/${translator_pkg}.zip"
RUN unzip "${translator_pkg}.zip" -d "${translator_pkg}"
RUN install -Dm755 "${translator_pkg}"/lib/libLLVMSPIRVLib.so.9 /usr/lib
RUN install -Dm755 "${translator_pkg}"/bin/llvm-spirv /usr/bin
RUN rm -rf "${translator_pkg}"*

ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

WORKDIR /root
ENTRYPOINT bash
