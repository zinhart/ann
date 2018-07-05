#container with only cpu-threaded routines using MKL
FROM ubuntu:18.04
MAINTAINER zinhart

# RUN executes a shell command

RUN apt-get update && apt-get install -y --assume-yes apt-utils cpio curl wget git make cmake gcc g++ python3-pip

RUN cd /tmp && \
wget -q http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12414/l_mkl_2018.1.163.tgz && \
tar -xzf l_mkl_2018.1.163.tgz && \
cd l_mkl_2018.1.163 && \
sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
sed -i 's/ACTIVATION_TYPE=exist_lic/ACTIVATION_TYPE=trial_lic/g' silent.cfg && \
./install.sh -s silent.cfg && \
# Clean up
  cd .. && rm -rf *

# Configure dynamic link
RUN echo "${MKL_PATH}/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && ldconfig && \
  echo ". /opt/intel/bin/compilervars.sh intel64" >> /etc/bash.bashrc
RUN mkdir app
RUN cd app && git clone https://github.com/zinhart/ann.git

WORKDIR /app/ann
RUN git submodule update --remote --recursive
RUN ./debug-script
#CMD [python3, ann.py, ann_tests]
CMD [/debug/test/ann_tests]
