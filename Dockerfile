# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ubuntu:20.04

# ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update --fix-missing && \
    apt-get install -y \
                    python3-dev \
                    python3-pip \
                    git \
                    build-essential \
                    libgl1-mesa-dev \
                    mesa-utils \
                    libglu1-mesa-dev \
                    fontconfig \
                    libfreetype6-dev
RUN pip3 install opencv-python-headless
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install shapely==1.7.1 pyyaml

RUN mkdir /workspace
COPY . /workspace

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
