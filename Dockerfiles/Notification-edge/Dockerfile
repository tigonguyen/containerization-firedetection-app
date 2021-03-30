FROM nvcr.io/nvidia/l4t-base:r32.4.2

WORKDIR /usr/src/app

COPY  source_list/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  source_list/jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN apt-get update && apt-get install -y libopencv-python && apt-get install -y --no-install-recommends \
	python3-pip \
	python3-dev \
	build-essential \
	zlib1g-dev \
	zip \
	libjpeg8-dev && rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools Cython wheel
RUN pip3 install numpy --verbose

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "./firenet-notification.py" ]