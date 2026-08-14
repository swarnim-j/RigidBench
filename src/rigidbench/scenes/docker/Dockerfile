FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH="$PATH:/bin/4.5/python/bin/"
ENV BLENDER_PATH="/bin/4.5"
ENV BLENDERPY="/bin/4.5/python/bin/python3.11"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y \
    wget xz-utils libopenexr-dev bzip2 build-essential \
    zlib1g-dev libxmu-dev libxi-dev libxxf86vm-dev libfontconfig1 \
    libxrender1 libgl1-mesa-glx libxkbcommon0 libsm6 libice6 \
    libegl1 libegl-mesa0 libgles2 xvfb ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://download.blender.org/release/Blender4.5/blender-4.5.9-linux-x64.tar.xz \
    && tar -xf blender-4.5.9-linux-x64.tar.xz --strip-components=1 -C /bin \
    && rm -rf blender-4.5.9-linux-x64.tar.xz

RUN ${BLENDERPY} -m ensurepip \
    && ${BLENDERPY} -m pip install --upgrade pip \
    && ${BLENDERPY} -m pip install numpy shapely scipy pymunk pyyaml requests

WORKDIR /
CMD ["blender", "--version"]
