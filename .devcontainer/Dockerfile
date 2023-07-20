# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/codespaces-linux/.devcontainer/base.Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/universal:2-focal

RUN apt-get update && apt-get -y install --no-install-recommends \
   ffmpeg \ 
   python3.11-venv \
   gcc \
   pciutils

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update \
&& apt-get install -y nvidia-docker2 \
&& systemctl restart docker


# Install Pipenv
RUN pip install --no-cache-dir pipenv
#create global virtual environment using python standard library tools of virtualenv
ARG USER="codespace"
ARG VENV_PATH="/home/${USER}/venv"
#COPY requirements.txt /tmp/
COPY Pipfile Pipfile.lock /tmp/
COPY Makefile /tmp/
RUN su $USER -c "/usr/bin/python3 -m venv /home/${USER}/venv" \
   && su $USER -c "${VENV_PATH}/bin/pipenv install" \
   && rm -rf /tmp/Pipfile Pipfile.lock

