# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.06-py3
FROM nvcr.io/nvidia/pytorch:22.06-py3
# Select the working directory
WORKDIR  /nesypi

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add cuda
RUN #pip3 install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# Add fonts for serif rendering in MPL plots
RUN apt-get update
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install --yes ttf-mscorefonts-installer
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN apt-get install dvipng cm-super fonts-cmu --yes
RUN apt-get install fonts-dejavu-core --yes
RUN pip install opencv-python==4.5.5.64
RUN git clone https://github.com/ml-research/NeSy-PI