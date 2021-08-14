# pull tensorflow image with python3 
FROM tensorflow/tensorflow:2.1.0-py3

# set the working directory to /app
WORKDIR /app

# transfer content from current dir to /app in container
ADD . /app

# install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# install python packages
RUN pip install -r requirements.txt 

# start uWSGI using config file 
CMD ["uwsgi", "app.ini"]