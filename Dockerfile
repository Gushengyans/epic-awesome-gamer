FROM python:3.10 as builder

WORKDIR /home/epic

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update -y \
    && apt install -y wget

COPY src ./
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt install -y ./google-chrome-stable_current_amd64.deb \
    && rm ./google-chrome-stable_current_amd64.deb \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/yolov6t.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/rainbow.yaml \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/domestic_cat.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/lion.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/bridge.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/bedroom.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/living_room.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/conference_room.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/horse.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/dog.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/smiling_dog.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/horse_made_of_clouds.onnx \
    && wget -P model/ https://github.com/QIN2DIM/hcaptcha-challenger/releases/download/model/seaplane.onnx
