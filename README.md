# ArtemisBrailleGlasses
AI powered braille learning of the future 





## Overview

Part 1 of Pipeline: Artemis Net (Braille Translation)

--a pytorch based deep convolutional neural network (CNN) composed of residual skip connections (focal cross-entropy loss)
--Deployed on Fast API Endpoint, which allows the Pi to make a HTTP request to the model to get the translated braille


Part 2 of Pipeline: Voice Interactivity Engine

- a voice assistant using Groq's llama3.1:70b w/  fast inference in combination with elevenlabs API for TTS
- uses google SR for STT
- uses pyAudio for audio output


## Installation For the Voice Interactivity Engine
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install portaudio19-dev clang pulseaudio alsa-utils alsa-tools libasound2-dev flac libjpeg-dev fscamera ffmpeg
pip install -r requirements.txt
```
