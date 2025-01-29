# Reading analog meters with TinyML

## Description 
This repository contains all the files related to reading an analog meter with tiny machine learning.

There are the following files:
1) **functions.py**\
This contains all of the needed functions and packages related to the processing of the images, and making/training of the Tensorflow model.
Within each function there is docstring explaining the function, showing the variables, and its output.
For further information about specific points, comments are made explaining those things.

2) **model_and_processing.ipynb**\
This contains all of the code related to the processing of all of the images.
The markdown comments are there to explain the broader concept of the code.
It also contains the Tensorflow model.
With an explanation of how the models works, and how it is made.
The best model, based on Tensorflow itself, is then downloaded.

4) **smart_meter_model.keras**\
This contains our best pre-trained model, made in the **model_and_processing.ipynb** file.

5) **api_model.py**\
This contains the back-end of a local api.
Used to try and communicate with an Arduino, to prototype hardware for the reading of analog meters.

6) **image_viewer_api_call.ipynb**\
This contains the calling of the local api.
It isn't fully developed, due to the restraints of the Arduino.

## Requirements
In all of the previous files, there are multiple packages that are not pre-installed.
These are the following packages that need to be installed:
1) Numpy:
```sh
python -m pip install numpy
```

```sh
conda install -c conda-forge numpy
```

2) Cv2
3) Tensorflow
4) Requests
5) fast_api
6) pydantic
