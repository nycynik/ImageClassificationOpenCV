# Image Classification with Google Models and OpenCV

OpenCV image classification.

https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html

# env setup

Set up using anaconda and openCV

    conda env create -f environment.yml

-- or -- 

    conda update -n base -c defaults conda
    conda create --name deep python=3

    conda activate deep
    pip install cmake
    pip install numpy
    pip install opencv-contrib-python
    conda install -c conda-forge dlib

# to run

    python process.py

This will process the images in images, images must be 244, 244



