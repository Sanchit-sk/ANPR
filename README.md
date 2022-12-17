<p align="center">
  <img src="https://github.com/Sanchit-sk/ANPR/blob/master/Images/ANPR%20Project%20banner.png" />
</p>

## Introduction 

Hi there! Let's get you acquainted with this project and its key objectives. <br/>
ANPR stands for Automatic number plate recognition system. This project aims to create an all-weather ANPR algorithm using
just image processing techniques and an OCR. Such a system can be utilized to automate the process of recording the vehicles number plates at the gates 
or checkpoints of an institute's campus, a secured locality, malls, etc.

## Installation and usage

The ANPR script has been written in python language and extensively uses the OpenCV library for various image processing methods.
Here is the list of dependencies for this project :-
- [opencv-python](https://pypi.org/project/opencv-python/) 
- [numpy](https://pypi.org/project/numpy/)
- [pytesseract](https://pypi.org/project/pytesseract/)

Run the ANPR script by typing the command below in the terminal
```
python ANPR script.py
```

## Basic Methodology

The main task of image processing in ANPR is to basically localise the license plate region in the frame, extract it out, enhance the image and then pass it
to the OCR to extract the text from the license plate image.
There are numerous ways to perform ANPR using image processing techniques. It can be done by either using morphological operations to highlight the text
in the number plate or it can be done by edge detection method. The current implmented algorithm uses the latter technique to filter the candidate
areas that may contain the license plate.

The algorithm can be thought of broadly divided into these stages :-
1. Pre-processing the input frame
2. Contour formation and filtering
3. Filtered regions extraction
4. Detecting the regions containing text
5. Image Cleaning
6. OCR to extract the number plate text

Below image depicts the various stages of algorithm along with their results.

<p align="center">
<img src = "https://github.com/Sanchit-sk/ANPR/blob/master/Images/ANPR%20results.png" height = "400" width = "700" />
</p>

## What's next?

Though the current approach is working fine with localising the number plate in the given frame, this is still just a start towards making this project
to work well in all the weather and lighting conditions. These are the objective that must be fulfilled next :-

- [x] Basic localisation and extraction of number plate 
- [ ] Number Plate Image enhancement to reduce the errors at OCR stage
- [ ] Using better image enhancement techniques for better efficiency
- [ ] Making this algorithm to work well in all sorts of lighting conditions
- [ ] Making a specialized OCR for the license plate text recognition
