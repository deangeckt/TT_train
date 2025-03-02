# Table Tennis Shot Classification

## Main goal: 
The main goal of this project is to classify and distinguish between technically correct table tennis shot to one that is not.

## Application:
A real time mobile application used to improve player skill during practice.

## Architecture and Description
We processes the raw table-tennis footage by first splitting the video into individual shots, then using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) to extract a set of 3D body landmarks for every frame in each shot. These time‐ordered landmark sequences are finally fed into a supervised deep learning model we trained.

https://github.com/user-attachments/assets/c53e01e7-4326-4e95-95b9-7f24601b5dff


the raw images are in the following format:

https://github.com/user-attachments/assets/abc818af-1adf-4233-b707-79e6cd19313b


Thanks to Arik Shapira, the manager of TT [haifa team](https://tthaifa.co.il) for letting me film 🏓📸.

