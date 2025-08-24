
# Hand Gesture Media Control

A Python project that uses **OpenCV** and **MediaPipe** to control system media with hand gestures.  

- Control **volume** using the distance between both hands’ index fingers.  
- Control **brightness** using the distance between the thumb and middle finger of the left hand.  

---

## Features
- Real-time hand tracking with MediaPipe  
- Gesture-based **volume control**  
- Gesture-based **brightness control**  

---

## Installation
Clone the repository:
```bash
git clone https://github.com/planetraman/Hand-Gesture-MediaControl.git
cd Hand-Gesture-MediaControl
```
## Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the script:
```bash
python main.py
```
- Move both index fingers apart/closer → adjust volume

- Move left thumb & middle finger apart/closer → adjust brightness

## Requirements

- Python 3.10+

- OpenCV

- MediaPipe

- pycaw (for volume control on Windows)

- screen-brightness-control

## Author  
Developed by [Raman Singh](https://github.com/planetraman)  

