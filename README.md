# IAPR_Project

## TO-DO List

- [X] Extract signs, digits (Data Augmentation) : rotation, size, brighthness variant ... etc
- [X] Train a classifier on them (CNN)
	* [ ] Improve accuracy of the model.
- [X] Method to track robot
- [X] Plot a line for robot trajectory
- [ ] At each robot stop (detection), display the number on screen
- [X] Plot a box around the detected objects and label them
	* [X]  Function to draw rectangle around the digits, operators.
		
	* [ ]  Function to see if robot passed by the object (True => label it).
		
	* [ ]  Function to draw equation on the frame (if object is detected).
		
- [ ] Calculate and display output  (On terminal and the last frame)




# Usage:
```
python  main.py --input /path/to/input/video.avi --output /path/to/your/result.avi
```

***Video :***  ".avi" format, recorded at 2 FPS

***Requirement:*** 
- The current state of the formula at time t
- The trajectory of the robot from start to time t
               
***Deadline:*** May 28th, 11:59 PM