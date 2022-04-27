from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash 
from django.contrib.auth.forms import UserCreationForm, UserChangeForm, PasswordChangeForm
from django.contrib import messages 
from .forms import SignUpForm, EditProfileForm 
import cv2 
import dlib
import time  
import imutils 
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from scipy.spatial import distance as dist

from imutils import face_utils
from playsound import playsound
from imutils.video import WebcamVideoStream

from matplotlib import style 
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib.animation as animation



def home(request): 
	return render(request, 'authenticate/home.html', {})

def login_user (request):
	if request.method == 'POST':  
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(request, username=username, password=password)
		if user is not None:
			login(request, user)
			messages.success(request,("You are logged in"))
			return redirect('home')  
		else:
			messages.success(request,('Error in logging in'))
			return redirect('login') 
	else:
		return render(request, 'authenticate/login.html', {})

def logout_user(request):
	logout(request)
	messages.success(request,('You are now logged out'))
	return redirect('home')

def register_user(request):
	if request.method =='POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data['username']
			password = form.cleaned_data['password1']
			user = authenticate(username=username, password=password)
			login(request,user)
			messages.success(request, ('You are now registered'))
			return redirect('home')
	else: 
		form = SignUpForm() 

	context = {'form': form}
	return render(request, 'authenticate/register.html', context)

def edit_profile(request):
	if request.method =='POST':
		form = EditProfileForm(request.POST, instance= request.user)
		if form.is_valid():
			form.save()
			messages.success(request, ('You have edited your profile'))
			return redirect('home')
	else: 		
		form = EditProfileForm(instance= request.user) 

	context = {'form': form}
	return render(request, 'authenticate/edit_profile.html', context)
	#return render(request, 'authenticate/edit_profile.html',{})



def change_password(request):
	if request.method =='POST':
		form = PasswordChangeForm(data=request.POST, user= request.user)
		if form.is_valid():
			form.save()
			update_session_auth_hash(request, form.user)
			messages.success(request, ('You have edited your password'))
			return redirect('home')
	else: 		 
		form = PasswordChangeForm(user= request.user) 

	context = {'form': form}
	return render(request, 'authenticate/change_password.html', context)

# Function to calculate EAR
def eye_aspect_ratio(eye):
	# Vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal eye landmarks 
	C = dist.euclidean(eye[0], eye[3])

	# The EAR Equation 
	EAR = (A + B) / (2.0 * C)
	return EAR

# Function to calculate MAR
def mouth_aspect_ratio(mouth): 
	A = dist.euclidean(mouth[13], mouth[19])
	B = dist.euclidean(mouth[14], mouth[18])
	C = dist.euclidean(mouth[15], mouth[17])

	MAR = (A + B + C) / 3.0
	return MAR

def model_connect(request):
    # Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
	EAR_THRESHOLD = 0.22
	# Another constant which will work as a threshold for MAR value
	MAR_THRESHOLD = 10
	# Declare another costant to hold the consecutive number of frames to consider for a blink 
	CONSECUTIVE_FRAMES = 15

	# Initialize two counters 
	FRAME_COUNT = 0 

	# Define the path of the shape predictor model
	predictor_path = "my_site/authenticate/models/face_landmarks_68.dat"

	# Intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
	print("[INFO]Loading the predictor.....")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)
	time.sleep(2)
	print("Predictor loaded successfully!\n")

	# Grab the indexes of the facial landamarks for the mouth, left and right eyes respectively 
	(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

	# Start the video stream and allow the camera to warm-up
	print("[INFO]Loading Camera.....")
	cam = WebcamVideoStream(src=0).start()
	time.sleep(2) 
	print("Camera loaded successfully!\n")
	
	# Loop over all the frames and detect the faces
	while True: 

		# Extract a frame 
		img = cam.read()

		# Display text "PRESS 'q' TO EXIT"
		cv2.putText(img, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 

		# Resize the frame 
		img = imutils.resize(img, width = 500)

		# Convert the frame to grayscale 
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Detect faces 
		faces = detector(img, 1)

		# Now loop over all the face detections and apply the predictor 
		for (i, face) in enumerate(faces): 

			# Apply the shape predictor model
			shape = predictor(gray, face)

			# Convert it to a (68, 2) size numpy array 
			shape = face_utils.shape_to_np(shape)

			# Draw a rectangle over the detected face 
			(x, y, w, h) = face_utils.rect_to_bb(face) 
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

			# Tag "Driver" for the detected face 
			cv2.putText(img, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			leftEye = shape[lstart:lend]
			rightEye = shape[rstart:rend] 
			mouth = shape[mstart:mend]

			# Compute the EAR for both the eyes 
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# Take the average of both the EAR
			EAR = (leftEAR + rightEAR) / 2.0

			# Calculate MAR for the mouth
			MAR = mouth_aspect_ratio(mouth)

			# Compute the convex hull for both the eyes and then visualize it
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)

			# Draw the contours 
			cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(img, [mouth], -1, (0, 255, 0), 1)

			# Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
			# Thus, count the number of frames for which the eye remains closed 
			if EAR < EAR_THRESHOLD: 

				FRAME_COUNT += 1

				cv2.drawContours(img, [leftEyeHull], -1, (0, 0, 255), 1)
				cv2.drawContours(img, [rightEyeHull], -1, (0, 0, 255), 1)

				if FRAME_COUNT >= CONSECUTIVE_FRAMES: 

					playsound('my_site/authenticate/sounds/alarm.mp3')
					cv2.putText(img, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			else: 

				if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
					playsound('my_site/authenticate/sounds/warning.mp3')

				FRAME_COUNT = 0

			# Check if the person is yawning
			if MAR > MAR_THRESHOLD:

				cv2.drawContours(img, [mouth], -1, (0, 0, 255), 1) 
				cv2.putText(img, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				playsound('my_site/authenticate/sounds/alarm.mp3')
				playsound('my_site/authenticate/sounds/warning_yawn.mp3')

		#display the frame 
		cv2.imshow("Output", img)
		key = cv2.waitKey(1) & 0xFF 	

		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	cam.stop()

	return render(request, 'authenticate/home.html')
