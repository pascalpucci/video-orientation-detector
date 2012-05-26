#!/usr/bin/env python

# This tool "videoOrientationDetector.py" aims to detect videos Orientation.
# This tool use "face detection algorithm" possibly present in your video. Best
# detection require also some faces of persons in your video.  
# Enjoy ! (if it work ! :-D, for my videos, it work fine)
#
# Example of use for convertion to MP4 video :
#
#     orientation=$(./videoOrientationDetector.py video.mov)
#     ffmpeg -y -i video.mov -threads 4 -acodec libfaac -vcodec libx264 -b:v 600k $orientation videoWithGoodOrientation.tmp
#     qt-faststart videoWithGoodOrientation.tmp videoWithGoodOrientation.mp4
#
# To remove output : ./videoOrientationDetector.py 2> /dev/null or ./videoOrientationDetector.py -Q
#
# Copyright (C) 2012  Pascal Pucci <pascal.pucci@pascalou.org>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import cv2.cv as cv
from optparse import OptionParser
import time
import threading
import numpy as np

# haar files
path_haarfile = '/usr/share/opencv/haarcascades/'
haar_face = '%s/haarcascade_frontalface_default.xml' % path_haarfile
haar_eye = '%s/haarcascade_eye.xml' % path_haarfile
haar_mouth = '%s/haarcascade_mcs_mouth.xml' % path_haarfile
haar_eyepair = '%s/haarcascade_mcs_eyepair_small.xml' % path_haarfile

# haar args    
min_size = (20,20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

def searchForm(haar,formregion,winform,color=0,firstonly=0,view=0):
    
    if not color:
	color = cv.RGB(255, 0, 0)
    formCascade = cv.Load(haar)
    forms = cv.HaarDetectObjects(formregion, formCascade, cv.CreateMemStorage(0), haar_scale, min_neighbors, haar_flags, (25,15))

    point = []
    if forms:
	for form in forms:
	    x1 = form[0][0]
	    y1 = form[0][1]
	    x2 = form[0][0] + form[0][2]
	    y2 = form[0][1] + form[0][3]
	    midpoint = (int((x2+x1)/2) , int((y2+y1)/2))
	    point.append(midpoint)

	    if view:
		cv.Rectangle(formregion, (x1, y1), (x2, y2), color, 1, 8, 0)
		cv.ShowImage(winform,formregion)
	    
	    if firstonly:
		break
    return point

def detect_and_draw(img, win, verbose = 1, view = 1):

    if view:
	cv.NamedWindow(win, 1)

    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
			       cv.Round (img.height / image_scale)), 8, 1)
    score=0
    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)
    
#    cascadeB = cv.Load('/home/nathalie/tmp/OpenCV-2.4.0/data/haarcascades/haarcascade_mcs_upperbody.xml')
#    bodys = cv.HaarDetectObjects(small_img, cascadeB, cv.CreateMemStorage(0),
#				 haar_scale, min_neighbors, haar_flags, min_size)
#
#    if bodys:
#	for ((x, y, w, h), n) in bodys:
#	    score += 1
#	    pt1 = (int(x * image_scale), int(y * image_scale))
#	    pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
#	    cv.Rectangle(img, pt1, pt2, cv.RGB(0, 0, 0), 1, 8, 0)
#	    x1=int(x * image_scale)
#	    y1=int(y * image_scale)
#	    x2=int((x + w) * image_scale)
#	    y2=int((y + h) * image_scale)
#	
#	    winbody = "%s_body" % win
#	    cv.NamedWindow(winbody, 2)
#	    bodyregion=cv.GetSubRect(img, (x1,y1,x2-x1,y2-y1))
#	    cv.ShowImage(winbody,bodyregion)

    cascade = cv.Load(haar_face)
    faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
				 haar_scale, min_neighbors, haar_flags, min_size)
    if faces:
	for ((x, y, w, h), n) in faces:
	    score += 2
	    winface = "%s_face" % win
	    if view:
		cv.NamedWindow(winface, 2)
		pt1 = (int(x * image_scale), int(y * image_scale))
		pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
		cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 1, 8, 0)

	    x1=int(x * image_scale)
	    y1=int(y * image_scale)
	    x2=int((x + w) * image_scale)
	    y2=int((y + h) * image_scale)
	    faceregion=cv.GetSubRect(img, (x1,y1,x2-x1,y2-y1))

	    #sys.stderr.write("x1=%i y1=%i x2=%i y2=%i" % (x1, y1, x2, y2))
	    if view:
		cv.ShowImage(winface,faceregion)
	    if verbose:
		sys.stderr.write("Face detected on %s !\n" % win)

	    #eyes
	    eyes = searchForm(haar_eye,faceregion,winface,view)
	    mouth = searchForm(haar_mouth,faceregion,winface,cv.RGB(0, 255, 0),1,view)
	    eyepair = searchForm(haar_eyepair,faceregion,winface,cv.RGB(0, 0, 255),1,view)

	    leyes = len(eyes)
	    lmouth = len(mouth)
	    leyepair = len(eyepair)

	    if leyes or lmouth or leyepair:
		if leyepair:
		    score+=16
		    if verbose:
			sys.stderr.write("Tow eyes detected on %s !\n" % win)

		#sys.stderr.write("> %s eyes=%i mouth=%i eyepair=%i\n" % (win,leyes,lmouth,leyepair))
		if leyes and lmouth:
		    V = eyes[0][1] - mouth[0][1]
		    if V<0:
			score+=4
			if verbose:
			    sys.stderr.write("Eyes on top of the mouth detected on %s !\n" % win)
			if leyes >=2:
			    score+=8
			    #sys.stderr.write(eyes)
			    h1 = eyes[0][0]
			    h2 = mouth[0][0]
			    h3 = eyes[1][0]
			    if (h3>h1 and h2<h3 and h2>h1) or (h3<h1 and h2>h3 and h2<h1):
				if verbose:
				    sys.stderr.write("Mouth beetween eyes detected on %s !\n" % (win))
				score+=16
				v1 = int((eyes[1][0]+eyes[1][1])/2)
				v2 = int(mouth[0][1])
				if (v2 - v1) > 0 :
				    if verbose:
					sys.stderr.write("Mouth beetween eyes detected under eyes on %s !\n" % win)
				    score+=32
    if view:	
	cv.ShowImage(win, img)
    
    return score

class Detect(threading.Thread):
    def __init__(self, image,rot,win,verbose,view):
        threading.Thread.__init__(self)
        self.win = win
        self.image = image
        self.rot = rot
        self.Terminated = False
	self.score=0
	self.verbose=verbose
	self.view=view
	self.timg=[]

    def run(self):
	global Gscore
	if not self.Terminated:
	    if self.rot == 270:
		self.timg = cv.CreateImage((self.image.height,self.image.width), self.image.depth, self.image.channels)
		cv.Transpose(self.image,self.timg)
		cv.Flip(self.timg,self.timg,flipMode=0)
		Gscore[2] += detect_and_draw(self.timg, self.win,self.verbose,self.view)
	    elif self.rot == 90:
		self.timg = cv.CreateImage((self.image.height,self.image.width), self.image.depth, self.image.channels)
		cv.Transpose(self.image,self.timg)
		cv.Flip(self.timg,self.timg,flipMode=1)
		Gscore[1] += detect_and_draw(self.timg, self.win,self.verbose,self.view)
	    else:
		Gscore[0] += detect_and_draw(self.image, self.win,self.verbose,self.view)

	    #sys.stderr.write("> %s %i\n" % (self.rot, self.score))

    def stop(self):
        self.Terminated = True

class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'

    def disable(self):
	self.GREEN = ''
	self.YELLOW = ''
	self.ENDC = ''


if __name__ == '__main__':

    start = time.clock()

    parser = OptionParser(usage = "usage: %prog [options] [video filename]")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="make lots of noise [default: no]") 
    parser.add_option("-q", "--quiet", action="store_false", dest="verbose", default=False, help="be quiet [default; yes]") 
    parser.add_option("-Q", "--veryQuiet", action="store_true", dest="veryquiet", default=False, help="be very quiet [default; no]") 
    parser.add_option("-s", "--show", action="store_true", dest="view", default=False, help="show X display detection [default: no]") 
    parser.add_option("-t", "--try", type="int", dest="mtry", default=60, help="Maximum detection duration [default: 60s]") 
    parser.add_option("-m", "--maxFrame", type="int", dest="mframe", default=200, help="Maximum frames to analyse [default: 200]") 
    parser.add_option("-c", "--score", type="int", dest="score", default=80, help="Score to declare winner [default: 80]") 
    parser.add_option("-a", "--typeArgumentOutput", dest="argoutput", default='ffmpeg', help="Print argument output for 'mencoder' or 'ffmpeg' [default: ffmpeg]") 
    parser.add_option("-l", "--lowspeed", action="store_true", dest="lowspeed", default=False, help="Low speed for DEMO only [default: no]") 
    (options, args) = parser.parse_args()

    VERBOSE = options.verbose
    VIEW = options.view
    MAXTRY = options.mtry
    MAXFRAME = options.mframe
    SCORE = options.score
    ARGOUTPUT = options.argoutput
    VERYQUIET = options.veryquiet
    LOWSPEED = options.lowspeed
    if LOWSPEED:
	VIEW = True
    
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    input_name = args[0]

    # Check File Presence 
    if not os.path.isfile(input_name):
	sys.stderr.write("\n%s is not present. Please give file as argument.\n\n" % input_name)
	parser.print_help()
	sys.exit(1)
    
    if not os.path.isfile(haar_face):
	sys.stderr.write("\nHaar file %s is not present. Please give file as argument.\n\n" % haar_face)
	parser.print_help()
	sys.exit(1)

    if not os.path.isfile(haar_eye):
	sys.stderr.write("\nHaar file %s is not present. Please give file as argument.\n\n" % haar_eye)
	parser.print_help()
	sys.exit(1)

    if not os.path.isfile(haar_eyepair):
	sys.stderr.write("\nHaar file %s is not present. Please give file as argument.\n\n" % haar_eyepair)
	parser.print_help()
	sys.exit(1)

    if not os.path.isfile(haar_mouth):
	sys.stderr.write("\nHaar file %s is not present. Please give file as argument.\n\n" % haar_mouth)
	parser.print_help()
	sys.exit(1)

    # take movie
    capture = cv.CaptureFromFile(input_name)
    global Gscore
    Gscore = np.array([ 0, 0, 0 ])

    # One Thread per rotation
    image = cv.QueryFrame(capture)
    image90 = cv.CreateImage((image.width,image.height),image.depth,image.channels);
    image270 = cv.CreateImage((image.width,image.height),image.depth,image.channels);
    cv.Copy(image,image90);
    cv.Copy(image,image270);

    emoy = 0
    frame = 0
    elapsed = 0
    successfullpourcent = 0

    # let's GO
    while image and emoy < SCORE and elapsed < MAXTRY and frame <= MAXFRAME:

	Original = Detect(image,0,"Original",VERBOSE,VIEW)
	Rotation_90 = Detect(image90,90,"Rotation_90",VERBOSE,VIEW)
	Rotation_270 = Detect(image270,270,"Rotation_270",VERBOSE,VIEW)

	Original.start()
	Rotation_90.start()
	Rotation_270.start()
	maxv = Gscore.max()
	mean = int(Gscore.mean())
	emoy = maxv - mean
	
	if VERBOSE > 1:
	    sys.stderr.write("> %i: %i %i %i moy=%i max=%i emoy=%i <\n" % (frame,Gscore[0],Gscore[1],Gscore[2], mean, maxv, emoy))

	successfullpourcent = int (emoy * 100 / SCORE)	
	resttime = int (MAXTRY - elapsed)	

	if Gscore[2] > Gscore[1] and Gscore[2] > Gscore[0]:
	    winner = "Rotation -90"
	elif Gscore[1] > Gscore[0] and Gscore[1] > Gscore[2]:
	    winner = "Rotation 90"
	else:
	    winner = "Original"

	lowtxt=''
	if LOWSPEED:
	    lowtxt = " - LOWSPEED ENABLED"

	if not VERYQUIET:
	    sys.stderr.write("File: %s - %sSuccess: %i/100%s - Rest Time: %is - %sDetection: %s%s %s\n" % (input_name,bcolors.YELLOW,successfullpourcent,bcolors.ENDC,resttime,bcolors.GREEN,winner,bcolors.ENDC,lowtxt))

	try:	
	    image = cv.QueryFrame(capture)
	    frame += 1
	except: 
	    image = False

	if image:
	    cv.Copy(image,image90)
	    cv.Copy(image,image270)

	elapsed = (time.clock() - start)

	if LOWSPEED:
	    time.sleep(1.5)

    	if cv.WaitKey(10) == 27:
	    break
    
    if VIEW:
	cv.DestroyAllWindows()
	if LOWSPEED:
	    time.sleep(3)

    output=''
    if winner=="Rotation 90":
	if ARGOUTPUT=="mencoder":
	    output = "-vf rotate=1\n"
	else:
	    output = "-vf transpose=1\n"
    elif winner=="Rotation -90": 
	if ARGOUTPUT=="mencoder":
	    output = "-vf rotate=3\n"
	else:
	    output = "-vf transpose=2\n"

    sys.stdout.write(output)

# videoOrientationDetector.py  Copyright (C) 2012  Pascal Pucci <pascal.pucci@pascalou.org>
