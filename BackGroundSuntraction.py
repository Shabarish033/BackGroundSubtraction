# -*- coding: utf-8 -*-
import pygame
import cv2

def resize(bg,ref_img):
	width = ref_img.shape[1]
	height = ref_img.shape[0]
	dimensions = (width, height)
	resized = cv2.resize(bg, dimensions, interpolation = cv2.INTER_AREA)
	return resized

video = cv2.VideoCapture(0)
BackgroundVideo = cv2.VideoCapture("Beach.mp4")
ret0, ref_img = video.read()

#Initialize Music
pygame.mixer.init()
#Load music
pygame.mixer.music.load("Sea.mp3")
#Start Playing the Song
pygame.mixer.music.play()

while(True):
        ret1, Curr_img = video.read()      
        
        background_object = cv2.createBackgroundSubtractorMOG2(history = 50, 
                                                               varThreshold = 150, 
                                                               detectShadows = False)
        foreground_mask1 = background_object.apply(ref_img)
        foreground_mask2 = background_object.apply(Curr_img)
        
        ret2, bg = BackgroundVideo.read()
        ret2, bg2 = BackgroundVideo.read()
        
        bg = resize(bg,ref_img)
        bg2 = resize(bg2,ref_img)
        
        MatrixOfNonZeroElements = cv2.findNonZero(foreground_mask2)
        
        if MatrixOfNonZeroElements is None:
            continue
        else:
            NonZeroCoOrdinates = MatrixOfNonZeroElements[:,0,:]
            for i in range(len(NonZeroCoOrdinates)):
                bg[NonZeroCoOrdinates[i, 1], NonZeroCoOrdinates[i, 0]] = 0
                bg[NonZeroCoOrdinates[i, 1], NonZeroCoOrdinates[i, 0]] = Curr_img[NonZeroCoOrdinates[i, 1], NonZeroCoOrdinates[i, 0]]
                
        cv2.imshow('Background Video', bg2)
        cv2.imshow('Background Removal',bg) 
        key = cv2.waitKey(5) & 0xFF
        if ord('q') == key:
            #Stop Music
            pygame.mixer.music.stop()
            break
        
cv2.destroyAllWindows()
video.release()