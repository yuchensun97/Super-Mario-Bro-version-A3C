"""
This script define the training environment
"""
import cv2
import numpy as np
from collections import deque

from params import *

def preprocess(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)   # convert to grey scale to improve trainning speed
        frame = cv2.resize(frame,(128,128))[None,:,:]/255.
    else:
        return np.zeros((1,128,128))

def video(video_frame,state,new_episode_bool):
    """
    Args:
        video_frame -- a deque object used to store frames
        state -- agent's observation
        new_episode_bool -- a boolean used to determine a new episode
    """
    frame = preprocess(state)

    if new_episode_bool:
        # clear stacked frames
        stacked_frame = deque([np.zeros((128,128),dtype=np.int) for i in range(stack_size)],maxlen=4)
        stacked_frame.append(frame)
        stacked_frame.append(frame)
        stacked_frame.append(frame)
        stacked_frame.append(frame)

        stacked_state = np.stack(stacked_frame,axis=2)
    else:
        # append frame to deque
        stacked_frame.append(frame)
        stacked_state = np.stack(stacked_frame,axis=2)

    return stacked_state,stacked_frame