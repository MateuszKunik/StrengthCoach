import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


mp_pose = solutions.pose

def landmark2array(landmark):
    return np.array(
        [
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ]
    )

def array2landmark(array):
    return landmark_pb2.NormalizedLandmark(
        x=array[0],
        y=array[1],
        z=array[2],
        visibility=array[3]
    )

def get_custom_landmarks(indexes, landmarks):
    """
    
    """
    # Create customize landmarks list
    custom_landmarks = landmark_pb2.NormalizedLandmarkList()

    # Extend list by chosen landmarks
    custom_landmarks.landmark.extend(
        [landmarks.landmark[index] for index in indexes])

    # Calculate the coordinates of neck landmark
    left_shoulder = landmark2array(
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
    right_shoulder = landmark2array(
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    neck = np.mean([left_shoulder, right_shoulder], axis=0)
    neck_landmark = array2landmark(neck)

    # Add neck landmark to custom list
    custom_landmarks.landmark.add().CopyFrom(neck_landmark)

    return custom_landmarks