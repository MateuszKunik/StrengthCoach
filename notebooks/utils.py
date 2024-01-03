import numpy as np
import pandas as pd

from mediapipe.framework.formats import landmark_pb2


def landmark2array(landmark):
    """
    
    """
    return np.array(
        [
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ]
    )


def array2landmark(array):
    """
    
    """
    return landmark_pb2.NormalizedLandmark(
        x=array[0],
        y=array[1],
        z=array[2],
        visibility=array[3]
    )


def calculate_thorax_coordinates(mp_pose, landmarks):
    """
    Based on left and right shoulder coordinates function calculates the coordinates of thorax.
    """

    # Extract left and right shoulder landmarks and convert it to array
    left_shoulder = landmark2array(
        landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
    right_shoulder = landmark2array(
        landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    
    # Calculate the thorax coordinates and convert an array to landmark
    thorax = array2landmark(
        np.mean([left_shoulder, right_shoulder], axis=0)
    )

    return thorax


def get_custom_landmarks(mp_pose, custom_pose, landmarks):
    """
    
    """
    # Create customize landmarks list
    custom_landmarks = landmark_pb2.NormalizedLandmarkList()

    # Extend list by selected landmarks
    custom_landmarks.landmark.extend(
        [landmarks.landmark[index] for index in custom_pose.values])

    # Calculate the coordinates of thorax landmark
    thorax = calculate_thorax_coordinates(mp_pose, landmarks)

    # Add thorax landmark to custom list
    custom_landmarks.landmark.add().CopyFrom(thorax)

    return custom_landmarks


def prepare_dataframe(custom_pose):
    """
    
    """
    axes = ['X', 'Y', 'Z']
    names = custom_pose.get_landmarks().values()
    column_names = [('_'.join([name, axis])).lower() for name in names for axis in axes]

    return pd.DataFrame(columns = ['Id', 'timestamp'] + column_names)