import numpy as np

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


def calculate_coordinates(mp_pose, landmarks, target):
    """
    Based on left and right shoulder/hip coordinates function calculates the coordinates of thorax/pelvis.
    """

    # Check which custom landmark as a target has been chosen
    if target.lower() == 'thorax':
        values = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]

    elif target.lower() == 'pelvis':
        values = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP
        ]

    # Extract left and right landmarks and convert it to array
    left = landmark2array(
        landmarks.landmark[values[0]]
    )
    right = landmark2array(
        landmarks.landmark[values[1]]
    )
    
    # Calculate coordinates and convert an array to landmark_pb2 type
    target = array2landmark(
        np.mean([left, right], axis=0)
    )

    return target   



def get_custom_landmarks(mp_pose, custom_pose, landmarks, custom_landmarks=None):
    """
    
    """
    # Create customize landmarks list
    custom_landmark_list = landmark_pb2.NormalizedLandmarkList()

    # Extend list by selected landmarks
    custom_landmark_list.landmark.extend(
        [landmarks.landmark[index] for index in custom_pose.selected_values])

    if custom_landmarks:
        for landmark_name in custom_landmarks:
            # Calculate the coordinates of custom landmark
            custom = calculate_coordinates(mp_pose, landmarks, landmark_name)

            # Add thorax landmark to custom list
            custom_landmark_list.landmark.add().CopyFrom(custom)

    return custom_landmark_list