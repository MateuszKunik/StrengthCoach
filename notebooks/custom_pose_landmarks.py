from mediapipe import solutions
mp_pose = solutions.pose


# Selected values of pose landmarks corresponding to MediaPipe library
values = [
    0, # NOSE
    11, # LEFT_SHOULDER
    12, # RIGHT_SHOULDER
    13, # LEFT_ELBOW
    14, # RIGHT_ELBOW
    15, # LEFT_WRIST
    16, # RIGHT_WRIST
    19, # LEFT_INDEX
    20, # RIGHT_INDEX
    23, # LEFT_HIP
    24, # RIGHT_HIP
    25, # LEFT_KNEE
    26, # RIGHT_KNEE
    27, # LEFT_ANKLE
    28, # RIGHT_ANKLE
    31, # LEFT_FOOT_INDEX
    32, # RIGHT_FOOT_INDEX
]


# Create a dictionary to store landmark names and new values
landmarks = dict()
for index, value in enumerate(values):
    # Extract pose landmark names
    name = mp_pose.PoseLandmark(value).name
    # Complete the dictionary with names and values
    landmarks[value] = {
        'landmark_name' : name,
        'new_value' : index
    }

# Add a completely new pose landmark
landmarks[max(landmarks) + 1] = {
    'landmark_name' : 'THORAX',
    'new_value' : len(landmarks)
}


# Create a custom connections set
connections = set()
for connection in mp_pose.POSE_CONNECTIONS:
    # Extract old values from POSE_CONNECTIONS
    v1, v2 = connection
    # Check if the values are expected
    if v1 in values and v2 in values:
        # Create connections for new values
        connections.add(
            (landmarks[v1]['new_value'], landmarks[v2]['new_value'])
        )

# Add a new connection
connections.add(
    (landmarks[0]['new_value'], landmarks[33]['new_value'])
        )