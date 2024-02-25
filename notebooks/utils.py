import torch
import numpy as np
import pandas as pd

from mediapipe.framework.formats import landmark_pb2


def landmark2array(landmark):
    """
    
    """
    return np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])


def array2landmark(array):
    """
    
    """
    return landmark_pb2.NormalizedLandmark(array[0], array[1], array[2], array[3])


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



def get_custom_landmarks(mp_pose, custom_pose, landmarks):
    """
    
    """
    # Create customize landmarks list
    custom_landmark_list = landmark_pb2.NormalizedLandmarkList()

    # Extend list by selected landmarks
    custom_landmark_list.landmark.extend(
        [landmarks.landmark[index] for index in custom_pose.selected_values])

    for landmark_name in custom_pose.custom_landmarks.keys():
        # Calculate the coordinates of custom landmark
        custom = calculate_coordinates(mp_pose, landmarks, landmark_name)

        # Add thorax landmark to custom list
        custom_landmark_list.landmark.add().CopyFrom(custom)

    return custom_landmark_list


def sort_and_assign(data, batch_size, n_groups, ascending=True):
    """
    
    """

    data = data.sort_values(by='Frequency', ascending=ascending).reset_index(drop=True)

    data['GroupNumber'] = pd.cut(
        data.index + 1,
        bins = range(0, len(data) + batch_size, batch_size),
        labels = range(n_groups)
    )

    tmp = data.groupby(by='GroupNumber', as_index=False)['Frequency'].max()
    tmp = tmp.rename(columns={'Frequency': 'MaxFrequency'})

    data = pd.merge(data, tmp, on='GroupNumber')

    # Calculate how many frames should be added to each file on average
    mean = (data['MaxFrequency'] - data['Frequency']).mean()
    
    return data, mean


def assign_groups(data, batch_size):
    """
    
    """
    # Create a list of file IDs
    file_ids = data['FileId'].unique()
    # Calculate the number of groups
    n_groups = int(np.ceil(len(file_ids) / batch_size))

    #
    freq_data = data.groupby(by='FileId', as_index=False).size()
    freq_data = freq_data.rename(columns={'size': 'Frequency'})
    
    df_1, mean_1 = sort_and_assign(freq_data, batch_size, n_groups)
    df_2, mean_2 = sort_and_assign(freq_data, batch_size, n_groups, ascending=False)

    # Choose a better sorting option
    if mean_1 > mean_2:
        freq_data = df_2
        # mean = mean_2
    else:
        freq_data = df_1
        # mean = mean_1

    return pd.merge(data, freq_data, on='FileId')


def floor_ceil(x):
    """
    
    """
    return int(np.floor(x)), int(np.ceil(x))


def add_padding(data):
    """
    Add padding to the dataframe by max frequency group
    """
    # Reset index
    data = data.reset_index(drop=True)

    # Calculate how much padding should be added
    difference = data.loc[0, 'MaxFrequency'] - data.loc[0, 'Frequency']

    if difference > 1:
        # Calculate how many padding should be added to the beginning and to the end
        front, back = floor_ceil(difference / 2)

        # Get the first and last record
        first_record, last_record = data.iloc[0], data.iloc[-1]

        # Prepare data frames
        to_beginning = pd.concat(front * [pd.DataFrame([first_record])])
        to_end = pd.concat(back * [pd.DataFrame([last_record])])

        # Return concatenated data frames
        return pd.concat([to_beginning, data, to_end], ignore_index=True)

    elif difference == 1:
        # Get only the last record
        last_record = data.iloc[-1]

        # Return concatenated data frames
        return pd.concat([data, pd.DataFrame([last_record])], ignore_index=True)

    else:
        return data
    

def dataloader_function(data, batch_size):
    for _, group_data in data.groupby(by='GroupNumber'):
        # Drop the GroupNumber column
        group_data = group_data.drop(columns='GroupNumber')

        # Prepare group tensor storage
        group_tensors = torch.tensor([])

        for _, file_data in group_data.groupby(by='FileId'):
            # Drop the FileId column
            file_data = file_data.drop(columns='FileId')

            # Add padding to dataframe by MaxFrequency in the group
            adjusted = add_padding(file_data)
            # Pick columns to drop
            to_drop = ['Timestamp', 'Frequency', 'MaxFrequency']
            # Drop unnecessary columns and convert the dataframe to a numpy array
            array = adjusted.drop(columns=to_drop).to_numpy()

            # Convert numpy array to pytorch tensor
            tensor = torch.from_numpy(array).unsqueeze(dim=0)
            # Concatenate to other tensors in the group
            group_tensors = torch.cat((group_tensors, tensor), dim=0)
        
        print(group_tensors.shape)