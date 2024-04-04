import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from mediapipe.framework.formats import landmark_pb2
from sklearn.model_selection import train_test_split


def landmark2array(landmark):
    """
    
    """
    return np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])


def array2landmark(array):
    """
    
    """
    return landmark_pb2.NormalizedLandmark(x=array[0], y=array[1], z=array[2], visibility=array[3])


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


def split_data(data, proportions):
    """
    
    """

    proportions = np.array(proportions)

    if proportions.sum() == 1:
        # Get unique file IDs
        file_ids = data['FileId'].unique()

        if len(proportions) < 3 or proportions[2] == 0:
            train, valid = train_test_split(file_ids, test_size=proportions[1])

            # Put ids into dictionary
            return {"train": train, "validation": valid}
        
        else:
            # Calculate valid and test joint part and only test part
            joint_part = proportions[1:].sum()
            only_test = proportions[2] / joint_part

            # Split the files into three lists
            train, ids_to_split = train_test_split(file_ids, test_size=joint_part)
            valid, test = train_test_split(ids_to_split, test_size=only_test)

            # Put ids into dictionary
            return {"train": train, "validation": valid, "test": test}

    else:
        print(f'Please check proportions: {proportions}, they should add up to 1.')


lowess = sm.nonparametric.lowess

def lowess_function(column, frac, it):
    """
    
    """
    return lowess(
        exog = column.index,
        endog = column.values,
        frac = frac,
        it = it,
        return_sorted = False)

def smooth_data(data, frac, it):
    """

    """
    # Prepare smoothed data storage
    smoothed_data = pd.DataFrame()

    for _, file_data in data.groupby(by='FileId'):
        # Select columns ending with X, Y, or Z
        tmp = file_data.filter(regex='X$|Y$|Z$')

        # Smooth selected features using LOWESS smoother
        smoothed = tmp.apply(lowess_function, args=(frac, it), axis=0)
        file_data = file_data.assign(**smoothed)

        smoothed_data = pd.concat([smoothed_data, file_data])

    return smoothed_data


def save_checkpoints(checkpoints, target_dir, filename="my_checkpoints.pth"):
    """
    
    """
    print(f"Saving checkpoints in the directory: {target_dir}")
    torch.save(checkpoints, os.path.join(target_dir, filename))


def plot_curves(losses, target_dir=None):
    """

    """
    # Extract number of epochs
    n_epochs = len(losses['train_loss'])
    
    # Draw two curves on a single plot
    plt.plot(
        range(1, n_epochs + 1),
        losses['train_loss'],
        label='Training Loss Curve'
    )

    plt.plot(
        range(1, n_epochs + 1),
        losses['valid_loss'],
        label='Validation Loss Curve'
    )
    
    # Add title and axis labels
    plt.title('Loss Curves for Training and Validation Data')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE Loss')

    # Add legend
    plt.legend()

    # If necessary save the plot to a file
    if target_dir:
        plt.savefig(os.path.join(target_dir, 'plot.png'))
    
    # Display the plot 
    plt.show()