import os
import numpy as np
import pandas as pd
import cv2
import random

from itertools import product

from utils import get_custom_landmarks, landmark2array


class Video2DataFrame():
    """
    
    """
    def __init__(self, mp_pose, mp_drawing, custom_pose, labels=True):
        # Initialize main objects
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.custom_pose = custom_pose
        self.labels = labels

        # 
        self.labels_to_extract = self.get_labels()


    def get_labels(self):
        """
        
        """
        # Create a list of all available labels contained in the file name
        available_labels = ['FileId', 'Id', 'CameraPosition', 'SetNumber', 'Repetitions', 'RepNumber', 'Load', 'Lifted']

        # Check the type and assign labels names to extract
        if isinstance(self.labels, list):
            to_extract = self.labels

        elif isinstance(self.labels, bool):
            if self.labels:
                # Take all available labels
                to_extract = available_labels
            else:
                # Take FileId only
                to_extract = [available_labels[0]]
        else:
            print("Wrong data type.")

        return to_extract


    def extract_labels(self, source):
        """
        
        """
        # Get file name without extension
        file_name = os.path.splitext(os.path.basename(source))[0]

        # Extract labels from file name
        labels = file_name.split('_')

        Id = int(labels[0])
        # Create a camera position value mapping
        position_mapping = {'L': 'left', 'C': 'center', 'R': 'right'}
        camera_position = position_mapping[labels[-1]]

        set_number, repetitions, rep_number, load, lifted = list(
            map(int, labels[1: -1]))
        
        if load % 5 != 0:
            load += 0.5
        
        # Create a dictionary to store labels
        extracted = {
            'FileId': file_name,
            'Id': Id,
            'CameraPosition': camera_position,
            'SetNumber': set_number,
            'Repetitions': repetitions,
            'RepNumber': rep_number,
            'Load': load,
            'Lifted': lifted}
        
        # Return a new dictionary based on the get_labels method
        return {key: value for key, value in extracted.items() if key in self.labels_to_extract}


    def prepare_dataframe(self):
        """
        
        """
        # Get landmark names from the CustomPoseLandmark class
        landmarks = self.custom_pose.get_dictionary().values()
        # Prepare a list of coordinate axis names
        axes = ['X', 'Y', 'Z']

        # Prepare a storage for column names
        column_names = self.labels_to_extract + ['Timestamp']
        # Generate all combinations of landmark names and axes
        combinations = []
        for landmark, axis in list(product(landmarks, axes)):
            combinations.append(''.join([landmark.title().replace('_', ''), axis]))

        # Extend column names by combinations
        column_names.extend(combinations)

        return pd.DataFrame(columns=column_names)
    

    def convert_video(self, source, detection, tracking, video_display):
        """
        
        """
        # Prepare empty dataframe using prepare_dataframe method
        dataframe = self.prepare_dataframe()
        # Extract labels using extract_labels method
        labels = self.extract_labels(source)
        # Get file ID
        file_id = labels['FileId']


        # Reset time step
        time_step = 0

        print(f"Converting {file_id} file to dataframe...")

        # Capture the video from source
        cap = cv2.VideoCapture(source.__str__())

        # Setup MediaPipe instance
        with self.mp_pose.Pose(
            min_detection_confidence=detection,
            min_tracking_confidence=tracking,
            enable_segmentation=False
        ) as pose:
            while cap.isOpened():
                ret, image = cap.read()

                # Check if frame reading was successful
                if ret:
                    time_step += 1

                    # Resize image
                    image = cv2.resize(image, (720, 1280))
                    # Recolor image for image processing
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    # Image MediaPipe processing -> detection
                    results = pose.process(image)
                    # Recolor back to BGR for visualization
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Prepare a single record storage 
                    record = list(labels.values()) + [time_step]
                    
                    if results.pose_landmarks:
                        # Get custom pose landmarks
                        custom_pose_landmarks = get_custom_landmarks(
                            self.mp_pose, self.custom_pose, results.pose_landmarks)
                        
                        # Create record containing file id, time step and pose landmark coordinates
                        for landmark in custom_pose_landmarks.landmark:
                            # Extract pose landmarks coordinates and store as array
                            coordinates = landmark2array(landmark)[:3]
                            # Save in record storage
                            record += coordinates.tolist()
                        
                        # Display video if necessary
                        if video_display:
                            connections = self.custom_pose.get_connections()
                            self.mp_drawing.draw_landmarks(
                                image, custom_pose_landmarks, connections)
                            
                            cv2.imshow(f'You are watching {file_id} video.', image)

                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

                            if not cap.isOpened():
                                exit()

                    else:
                        # Protection against missing results
                        record += np.zeros(len(dataframe.columns) - len(record)).tolist()
                        print(f"Please check {file_id} file in {time_step}th frame.")

                    # Save collected data in DataFrame format
                    dataframe = pd.concat(
                        [dataframe, pd.DataFrame([record], columns=dataframe.columns)],
                        ignore_index=True)        
                    
                else:
                    break

        cap.release()
        cv2.destroyAllWindows()  

        return dataframe


    def get_dataframe(self, source, n_samples=None, detection=0.5, tracking=0.5, video_display=False):
        """
        
        """
        # Check where a given path leads
        if os.path.isdir(source):
            # Prepare empty dataframe based on CustomPoseLandmark class
            dataframe = self.prepare_dataframe()

            # Create list of all files from main directory
            files = os.listdir(source)
            
            if n_samples:
                # Create a list of randomly selected paths
                files = random.sample(files, n_samples)

            for file in files:
                # Create a file path
                file_path = os.path.join(source, file)
                # Convert each video and save it to a common DataFrame
                tmp = self.convert_video(file_path, detection, tracking, video_display)
                dataframe = pd.concat([dataframe, tmp], ignore_index=True)

        elif os.path.isfile(source):
            dataframe = self.convert_video(source, detection, tracking, video_display)

        else:
            print("File read failed. Check a given path.")

        return dataframe