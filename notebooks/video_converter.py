import os
import numpy as np
import pandas as pd
import cv2
import random

from itertools import product

from utils import get_custom_landmarks, landmark2array
# from canonical_form import PoseCanonicalForm


class Video2DataFrame():
    """
    
    """
    def __init__(
            self,
            mp_pose,
            mp_drawing,
            custom_pose,
    ):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.custom_pose = custom_pose


    def prepare_dataframe(self):
        """
        
        """
        # Get landmark names from the CustomPoseLandmark class
        landmarks = self.custom_pose.get_dictionary().values()
        # Prepare a list of coordinate axis names
        axes = ['X', 'Y', 'Z']

        # Prepare a storage for column names
        column_names = ['FileId', 'Timestamp']
        # Generate all combinations of landmark names and axes
        combinations = ['_'.join(comb) for comb in list(product(landmarks, axes))]
        # Extend column names by combinations
        column_names.extend(combinations)

        return pd.DataFrame(columns=column_names)


    def convert_video(self, file_path, detection, tracking, video_display):
        """
        
        """
        # Prepare empty dataframe based on CustomPoseLandmark class
        dataframe = self.prepare_dataframe()
        # Get file ID from file path
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        # Reset time step
        time_step = 0

        print(f"Converting {file_id} file to dataframe...")

        # Capture the video from source
        cap = cv2.VideoCapture(file_path.__str__())

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
                    record = [file_id, time_step]
                    
                    if results.pose_landmarks:
                        # Get custom pose landmarks
                        custom_pose_landmarks = get_custom_landmarks(
                            mp_pose=self.mp_pose,
                            custom_pose=self.custom_pose,
                            landmarks=results.pose_landmarks)
                        
                        # Create record containing file id, time step and pose landmark coordinates
                        for landmark in custom_pose_landmarks.landmark:
                            # Extract pose landmarks coordinates and store as array
                            coordinates = landmark2array(landmark)[:3]
                            # Save in record storage
                            record += coordinates.tolist()
                        
                        # Display video if necessary
                        if video_display:
                            self.mp_drawing.draw_landmarks(
                                image,
                                landmark_list=custom_pose_landmarks,
                                connections=self.custom_pose.get_connections()
                            )
                            
                            cv2.imshow(f'You are watching {file_id} video.', image)

                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

                            if not cap.isOpened():
                                exit()

                    else:
                        # Protection against missing results
                        record += np.zeros(len(dataframe.columns) - len(record)).tolist()

                    # Save collected data in DataFrame format
                    dataframe = pd.concat(
                        [dataframe, pd.DataFrame([record], columns=dataframe.columns)],
                        ignore_index=True)
                
                else:
                    print("Frame read failed.")
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

            # Create list of all paths from main directory
            path_list = list(source.iterdir())
            
            if n_samples:
                # Create a list of randomly selected paths
                path_list = random.sample(path_list, n_samples)

            for file_path in path_list:
                # Convert each video and save it to a common DataFrame
                tmp = self.convert_video(file_path, detection, tracking, video_display)
                dataframe = pd.concat([dataframe, tmp], ignore_index=True)

        elif os.path.isfile(source):
            dataframe = self.convert_video(source, detection, tracking, video_display)

        else:
            print("File read failed. Check a given path.")

        return dataframe