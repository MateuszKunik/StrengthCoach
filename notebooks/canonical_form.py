import math as m
import numpy as np

from utils import *


class PoseCanonicalForm():
    """
    
    """
    def __init__(
            self,
            mp_pose,
            custom_pose,
            landmark_list,
            default_position=np.array([1, 0, 1]),
            default_point=np.zeros(shape=(3,)),
            default_length=0.25
):      
        # Initialize 
        self.mp_pose = mp_pose
        self.custom_pose = custom_pose
        self.landmark_list = landmark_list

        self.default_position = default_position
        self.default_point = default_point
        self.default_length = default_length

        # Extract the coordinates of left and right hips for rotation and scaling
        self.left_hip = landmark2array(
            landmark_list.landmark[
                self.custom_pose.get_value(
                    self.mp_pose.PoseLandmark.LEFT_HIP.value
                )
            ])[:3]
        
        self.right_hip = landmark2array(
            landmark_list.landmark[
                self.custom_pose.get_value(
                    self.mp_pose.PoseLandmark.RIGHT_HIP.value
                )
            ])[:3]
        
        # Determine the hips vector
        self.hips_vector = self.left_hip - self.right_hip

        
    def rotation_procedure(self):
        """
        
        """
        # Calculate the dot product of default position vector and hips vector
        dot_product = np.dot(self.default_position, self.hips_vector)

        # Calculate the cosine of default position vector and hips vector
        cos_alpha = dot_product / (
            np.linalg.norm(self.default_position) * np.linalg.norm(self.hips_vector)
            )

        # Protection against numerical errors - cosine value should be in the range [-1, 1]
        bounded_cos_alpha = min(1, max(-1, cos_alpha))

        # Calculate alpha using arccosine function
        alpha = np.arccos(bounded_cos_alpha)

        return alpha
        
    
    def rotate(self, coordinates):
        """
        
        """
        alpha = self.rotation_procedure()

        rotation_matrix = np.array(
            [
                [m.cos(alpha), 0, m.sin(alpha)],
                [0, 1, 0],
                [-m.sin(alpha), 0, m.cos(alpha)],
            ]
        )

        return np.dot(rotation_matrix, coordinates)


    def translation_procedure(self):
        """
        
        """
        # Extract the coordinates of pelvis from landmark list
        pelvis = landmark2array(
            self.landmark_list.landmark[
                self.custom_pose.get_value('PELVIS')
            ])[:3]

        # Determine a vector anchored at the origin and pelvis point
        vector = self.default_point - pelvis

        return vector


    def translate(self, coordinates):
        """
        
        """
        vector = self.translation_procedure()

        return vector + coordinates


    def scaling_procedure(self):
        """
        
        """
        # Calculate the length of hips vector
        hip_length = np.linalg.norm(self.hips_vector)

        # Calculate the scale factor
        scale_factor = self.default_length / hip_length

        return scale_factor
    

    def scale(self, coordinates):
        """
        
        """
        scale_factor = self.scaling_procedure()

        return scale_factor * coordinates
    

    def transform(self, coordinates):
        """
        
        """
        
        transformed = self.scale(self.translate(self.rotate(coordinates)))

        return transformed