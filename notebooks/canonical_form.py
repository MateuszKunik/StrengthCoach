import math as m
import numpy as np

from utils import landmark2array


class PoseCanonicalForm():
    """
    A class to transform pose landmarks into a canonical form.

    This class handles the rotation, translation, and scaling of pose landmarks 
    to a standardized form, making comparisons and further processing easier.

    Args:
        mp_pose (object): MediaPipe pose object.
        custom_pose (object): Custom pose object with mappings.
        landmark_list (object): List of pose landmarks.
        default_position (np.array, optional): Default vector position for alignment. Default is [1, 0, 1].
        default_point (np.array, optional): Default point for translation. Default is [0, 0, 0].
        default_length (float, optional): Default length for scaling. Default is 0.25.

    Attributes:
        left_hip (np.array): Coordinates of the left hip.
        right_hip (np.array): Coordinates of the right hip.
        hips_vector (np.array): Vector from right hip to left hip.
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
        Calculates the rotation angle (alpha) required to align the hips vector with the default position vector.

        Returns:
            float: Rotation angle in radians.
        """
        # Calculate the dot product of default position vector and hips vector
        dot_product = np.dot(self.default_position, self.hips_vector)

        # Calculate the cosine of default position vector and hips vector
        cos_alpha = dot_product / (
            np.linalg.norm(self.default_position) * np.linalg.norm(self.hips_vector)
            )

        # Protection against numerical errors
        bounded_cos_alpha = min(1, max(-1, cos_alpha))

        # Calculate alpha using arccosine function
        alpha = np.arccos(bounded_cos_alpha)

        return alpha
        
    
    def rotate(self, coordinates):
        """
        Rotates the given coordinates by the calculated rotation angle.

        Args:
            coordinates (np.array): Coordinates to rotate.

        Returns:
            np.array: Rotated coordinates.
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
        Calculates the translation vector required to move the pelvis to the default point.

        Returns:
            np.array: Translation vector.
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
        Translates the given coordinates by the calculated translation vector.

        Args:
            coordinates (np.array): Coordinates to translate.

        Returns:
            np.array: Translated coordinates.
        """
        vector = self.translation_procedure()

        return vector + coordinates


    def scaling_procedure(self):
        """
        Calculates the scale factor to adjust the length of the hips vector to the default length.

        Returns:
            float: Scale factor.
        """
        # Calculate the length of hips vector
        hip_length = np.linalg.norm(self.hips_vector)

        # Calculate the scale factor
        scale_factor = self.default_length / hip_length

        return scale_factor
    

    def scale(self, coordinates):
        """
        Scales the given coordinates by the calculated scale factor.

        Args:
            coordinates (np.array): Coordinates to scale.

        Returns:
            np.array: Scaled coordinates.
        """
        scale_factor = self.scaling_procedure()

        return scale_factor * coordinates
    

    def transform(self, coordinates):
        """
        Applies rotation, translation, and scaling to transform the given coordinates to the canonical form.

        Args:
            coordinates (np.array): Coordinates to transform.

        Returns:
            np.array: Transformed coordinates.
        """
        
        transformed = self.scale(self.translate(self.rotate(coordinates)))

        return transformed