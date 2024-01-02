class CustomPoseLandmark():
    """
    
    """
    def __init__(self, mp_pose):
        # Initialize MediaPipe solutions
        self.mp_pose = mp_pose

        # Selected values of pose landmarks corresponding to PoseLandmark class from MediaPipe library
        self.values = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32]


    def get_landmarks(self):
        """
        
        """
        # Create a dictionary to store a custom pose landmark names
        landmarks = dict()

        for value in self.values:
            # Extract pose landmark name from MediaPipe solutions
            name = self.mp_pose.PoseLandmark(value).name
            # Get mapped value
            custom_value = self.get_custom_value(value)
            # Sign mapped value and pose landmark name to the dictionary
            landmarks[custom_value] = name
            
        # Add a completely new pose landmark
        landmarks[len(landmarks)] = 'THORAX'

        return landmarks
    

    def generate_mapping(self):
        """
        
        """
        # Create a mapping storage
        mapping = {}

        for index, value in enumerate(self.values):
            mapping[value] = index

        return mapping


    def get_custom_value(self, value):
        """
        
        """
        mapping = self.generate_mapping()

        if value in mapping:
            return mapping[value]
        else:
            return len(mapping)


    def get_connections(self):
        """
        
        """
        # Create a connections storage
        connections = set()

        for connection in self.mp_pose.POSE_CONNECTIONS:
            # Extract default connection values from MediaPipe solutions
            value_1, value_2 = connection
            # Check if the values are expected
            if (value_1 in self.values) and (value_2 in self.values):
                # Get custom values
                custom_1 = self.get_custom_value(value_1)
                custom_2 = self.get_custom_value(value_2)

                # Create a connection for new values
                connections.add((custom_1, custom_2))

        # Add a custom connection between NOSE and THORAX
        connections.add((self.get_custom_value(0), self.get_custom_value(33)))

        return connections