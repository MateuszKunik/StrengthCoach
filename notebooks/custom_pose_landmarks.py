class CustomPoseLandmark():
    """
    
    """
    def __init__(self, mp_pose, selected_values=list(range(33)), custom_landmarks=None):
        # Initialize MediaPipe solutions
        self.mp_pose = mp_pose

        # Initialize selected values and custom landmark list
        self.selected_values = selected_values
        self.custom_landmarks = custom_landmarks


    def generate_mapping(self):
        """
        
        """
        # Create a mapping storage
        mapping = {}

        for index, value in enumerate(self.selected_values):
            mapping[value] = index

        return mapping


    def get_value(self, item):
        """
        
        """
        # Create dictionary using generate_mapping method
        mapping = self.generate_mapping()

        # Check the type of entered value
        if type(item) == int:
            if item in mapping:
                return mapping[item]
            else:
                print("Error value")
        
        elif type(item) == str:
            if item in self.custom_landmarks:
                return len(mapping) + list(self.custom_landmarks.keys()).index(item)
            else:
                print("Error value")

        else:
            print("Error value type")
        

    def get_landmarks(self):
        """
        
        """
        # Create a dictionary to store a custom pose landmark names
        landmarks = dict()

        for value in self.selected_values:
            # Extract pose landmark name from MediaPipe solutions
            name = self.mp_pose.PoseLandmark(value).name
            # Get mapped value
            custom_value = self.get_value(value)
            # Sign mapped value and pose landmark name to the dictionary
            landmarks[custom_value] = name

        # Check if custom landmarks has been entered
        if self.custom_landmarks:
            for name in self.custom_landmarks.keys():
                # Get new value
                custom_value = self.get_value(name)
                # Sign new created value and pose landmark name to the dictionary
                landmarks[custom_value] = name

        return landmarks
    

    def get_names(self):
        """
        
        """
        # Create dictionary using get_landmarks method
        landmarks = self.get_landmarks()
        names = {name: value for value, name in landmarks.items()
        }

        return names
    

    def get_connections(self):
        """
        
        """
        # Create dictionary using get_names method     
        names = self.get_names()

        # Create a connections storage
        connections = set()

        for connection in self.mp_pose.POSE_CONNECTIONS:
            # Extract default connection values from MediaPipe solutions
            value_1, value_2 = connection
            # Check if the values are expected
            if (value_1 in self.selected_values) and (value_2 in self.selected_values):
                # Get custom values using get_value method
                custom_value_1 = self.get_value(value_1)
                custom_value_2 = self.get_value(value_2)

                # Create a connection for custom values
                connections.add((custom_value_1, custom_value_2))

        # Check if custom landmarks has been entered
        if self.custom_landmarks:
            for custom_landmark in self.custom_landmarks.keys():
                # Get custom value for completely new landmark
                custom_value_1 = names[custom_landmark]

                for landmark in self.custom_landmarks[custom_landmark]:
                    # Get custom value using names dictionary
                    custom_value_2 = names[landmark]

                    # Create a connection for custom values
                    connections.add((custom_value_1, custom_value_2))
                    
        return connections