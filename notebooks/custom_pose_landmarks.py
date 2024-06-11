class CustomPoseLandmark():
    """
    Custom class for handling pose landmarks with optional custom landmarks.

    Attributes:
        mp_pose: MediaPipe pose module.
        selected_values (list): List of selected pose landmark indices.
        custom_landmarks (dict, optional): Dictionary of custom landmarks.
    """
    def __init__(self, mp_pose, selected_values=list(range(33)), custom_landmarks=None):
        """
        Initializes the CustomPoseLandmark with MediaPipe pose, selected values, and custom landmarks.

        Args:
            mp_pose: MediaPipe pose module.
            selected_values (list): List of selected pose landmark indices.
            custom_landmarks (dict, optional): Dictionary of custom landmarks.
        """
        # Initialize MediaPipe solutions
        self.mp_pose = mp_pose

        # Initialize selected values and custom landmark list
        self.selected_values = selected_values
        self.custom_landmarks = custom_landmarks


    def __len__(self):
        """
        Returns the number of elements in the pose landmark dictionary.

        Returns:
            int: Number of elements.
        """
        return len(self.get_dictionary())
    
    
    def num_elements(self):
        """
        Returns the number of elements in the pose landmark dictionary.

        Returns:
            int: Number of elements.
        """
        return len(self.get_dictionary())


    def generate_mapping(self):
        """
        Generates a mapping of selected values to their indices.

        Returns:
            dict: Mapping of selected values to indices.
        """
        # Create a mapping storage
        mapping = {}

        for index, value in enumerate(self.selected_values):
            mapping[value] = index

        return mapping


    def get_value(self, item):
        """
        Retrieves the mapped value for a given item, which can be an index or a custom landmark name.

        Args:
            item (int or str): Item to retrieve the mapped value for.

        Returns:
            int: Mapped value of the item.
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
        

    def get_dictionary(self):
        """
        Creates a dictionary mapping custom values to pose landmark names.

        Returns:
            dict: Dictionary mapping custom values to pose landmark names.
        """
        # Create a dictionary to store a new values and pose landmark names
        dictionary = dict()

        for value in self.selected_values:
            # Extract pose landmark name from MediaPipe solutions
            name = self.mp_pose.PoseLandmark(value).name
            # Get mapped value
            custom_value = self.get_value(value)
            # Sign mapped value and pose landmark name to the dictionary
            dictionary[custom_value] = name

        # Check if custom landmarks has been entered
        if self.custom_landmarks:
            for name in self.custom_landmarks.keys():
                # Get new value
                custom_value = self.get_value(name)
                # Sign new created value and pose landmark name to the dictionary
                dictionary[custom_value] = name

        return dictionary
    

    def get_reverse_dictionary(self):
        """
        Creates a dictionary mapping pose landmark names to custom values.

        Returns:
            dict: Dictionary mapping pose landmark names to custom values.
        """
        # Create dictionary to store pose landmark names and mapped values
        landmarks = self.get_dictionary()
        names = {name: value for value, name in landmarks.items()
        }

        return names
    

    def get_connections(self):
        """
        Retrieves the connections between pose landmarks, including custom landmarks if provided.

        Returns:
            set: Set of connections between pose landmarks.
        """
        # Create dictionary using get_reverse_dictionary method     
        names = self.get_reverse_dictionary()

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