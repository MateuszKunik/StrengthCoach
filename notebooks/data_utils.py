import pandas as pd
import cv2

from utils import *
from canonical_form import PoseCanonicalForm


def prepare_dataframe(dictionary):
    """
    
    """
    axes = ['X', 'Y', 'Z']
    names = dictionary.values()
    column_names = [('_'.join([name, axis])).lower() for name in names for axis in axes]

    return pd.DataFrame(columns = ['Id', 'timestamp'] + column_names)


def video2frame(
        source,
        mp_pose,
        mp_drawing,
        custom_pose,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        canonical_form=False,
        video_display=False
):
    """
    
    """
    # Prepare empty dataframe based on custom pose landmarks
    dictionary = custom_pose.get_dictionary()
    dataframe = prepare_dataframe(dictionary)
    # Get Id from source 
    Id = source.name.split('.')[0]
    # Reset time
    time = 0 
    
    # Capture the video from source
    cap = cv2.VideoCapture(source.__str__())

    # Setup MediaPipe instance
    with mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
        enable_segmentation=False
    ) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            
            # Check if frame reading was successful
            if ret:
                time += 1
                
                # Recolor image for image processing
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Image MediaPipe processing -> detection
                results = pose.process(image)

                # Recolor back to BGR for visualization
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                # Get custom pose landmarks
                custom_pose_landmarks = get_custom_landmarks(
                    mp_pose=mp_pose,
                    custom_pose=custom_pose,
                    landmarks=results.pose_landmarks,
                    custom_landmarks=list(custom_pose.custom_landmarks.keys()))

                if time == 1:
                    # Initialize PoseCanonicalForm class
                    canonical = PoseCanonicalForm(
                        mp_pose=mp_pose,
                        custom_pose=custom_pose,
                        landmark_list=custom_pose_landmarks,
                        default_position=np.array([1, 0, 1]),
                        default_point=np.zeros(shape=(3,)),
                        default_length=0.25
                    )

                # Prepare a single record storage 
                record = [Id, time]
                
                # Create record containing video id, actual time and pose landmark coordinates
                for landmark in custom_pose_landmarks.landmark:
                    # Extract pose landmarks coordinates and store as array
                    coordinates = landmark2array(landmark)[:3]
                    
                    # Check if canonical form has been chosen
                    if canonical_form:
                        # Transform coordinates using canonical form transformations
                        transformed = canonical.transform(coordinates)
                    # Save in record storage
                    record += transformed.tolist()
    
                # Save collected data in DataFrame format
                dataframe = pd.concat(
                    [dataframe, pd.DataFrame([record], columns=dataframe.columns)],
                    ignore_index=True
                )

                
                # Display video if necessary
                if video_display:
                    mp_drawing.draw_landmarks(
                        image,
                        landmark_list=custom_pose_landmarks,
                        connections=custom_pose.get_connections()
                    )

                    cv2.imshow(f'Video: {Id}', image)
                    
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break

                    if not cap.isOpened():
                        exit()
            
            else:
                break

    cap.release()
    cv2.destroyAllWindows()

    return dataframe


def generate_dataframe(
        path,
        mp_pose,
        mp_drawing,
        custom_pose,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        canonical_form=False,
        video_display=False
):
    """
    
    """
    # Prepare template dataframe using custom pose landmarks
    dictionary = custom_pose.get_dictionary()
    dataframe = prepare_dataframe(dictionary)

    for file_path in path.iterdir():
        tmp = video2frame(
            file_path,
            mp_pose,
            mp_drawing,
            custom_pose,
            detection_confidence,
            tracking_confidence,
            canonical_form,
            video_display
        )

        dataframe = pd.concat([dataframe, tmp], ignore_index=True)

    return dataframe