import face_recognition
import os
import cv2
import numpy as np


# specify parameters
known_dir = 'known'
unknown_dir = 'unknown'
tolerance = 0.6 # high tol gives many false postive
frame_thickness =1
font_thickness = 2
model = 'cnn' # hog


# load known faces
known_faces = []
known_names = []
print('loading known faces')
for name in os.listdir(known_dir):
    name_dir = known_dir + '/' + name
    for filename in os.listdir(name_dir):
        file_path = name_dir + '/' + filename
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)

        # skip bad recognition from known
        if len(encodings)==0:
            print('WARNING: face_encoding on ', file_path, 'return 0 faces')
            continue
        known_faces.append(encodings[0])
        known_names.append(name)

# load unknown faces
print('loading unknown faces')
for filename in os.listdir(unknown_dir):
    file_path = unknown_dir + '/' + filename
    image = face_recognition.load_image_file(file_path)
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings, locations):

        # compare to known faces, and return a list of boolen
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance)

        # sometimes, more than one matched faces, pick the closest 
        name = 'unknown'
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_idx = np.argmin(face_distances)

        # only if the min distance is also founded
        if matches[best_idx]:         
            name = known_names[best_idx]

        print('Match Found: %s' % name)

        # unknow -> red; known -> green
        color=[0, 255, 0]
        if name=='unknown':
            color = [0, 0, 255]

        # draw box around the face
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

        # draw box for displaying the name
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+17)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, name, (face_location[3]+2, face_location[2]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), font_thickness)

    cv2.imshow('my_window', image)
    cv2.waitKey(5000) # wait 5 sec each image

