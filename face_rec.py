import face_recognition
import os
import cv2
import numpy as np

known_dir = 'known'
unknown_dir = 'unknown'
tolerance = 0.6 # high tol gives many false postive
frame_thickness =1
font_thickness = 2
model = 'cnn' # hog


known_faces = []
known_names = []
print('loading known faces')
for name in os.listdir(known_dir):
    name_dir = known_dir + '/' + name
    for filename in os.listdir(name_dir):
        file_path = name_dir + '/' + filename
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings)==0:
            print('WARNING: face_encoding on ', file_path, 'return 0 faces')
            continue
        known_faces.append(encodings[0])
        known_names.append(name)


print('loading unknown faces')
for filename in os.listdir(unknown_dir):
    file_path = unknown_dir + '/' + filename
    image = face_recognition.load_image_file(file_path)
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance)

        name = 'unknown'
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_idx = np.argmin(face_distances)
        if matches[best_idx]:
            name = known_names[best_idx]

        print('Match Found: %s' % name)

        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        color=[0, 255, 0]
        if name=='unknown':
            color = [0, 0, 255]

        cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)

        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2]+17)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, name, (face_location[3]+2, face_location[2]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), font_thickness)

    cv2.imshow('my_window', image)
    cv2.waitKey(5000)

