# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:22:44 2023

@author: Rizal Fahmi Alifa
"""

import cv2
import face_recognition

# Muat gambar yang akan dikenali
person1_image = face_recognition.load_image_file("person1.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("person2.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding
]

known_face_names = [
    "Fahmi",
    "Ulum"
]

# inisialisasi beberapa variabel
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Mulai capture/tangkap video
cap = cv2.VideoCapture(0)

while True:
    # tangkap setiap frame video
    ret, frame = cap.read()

    # Resize frame dari video menjadi 1/4 ukuran
    # untuk mempercepat face detection processing 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Konversi gambar dari BGR (yang digunakan OpenCV)
    # ke RGB (yang digunakan face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Temukan semua wajah dan encoding wajah dalam frame video saat ini
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Lihat apakah wajah cocok dengan wajah yang dikenal
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Jika kecocokan ditemukan di known_face_encodings, gunakan yang pertama.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Tampilkan Hasilnya
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Gambar kotak di sekitaran wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Gambar label dengan nama di bawah muka
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Tampilkan hasil gambar
    cv2.imshow('Video', frame)

    # Jika tombol q ditekan, tutup jendela dan keluar dari program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
