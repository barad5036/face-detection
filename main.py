import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from datetime import datetime
import os
from github import Github
from PIL import Image

# Load environment variables for GitHub credentials
username = os.getenv("barad5036")
token = os.getenv("GITHUB_TOKEN")
repo_name = "FD.1"

# Set up Streamlit interface
st.title("Face Detection")
detected_faces_dir = "Detected_Faces"

# Ensure directory exists
if not os.path.exists(detected_faces_dir):
    os.makedirs(detected_faces_dir)
st.sidebar.metric("Total Saved Faces", len(os.listdir(detected_faces_dir)))

# Function to process uploaded images
def process_image(image):
    detector = MTCNN()
    img = Image.open(image)
    image_array = np.array(img)

    if image_array.size == 0:
        st.error("Invalid image data")
        return None, 0

    faces = detector.detect_faces(image_array)
    face_count = 0

    if not faces:
        st.warning("No faces detected")
        return image_array, face_count

    for face in faces:
        x, y, w, h = face['box']
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_face = image_array[y:y + h, x:x + w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_name = f"face_{timestamp}_{np.random.randint(1000)}.png"
            cv2.imwrite(os.path.join(detected_faces_dir, face_name), cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            face_count += 1

    return image_array, face_count

# Upload image and process
camera_image = st.camera_input("Take a picture")
if camera_image:
    result, count = process_image(camera_image)
    if result is not None:
        st.image(result, caption="Processed Image")
        st.success(f"Detected and saved {count} faces" if count > 0 else "No faces saved")

    # Push detected faces to GitHub
    if username and token:
        g = Github(token)
        repo = g.get_user().get_repo(repo_name)

        for file_name in os.listdir(detected_faces_dir):
            file_path = os.path.join(detected_faces_dir, file_name)
            with open(file_path, "rb") as file:
                content = file.read()

            try:
                existing_file = repo.get_contents(f"{detected_faces_dir}/{file_name}", ref="main")
                sha = existing_file.sha
            except Exception:
                sha = None

            try:
                if sha is None:
                    repo.create_file(
                        f"{detected_faces_dir}/{file_name}",
                        f"Add {file_name}",
                        content,
                        branch="main"
                    )
                else:
                    repo.update_file(
                        f"{detected_faces_dir}/{file_name}",
                        f"Update {file_name}",
                        content,
                        sha,
                        branch="main"
                    )
            except Exception as e:
                st.error(f"Error pushing file {file_name}: {e}")
    else:
        st.warning("GitHub credentials are not configured.")

# Sidebar for live video feed
st.sidebar.header("Live Camera Feed")
st.sidebar.write("Live camera feed is not supported on Streamlit Cloud. Use a local environment for this feature.")
