import streamlit as st
import os
import time
from kb_extraction import *

#
# Write a file locally in the ./data dir
#
def write_file_to_kb(bytes, name):
    path = os.path.join("./data", name)
    newFile = open(path, "wb")
    newFile.write(bytearray(bytes))

#
# Unzip the file passed as parameter
#
def unzip(name):
    import zipfile
    from datetime import datetime
    path_to_zip_file = os.path.join("./data", name)

    path = f"ddg_{name}_{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(path, exist_ok=True)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)
    #now we can recursively read the directory and upsert all files
    return path

@st.cache_resource
def createBot():
    return KnowledgeExtractor()


chatBot = createBot()

#Creating the chatbot interface
st.title("Smart recruitment")
st.subheader("Knowledge Extraction")

#Upload FORM
with st.form("my-form", clear_on_submit=True):
    upload_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    if st.form_submit_button("Upload"):

        if len(upload_files) > 0:
            with st.spinner("Please wait..."):
                for uploaded_file in upload_files:
                    bytes_data = uploaded_file.read()
                    write_file_to_kb(bytes_data, uploaded_file.name)
                upload_files.clear()
                
                container = st.empty()
                if(uploaded_file.name.endswith(".zip")):
                    #decompress the file
                    zip_path = unzip(uploaded_file.name)
                    chatBot.addCurricula(zip_path)
                else:
                    full_path = os.path.join("./data", uploaded_file.name)
                    chatBot.addCurriculum(full_path)
                container.success("File uploaded successfully!")  # Create a success alert
                time.sleep(2)  # Wait 2 seconds
                container.empty()
                #st.success('File uploaded successfully!', icon="✅")
        else:
            st.error("No file has been selected!")

user_input = st.text_area(label='You:', value="", height=200)

# Send request to DC4SE Bot.
if st.button("Submit"):
    res =""
    markdown = True
    ans = chatBot.answer("gabriele", user_input)
    res = ans["response"]
    if ans["markdown"]:
        st.markdown(res)
    else:
        st.text(res)
        

