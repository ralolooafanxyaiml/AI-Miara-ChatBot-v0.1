import os
import urllib.request
import zipfile
import shutil

DATA_URL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
ZIP_NAME = "cornell_corpus.zip"
TARGET_DIR = "./raw_data"

def create_dummy_data():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    
    dummy_path = os.path.join(TARGET_DIR, "dummy_chat.txt")
    if not os.path.exists(dummy_path):
        content = "Hello\nHi there!\nHow are you?\nI am fine.\nWhat is your name?\nI am a robot.\nGoodbye\nSee you!"
        with open(dummy_path, "w", encoding="utf-8") as f:
            f.write(content)

def download_and_extract():
    print(f"\n>> Downloading...")
    
    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_NAME)
        print("Download is completed")
    except Exception as e:
        print(f"ERROR!: {e}")
        return

    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    extracted_folder = "cornell movie-dialogs corpus"
    
    if os.path.exists(extracted_folder):
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)

        files_to_move = ["movie_lines.txt", "movie_conversations.txt"]
        
        for file_name in files_to_move:
            src = os.path.join(extracted_folder, file_name)
            dst = os.path.join(TARGET_DIR, file_name)
            
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"   -> Taşındı: {file_name}")
        
        print(">> Cleaning...")
        shutil.rmtree(extracted_folder)
        os.remove(ZIP_NAME)
        
        print(f"\nAll Datasets downloaded in '{TARGET_DIR}' file.")
    else:
        print("ERROR!")

if __name__ == "__main__":
    create_dummy_data()
    choice = input("Want to start to download? (y/n): ").lower()
    
    if choice == 'y':
        download_and_extract()
    else:
        print("Download is won't start, you can use demo version.")
