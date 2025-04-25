from flask import Flask, request
import os
import cv2
import threading
import time
import numpy as np
from models import Bookshelf
import requests
import vlc
import time
import json

app = Flask(__name__)

bookshelf = Bookshelf()

MAX_POSITION = 740

# Load the mappings.json to map titles to mp3 files
with open(os.path.join(os.path.dirname(__file__), "mappings.json")) as f:
    mappings = json.load(f)

@app.route("/data", methods=["POST"])
def data():
    # print("Received:", request.json)
    position = request.json

    if position > MAX_POSITION:
        return "OK", 200
    book = bookshelf.get(position)


    if book:
        print(f"Detected book position: {book}")
        
        # Get the actual title from LLM
        title = bookshelf.get_title_from_llm(position)
        title = title.strip()
        
        print(f"LLM identified: {title}")
        if title in mappings:
            play_book(mappings[title])
        else:
            print("Title not found in mappings.")
        url = "http://10.0.0.149:80/command"
        headers = {"Content-Type": "text/plain"}
        payload = "done"

        response = requests.post(url, data=payload, headers=headers)
        print("Status:", response.status_code)
        print("Response:", response.text)
    
    return "OK", 200


def play_book(audio_filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Folder where this script lives
    audio_path = os.path.join(base_dir, "audios", audio_filename)
    print("Playing: ", audio_path)
    p = vlc.MediaPlayer(audio_path)
    p.set_rate(2)
    p.play()
    state = p.get_state()
    while state != vlc.State.Ended:
        time.sleep(1)
        state = p.get_state()
    

def schedule_bookshelf_updates(n: int = 600):
    """Function to schedule bookshelf updates every minute"""
    while True:
        bookshelf.update_positions()
        print(bookshelf.books)
        time.sleep(n)


if __name__ == "__main__":
    # Start the bookshelf update scheduler in a separate thread
    update_thread = threading.Thread(target=schedule_bookshelf_updates, daemon=True)
    update_thread.start()

    # Run the Flask app
    app.run(host="0.0.0.0", port=8080)
