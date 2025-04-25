from flask import Flask, request
import os
import cv2
import threading
import time
import numpy as np
from models import Bookshelf
import requests


app = Flask(__name__)

bookshelf = Bookshelf()

MAX_POSITION = 740

@app.route("/data", methods=["POST"])
def data():
    # print("Received:", request.json)
    position = request.json

    if position > MAX_POSITION:
        return "OK", 200
    book = bookshelf.get(position)


    if book:
        print(book)
        url = "http://10.0.0.149:80/command"
        headers = {"Content-Type": "text/plain"}
        payload = "done"

        response = requests.post(url, data=payload, headers=headers)
        print("Status:", response.status_code)
        print("Response:", response.text)
    
    return "OK", 200


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
