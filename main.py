from flask import Flask, request
import os
import cv2
import threading
import time
import numpy as np
from models import Bookshelf


app = Flask(__name__)

logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
bookshelf = Bookshelf()

MAX_POSITION = 740

@app.route("/data", methods=["POST"])
def data():
    # print("Received:", request.json)
    position = request.json

    if position > MAX_POSITION:
        return "OK", 200
    book = bookshelf.get(position)

    # TODO: don't return till audio has played
    if book:
        print(book)

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
