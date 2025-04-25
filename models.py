import os
import cv2
import numpy as np


class Book:
    def __init__(self, title, author, description, position):
        self.title = title
        self.author = author
        self.description = description
        self.position = position

    def __repr__(self):
        return f"Book(title={self.title}, author={self.author}, description={self.description}, position={self.position})"


class Bookshelf:
    def __init__(self):
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        self.debug = True
        self.num_books = 18
        self.dimension = 70 * 10  # cm -> mm
        self.offset = 6.5 * 10  # mm
        self.books = []
        self.vertical_lines = []
        self.frame = None
        self.cropped_frame = None

    def add(self, book):
        self.books.append(book)

    def remove(self, book):
        self.books.remove(book)

    def get(self, position):
        print(position)
        # calculate L1 distance between position and each book's position
        distances = [abs(book.position - position) for book in self.books]
        return self.books[np.argmin(distances)]

    def update_positions(self):
        cap = cv2.VideoCapture(0)
        ret, self.frame = cap.read()

        if self.debug:
            cv2.imwrite(os.path.join(self.log_dir, "bookshelf.jpg"), self.frame)

        cap.release()

        self._canny_edge_detection()
        self._hough_lines_detection()
        self._kmeans_boundaries()
        self._detect_books()
        # self._crop_to_bookshelf()

    def _crop_to_bookshelf(self):
        # crop to the bookshelf (left and right most detected line)
        if self.vertical_lines is None:
            return

        left_line = self.vertical_lines[0]
        right_line = self.vertical_lines[-1]

        # crop to the bookshelf
        self.cropped_frame = self.frame[
            left_line[1] : right_line[1], left_line[0] : right_line[0]
        ]
        if self.debug:
            cv2.imwrite(
                os.path.join(self.log_dir, "cropped_frame.jpg"), self.cropped_frame
            )

    def _hough_lines_detection(self):

        # detect vertical lines in the cropped frame
        hough_img = self.frame.copy()
        rho = 10  # distance resolution in pixels
        theta = np.pi / 180  # angle resolution in radians (1°)
        threshold = 10  # minimum votes (intersections) to “detect” a line
        min_line_length = 150  # minimum length of a line (in pixels)
        max_line_gap = 50  # maximum gap between segments to link them

        lines = cv2.HoughLinesP(
            self.canny,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        vertical_lines = []
        angle_tolerance = 10 * np.pi / 180  # ±5° tolerance

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2((y2 - y1), (x2 - x1))
                if abs(abs(angle) - np.pi / 2) < angle_tolerance:
                    vertical_lines.append((x1, y1, x2, y2))
                    cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 10)

        self.vertical_lines = vertical_lines
        if self.debug:
            cv2.imwrite(
                os.path.join(self.log_dir, f"hough.jpg"),
                hough_img,
            )

    def _canny_edge_detection(self):
        gray_quantized = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_quantized, (25, 25), 0)
        blurred = cv2.GaussianBlur(blurred, (25, 25), 0)
        blurred = cv2.GaussianBlur(blurred, (25, 25), 0)
        self.blurred = cv2.GaussianBlur(blurred, (25, 25), 0)

        self.canny = cv2.Canny(blurred, 10, 30)

        if self.debug:
            cv2.imwrite(os.path.join(self.log_dir, "canny_sobel.jpg"), self.canny)
            cv2.imwrite(os.path.join(self.log_dir, "blurred.jpg"), self.blurred)

    def _kmeans_boundaries(self):
        boundaries = self.frame.copy()
        x_coords = np.array(
            [(x1 + x2) / 2 for (x1, y1, x2, y2) in self.vertical_lines],
            dtype=np.float32,
        ).reshape(-1, 1)

        n = self.num_books + 1  # + 2
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS
        attempts = 10

        # 3) run kmeans
        compactness, labels, centers = cv2.kmeans(
            data=x_coords,
            K=n,
            bestLabels=None,
            criteria=criteria,
            attempts=attempts,
            flags=flags,
        )

        book_centers = sorted(centers.flatten())

        lines = []
        for x in book_centers:
            x_int = int(round(x))
            cv2.line(
                boundaries, (x_int, 0), (x_int, boundaries.shape[0]), (0, 0, 255), 20
            )
            lines.append(x_int)

        self.boundaries = boundaries
        self.lines = lines

        if self.debug:
            cv2.imwrite(os.path.join(self.log_dir, f"kmeans.jpg"), boundaries)

    def _detect_books(self):
        # detect books in the cropped frame

        # Normalize lines to match the actual bookshelf dimensions
        normalized_lines = []

        # Get the first and last line positions
        first_line = self.lines[0]
        last_line = self.lines[-1]

        # Calculate the scaling factor based on the known dimensions
        scale_factor = self.dimension / (last_line - first_line)

        # Normalize each line position
        for line in self.lines:
            # First shift relative to the first line, then scale, then add the offset
            normalized_position = ((line - first_line) * scale_factor) + self.offset
            normalized_lines.append(int(round(normalized_position)))

        # Replace the original lines with normalized ones
        self.normalized_lines = normalized_lines

        self.books = []
        for i in range(len(self.normalized_lines) - 1):
            curr_book = Book(
                title=f"Book {i + 1}",
                author="",
                description="",
                position=self.normalized_lines[i],
            )
            self.books.append(curr_book)
