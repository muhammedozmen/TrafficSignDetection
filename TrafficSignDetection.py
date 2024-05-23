import cv2
import numpy as np
from tkinter import Tk, Button, filedialog, Toplevel, Label
import threading
import os
import time


# Function to load templates
def load_templates(template_dir):
    templates = []
    for filename in os.listdir(template_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            template = cv2.imread(os.path.join(template_dir, filename), 0)  # Load in grayscale
            templates.append(template)
    return templates


def resize_template_if_needed(image, template):
    img_h, img_w = image.shape[:2]
    temp_h, temp_w = template.shape[:2]
    if temp_h > img_h or temp_w > img_w:
        template = cv2.resize(template, (min(img_w, temp_w), min(img_h, temp_h)), interpolation=cv2.INTER_AREA)
    return template


def color_segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define adjusted color ranges for segmentation
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Create masks for each color range
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combine masks
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_black)

    # Debug: Show the mask
    cv2.imshow("Mask", mask)

    return mask


def edge_detection(image):
    # Adjusting the parameters for Canny edge detection
    edges = cv2.Canny(image, 50, 150)  # Lowered threshold values
    # Debug: Show edges
    cv2.imshow("Edges", edges)
    return edges


def template_matching(image, templates):
    matched = image.copy()
    for template in templates:
        template = resize_template_if_needed(image, template)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6  # Lowered threshold value
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(matched, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    return matched


def shape_analysis(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 8:
            cv2.drawContours(image, [approx], 0, (0, 0, 255), 5)
    return image


def resize_image(image, size=(640, 480)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def process_image(window, root, templates):
    file_path = filedialog.askopenfilename()
    if not file_path:
        window.destroy()
        root.deiconify()
        return

    image = cv2.imread(file_path)
    segmented = color_segmentation(image)
    edges = edge_detection(segmented)
    matched = template_matching(edges, templates)
    analyzed = shape_analysis(matched)
    resized_analyzed = resize_image(analyzed)  # Resize the processed image
    cv2.imshow('Processed Image', resized_analyzed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    window.destroy()
    root.deiconify()


def process_video(window, flag, root, templates):
    file_path = filedialog.askopenfilename()
    if not file_path:
        window.destroy()
        root.deiconify()
        return

    cap = cv2.VideoCapture(file_path)
    while flag[0] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        segmented = color_segmentation(frame)
        edges = edge_detection(segmented)
        matched = template_matching(edges, templates)
        analyzed = shape_analysis(matched)
        resized_analyzed = resize_image(analyzed)  # Resize the processed image
        cv2.imshow('Processed Video', resized_analyzed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()
    root.deiconify()


def real_time_processing(window, flag, root, templates):
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    while flag[0]:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        segmented = color_segmentation(frame)
        edges = edge_detection(segmented)
        matched = template_matching(edges, templates)
        analyzed = shape_analysis(matched)
        resized_analyzed = resize_image(analyzed)  # Resize the processed image

        # Display FPS on frame
        cv2.putText(resized_analyzed, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Real-Time Processing', resized_analyzed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()
    root.deiconify()


def start_real_time_processing(window, root, templates):
    flag = [True]

    def stop_real_time_processing():
        flag[0] = False

    Button(window, text="Stop Processing", command=stop_real_time_processing).pack()

    threading.Thread(target=real_time_processing, args=(window, flag, root, templates)).start()


def start_video_processing(window, root, templates):
    flag = [True]

    def stop_video_processing():
        flag[0] = False

    Button(window, text="Stop Processing", command=stop_video_processing).pack()

    threading.Thread(target=process_video, args=(window, flag, root, templates)).start()


def open_image_processing(root, templates):
    root.withdraw()
    window = Toplevel()
    window.title("Process Image")
    Label(window, text="Processing Image...").pack()
    Button(window, text="Start Processing", command=lambda: process_image(window, root, templates)).pack()
    Button(window, text="Back to Menu", command=lambda: [window.destroy(), root.deiconify()]).pack()


def open_video_processing(root, templates):
    root.withdraw()
    window = Toplevel()
    window.title("Process Video")
    Label(window, text="Processing Video...").pack()
    Button(window, text="Start Processing", command=lambda: start_video_processing(window, root, templates)).pack()
    Button(window, text="Back to Menu", command=lambda: [window.destroy(), root.deiconify()]).pack()


def open_real_time_processing(root, templates):
    root.withdraw()
    window = Toplevel()
    window.title("Real-Time Processing")
    Label(window, text="Processing in Real-Time...").pack()
    Button(window, text="Start Processing", command=lambda: start_real_time_processing(window, root, templates)).pack()
    Button(window, text="Back to Menu", command=lambda: [window.destroy(), root.deiconify()]).pack()


def create_gui():
    root = Tk()
    root.title("Traffic Sign Detection and Recognition")

    templates = load_templates("templates/")

    Label(root, text="Traffic Sign Detection and Recognition").pack()
    Button(root, text="Process Image", command=lambda: open_image_processing(root, templates)).pack()
    Button(root, text="Process Video", command=lambda: open_video_processing(root, templates)).pack()
    Button(root, text="Real-Time Processing", command=lambda: open_real_time_processing(root, templates)).pack()

    root.mainloop()


if __name__ == "__main__":
    create_gui()
