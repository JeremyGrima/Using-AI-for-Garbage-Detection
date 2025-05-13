import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import glob
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("runs/detect/train50_small_200/weights/CustomYOLOV8s.pt")

# Globals to store the current image
last_full_image = None
last_image_path = None

# Resize and show image based on frame size
def show_resized_image():
    global last_full_image

    if last_full_image:
        frame_width = image_frame.winfo_width()
        frame_height = image_frame.winfo_height()

        img = last_full_image.copy()
        img.thumbnail((frame_width, frame_height))

        img_tk = ImageTk.PhotoImage(img)
        image_panel.configure(image=img_tk)
        image_panel.image = img_tk

# Detect and display results
def detect_image():
    global last_full_image, last_image_path

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        results = model(file_path, save=True)

        result_dir = results[0].save_dir
        saved_images = glob.glob(f"{result_dir}/*.jpg") + glob.glob(f"{result_dir}/*.png")

        if saved_images:
            last_image_path = saved_images[0]
            img = Image.open(last_image_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            last_full_image = img
            show_resized_image()
            status_label.config(text="✅ Detection complete.", foreground="#8aff80")
        else:
            status_label.config(text="❌ No result image found.", foreground="#ff8080")

# --- GUI Setup ---
root = tk.Tk()
root.title("YOLOv8 Object Detector - Dark Mode")
root.geometry("850x620")
root.configure(bg="#1e1e1e")

# Make layout responsive
root.columnconfigure(0, weight=1)
root.rowconfigure(3, weight=1)

# ttk Style Configuration
style = ttk.Style()
style.theme_use("default")
style.configure("TButton",
                font=("Segoe UI", 12),
                padding=10,
                background="#2d2d2d",
                foreground="#ffffff")
style.map("TButton",
          background=[("active", "#444444")],
          foreground=[("active", "#ffffff")])
style.configure("TLabel",
                font=("Segoe UI", 11),
                background="#1e1e1e",
                foreground="#cccccc")

# Header Label
header = tk.Label(root, text="Custom YOLOv8 Object Detector",
                  font=("Segoe UI", 16, "bold"),
                  bg="#1e1e1e", fg="#ffffff")
header.grid(row=0, column=0, pady=(20, 10), sticky="n")

# Upload Button
upload_btn = ttk.Button(root, text="Upload and Detect Image", command=detect_image)
upload_btn.grid(row=1, column=0, pady=10)

# Status Label
status_label = ttk.Label(root, text="Upload an image to detect objects.")
status_label.grid(row=2, column=0, pady=5)

# Image Display Frame
image_frame = tk.Frame(root, bg="#2d2d2d", bd=2, relief="groove")
image_frame.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
image_frame.columnconfigure(0, weight=1)
image_frame.rowconfigure(0, weight=1)

# Image Panel
image_panel = tk.Label(image_frame, bg="#2d2d2d")
image_panel.grid(sticky="nsew")

# Responsive resize event
image_frame.bind("<Configure>", lambda e: show_resized_image())

# Run GUI loop
root.mainloop()
