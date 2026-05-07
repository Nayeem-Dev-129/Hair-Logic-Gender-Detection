from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

from predict import predict_image

# ==========================================
# MAIN WINDOW
# ==========================================

root = Tk()

root.title(
    "Hair Logic Gender Detection"
)

root.geometry("700x750")

root.configure(bg="white")

# ==========================================
# TITLE
# ==========================================

title = Label(
    root,
    text="Hair Logic Gender Detection System",
    font=("Arial",20,"bold"),
    bg="white"
)

title.pack(pady=20)

# ==========================================
# IMAGE PANEL
# ==========================================

panel = Label(root, bg="white")

panel.pack(pady=10)

# ==========================================
# RESULT LABEL
# ==========================================

result_label = Label(
    root,
    text="Upload an Image",
    font=("Arial",16),
    bg="white",
    justify=LEFT
)

result_label.pack(pady=20)

# ==========================================
# UPLOAD FUNCTION
# ==========================================

def upload_image():

    path = filedialog.askopenfilename()

    if path:

        # SHOW IMAGE

        img = Image.open(path)

        img = img.resize((300,300))

        img_tk = ImageTk.PhotoImage(img)

        panel.config(image=img_tk)

        panel.image = img_tk

        # PREDICT

        result = predict_image(path)

        output = f"""

Age: {result["Age"]}

Detected Gender:
{result["Detected Gender"]}

Hair Type:
{result["Hair Type"]}

Final Prediction:
{result["Final Prediction"]}

"""

        result_label.config(text=output)

# ==========================================
# BUTTON
# ==========================================

btn = Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Arial",16),
    padx=20,
    pady=10
)

btn.pack(pady=20)

# ==========================================
# RUN APP
# ==========================================

root.mainloop()