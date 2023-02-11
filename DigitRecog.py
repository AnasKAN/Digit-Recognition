from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import ctypes

model = load_model('mnist.h5')

def prediction (img):
    # resizing the image to 28 by 28
    img = img.resize((28,28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support the model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # predicting the class
    res = model.predict(img)[0]
    return np.argmax(res), max(res)
#          ^returns the digit ^ return the accuracy

class App(tk.Tk): # inherates from tkinter.Tk
    # constructor
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        
        # creating elements
        self.canvas = tk.Canvas(self, width=300, height = 300, bg = "white", cursor = "cross")
        self.label = tk.Label(self, text="Write here", font = ("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        
        # Grid Structure
        self.canvas.grid(row=0, column = 0, pady = 2)
        self.label.grid(row=0, column = 1, pady = 2, padx = 2)
        self.classify_btn.grid(row=1, column=1,pady=2, padx=2)
        self.button_clear.grid(row=1, column=0,pady=2)
        
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>",self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete('all')
        
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle out of the canvas
        
        #hwnd = ctypes.windll.user32.FindWindowW(0, self.canvas.winfo_id())
        #rect = ctypes.wintypes.RECT()
        #ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))
        
        rect = win32gui.GetWindowRect(HWND)
        a,b,c,d =rect
        rect = (a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect)
        
        digit, acc = prediction(im) # digit is the first returning value where acc is the accuracy
        self.label.configure(text='prediction: '+str(digit)+'\naccuracy: '+str(int(acc*100))+'%')
    
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r,fill='black')
        
app = App()
        
            
    