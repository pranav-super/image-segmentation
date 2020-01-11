#"Authored" by: Pranav Subramanian
#Based on https://stackoverflow.com/questions/5501192/how-to-display-picture-and-get-mouse-click-coordinate-on-it

from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk

def pickImage(): 
    ''' A method containing all of the UI elements of the image segmentation program.
        This lets users access a filepicker, pick an image, and pick sink and then source verticies.
        First, a user should select an image, then left click on all the sink vertices, right click,
        left click on all the source verticies, and right click again.
        Output: a tuple containing:
                    - an array of coordinates of source vertices
                    - an array of coordinates of sink vertices
                    - the filename of the image
                    - the dimensions of the image
    '''
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack()

    #adding the image
    File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #get image dimensions - https://stackoverflow.com/questions/6444548/how-do-i-get-the-picture-size-with-pil
    im = Image.open(File, 'r')
    width, height = im.size

    #for our purposes, we need a list containing all the points:
    sinks = []
    sources = []
    global timesPressedEnter #a global variable to allow switching between selecting sinks and sources and turning off
    timesPressedEnter = 0
    global sink #another such variable
    sink = True
        
    #function to be called when mouse is clicked
    def selectCoordinates(event):
        if sink:
            sinks.append(str(event.x-1) + "," + str(event.y-1))
            canvas.create_oval(event.x, event.y, event.x+1, event.y+1, outline="#f11", fill="#f11", width=2)
        else:
            sources.append(str(event.x-1) + "," + str(event.y-1))
            canvas.create_oval(event.x, event.y, event.x+1, event.y+1, outline="#1f1", fill="#1f1", width=2)

    #called on left click
    def switchIt(event):
        global timesPressedEnter
        global sink
        canvas.focus_set()
        timesPressedEnter+=1
        if timesPressedEnter > 1:
            root.destroy()
        else:
            sink = False

    #bind selection to left click, and switching to right click.
    canvas.bind("<Button-3>",switchIt)
    canvas.bind("<Button-1>",selectCoordinates)

    root.mainloop()
    return (sources, sinks, File, im.size)