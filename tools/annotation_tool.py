import PySimpleGUI as sg
# import PySimpleGUIQt as sg
import os.path
import PIL.Image
import io
import base64
from shutil import copyfile

"""
    Demo for displaying any format of image file.
    
    Normally tkinter only wants PNG and GIF files.  This program uses PIL to convert files
    such as jpg files into a PNG format so that tkinter can use it.
    
    The key to the program is the function "convert_to_bytes" which takes a filename or a 
    bytes object and converts (with optional resize) into a PNG formatted bytes object that
    can then be passed to an Image Element's update method.  This function can also optionally
    resize the image.
    
    Copyright 2020 PySimpleGUI.org
"""

def click(element):
    element.Widget.config(relief = "sunken")
    window.refresh()
    sleep(0.1)
    element.Widget.config(relief = "raised")
    window.refresh()
    element.click()


def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()



# --------------------------------- Define Layout ---------------------------------

# First the window layout...2 columns

left_col = [[sg.Text('Source Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER Source-'), sg.FolderBrowse()],
            [sg.Text('Dest Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER Dest-'), sg.FolderBrowse()],
            [sg.Button('Next')],
            [sg.Button('Save')],
            [sg.Button('Next Folder')],
            [sg.Button('Next 10 Folder')]]

# For now will only show the name of the file that was chosen
images_col = [[sg.Text('You choose from the list:')],
              [sg.Text(size=(80,1), key='-TOUT-')],
              [sg.Image(key='-IMAGE-')],
              [sg.Image(key='-MASK-')]]

# ----- Full layout -----
layout = [[sg.Column(left_col, element_justification='c'), sg.VSeperator(),sg.Column(images_col, element_justification='c')]]

# --------------------------------- Create Window ---------------------------------
window = sg.Window('Multiple Format Image Viewer', layout, resizable=True, finalize=True)

window.bind('<Key-F3>', 'F3')
window.bind('<Key-F5>', 'F5')
next = window['Next']
save = window['Save']


# ----- Run the Event Loop -----
# --------------------------------- Event Loop ---------------------------------
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if event == '-FOLDER Source-':                         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER Source-']
        try:
            file_list = sorted(os.listdir(folder))         # get list of files in folder
            print(file_list)
        except:
            file_list = []
        index_folder = 0
        index_image = 0

    elif event == '-FOLDER Dest-':                         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER Dest-']
        print(folder)

    elif event == 'Next':    # A file was chosen from the listbox

        subdir = file_list[index_folder]
        image_name = '%06d-color.jpg' % index_image

        filename = os.path.join(values['-FOLDER Source-'], subdir, image_name)
        save_image = filename
        window['-TOUT-'].update(filename)
        window['-IMAGE-'].update(data=convert_to_bytes(filename))

        # mask
        image_name = '%06d-label-binary.png' % index_image
        filename = os.path.join(values['-FOLDER Source-'], subdir, image_name)
        window['-MASK-'].update(data=convert_to_bytes(filename))

        # update index
        index_image += 1
        if index_image >= 9:
            index_folder += 1
            index_image = 0

    elif event == 'Next Folder':    # A file was chosen from the listbox

        # update index
        index_folder += 1
        index_image = 0

        subdir = file_list[index_folder]
        image_name = '%06d-color.jpg' % index_image

        filename = os.path.join(values['-FOLDER Source-'], subdir, image_name)
        save_image = filename
        window['-TOUT-'].update(filename)
        window['-IMAGE-'].update(data=convert_to_bytes(filename))

        # mask
        image_name = '%06d-label-binary.png' % index_image
        filename = os.path.join(values['-FOLDER Source-'], subdir, image_name)
        window['-MASK-'].update(data=convert_to_bytes(filename))

    elif event == 'Next 10 Folder':    # A file was chosen from the listbox

        # update index
        index_folder += 10
        index_image = 0

        subdir = file_list[index_folder]
        image_name = '%06d-color.jpg' % index_image

        filename = os.path.join(values['-FOLDER Source-'], subdir, image_name)
        save_image = filename
        window['-TOUT-'].update(filename)
        window['-IMAGE-'].update(data=convert_to_bytes(filename))

        # mask
        image_name = '%06d-label-binary.png' % index_image
        filename = os.path.join(values['-FOLDER Source-'], subdir, image_name)
        window['-MASK-'].update(data=convert_to_bytes(filename))

    elif event == 'Save':
        print(save_image)
        filename = save_image.replace(values['-FOLDER Source-'], values['-FOLDER Dest-'])
        head, tail = os.path.split(filename)
        print(filename)
        if not os.path.exists(head):
           os.makedirs(head)

        # color
        copyfile(save_image, filename)

        # depth
        save_image = save_image.replace('color', 'depth')
        save_image = save_image.replace('jpg', 'png')
        filename = filename.replace('color', 'depth')
        filename = filename.replace('jpg', 'png')
        copyfile(save_image, filename)

        # mask
        save_image = save_image.replace('depth', 'label-binary')
        filename = filename.replace('depth', 'label-binary')
        copyfile(save_image, filename)

        # meta
        save_image = save_image.replace('label-binary', 'meta')
        save_image = save_image.replace('png', 'mat')
        filename = filename.replace('label-binary', 'meta')
        filename = filename.replace('png', 'mat')
        copyfile(save_image, filename)

    elif event == 'F3':
        # click(ok)
        next.click()
    elif event == 'F5':
        # click(cancel)
        save.click()


# --------------------------------- Close & Exit ---------------------------------
window.close()
