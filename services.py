import os, shutil

def clear_result_folder():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

def create_result_folder_structure():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.mkdir(os.path.join(dir, 'saved_images'))
    os.mkdir(os.path.join(dir, 'saved_models'))