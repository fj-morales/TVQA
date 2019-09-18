import os
import shutil

# Create dir function
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def destroy_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path) 
        