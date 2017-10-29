import subprocess
import os, sys


def load_names_from_folders(path):
    names = os.listdir(path)
    return names

scannet_path ='/projects/world3d/2017-06-scannet/'

folders = load_names_from_folders(scannet_path)
for folder in folders:
	cmd = 'find '+scannet_path+folder+' -name "*.mat"'
	subprocess.call(cmd, shell=True)
