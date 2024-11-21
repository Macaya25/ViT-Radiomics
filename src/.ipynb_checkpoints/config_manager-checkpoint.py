# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:21:40 2024

@author: Mico
"""

import os
import yaml


def is_project_dir(path):
    return '.git' in os.listdir(path)

def get_project_dir():
    current_dir = os.getcwd()
    upper_dir = os.path.join(current_dir, '..')
    if is_project_dir(current_dir):
        project_dir = current_dir
    elif is_project_dir(upper_dir):
        project_dir = upper_dir
    else:
        dir_list = current_dir.split(os.sep)
        project_dir = os.sep.join(dir_list[0:dir_list.index('src')])
    print("porj:",project_dir)
    assert is_project_dir(project_dir)
    return project_dir

def load_all_ymls(config_folder, startswith='parameters'):
    config = {}
    for filename in os.listdir(config_folder):
        if filename.startswith(startswith):
            if filename.endswith(".yml") or filename.endswith(".yaml"):
                file_path = os.path.join(config_folder, filename)
                with open(file_path, "r") as file:
                    config_data = yaml.safe_load(file)
                    if config_data:
                        config.update(config_data)
    return config

def load_conf(startswith='parameters'):
    project_dir = ".."
    config_folder = os.path.join(project_dir, 'conf')
    config = load_all_ymls(config_folder, startswith)
    return config

if __name__ == "__main__":
    config = load_conf(startswith='parameters')
    print(config)
