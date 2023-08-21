import os
import sys

# Add the path of the parent directory (neuroTB) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Import the run_neuroTB function from run.main
from run.main import run_neuroTB

# Set the relative path of the config file
config_filepath = '..neuroTB/temp/08-21/135406/config'

# Call the run_neuroTB function with the config_filepath
run_neuroTB(config_filepath)
