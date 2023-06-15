#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 15:47:50 2018

@Author: Jongkil
"""
import subprocess
import numpy as np
import datetime

synMax = 0.01
dlyMax = 0.1

NeuPLUSFolderName = "C:/work/neurosynaptic-comp/neuplus"

NeuPLUSParamFolder = NeuPLUSFolderName + "/Parameters/"
NeuPLUSSynFolder = NeuPLUSFolderName + "/Synapses/"
NeuPLUSResultFolder = NeuPLUSFolderName + "/RESULT/"

def GetDestFolderName(env='work'):
    
    if (env == 'work'):
        foldername =  "D:/work/CSharp/neuplusCsharp/neuplus/bin/Release"
    elif (env == 'home'):
        foldername = "C:/work/neuplusCSharp/neuplus/bin/Release/"
    
    return foldername


def NeuPLUSRun(*args):
    
    print("(parameters.py) If the program is not launched, the script should be open in an external terminal.")
    
    commands = [NeuPLUSFolderName+"/neuplus.exe"]
    
    for i in range(len(args)):
        commands.append(args[i])
        
    start_time = datetime.datetime.now()
    subprocess.call(commands)
    end_time = datetime.datetime.now()
    
    elapsed_time = end_time - start_time
    
    print("(parameters.py) : NeuPLUS run is over.")
    print(" Elapsed time for the experiment is ", elapsed_time, " seconds.")
    
    return elapsed_time

