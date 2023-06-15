# -*- coding: utf-8 -*-

'''
This is a file for analysing custom floating point number system for hardware design.

List of check :
    1. Signed number.
    2. Subtract
    4. Overflow and underflow
    
Done.
    3. Array input.
'''
import numpy as np
import matplotlib.pyplot as plt

def FloatArithmetic(ExponentBit, MentisaBit, ExponentA, MentisaA, ExponentB, MentisaB):
    """

    Parameter
    ---------------------------------------------------------------------------
    

    """
    #Need sanity check 
    # Check dimension
    # Check ExponentBit
    # Check MentisaBit
    
    ExpComp = (ExponentA > ExponentB) + 0
    ExpCompInv = np.abs(ExpComp - 1)
    
    Exponent = ExpComp * ExponentA + ExpCompInv * ExponentB
    
    ExponentDiff = np.abs(ExponentA - ExponentB)
    
    MentisaA = (MentisaA + 2**MentisaBit) >> (ExpCompInv * ExponentDiff)
    MentisaB = (MentisaB + 2**MentisaBit) >> (ExpComp * ExponentDiff)
    
    Mentisa = MentisaA + MentisaB
    
    MentisaCheck = ((Mentisa >> (MentisaBit + 1)) == 1) + 0
    
    Exponent = Exponent + MentisaCheck
    Mentisa = (Mentisa >> MentisaCheck) % (2**MentisaBit)
    
    
    '''
    if ExponentA >= ExponentB:
        ExponentDiff = ExponentA - ExponentB
        MentisaB = (MentisaB + 2**MentisaBit) >> ExponentDiff
        MentisaA = (MentisaA + 2**MentisaBit)
        Exponent = ExponentA
    elif ExponentB > ExponentA:
        ExponentDiff = ExponentB - ExponentA
        MentisaA = (MentisaA + 2**MentisaBit) >> ExponentDiff
        MentisaB = (MentisaB + 2**MentisaBit)
        Exponent = ExponentB
    
    Mentisa = MentisaA + MentisaB
    
    if (Mentisa >> (MentisaBit + 1)) == 1:
        Exponent += 1
        Mentisa = (Mentisa >> 1) % (2**MentisaBit)
    else:
        Exponent = Exponent
        Mentisa = (Mentisa) % (2**MentisaBit)
    '''
    
    return [Exponent, Mentisa]

def FloatDecay(ExponentBit, MentisaBit, FloatNum):
    
    IntConv = FloatSplitToInt(FloatNum, ExponentBit, MentisaBit)
    
    ExponentTemp = IntConv[0]
    MentisaTemp = IntConv[1] - 1
    
    MentisaIndex = np.where(MentisaTemp < 0)[0]

    
    if (len(MentisaIndex) != 0):
        MentisaTemp[MentisaIndex] = 2**MentisaBit - 1
        
        ExponentTemp[MentisaIndex] = ExponentTemp[MentisaIndex] - 1
        
        ExponentIndex = np.where(ExponentTemp < 0)[0]
        
        if len(ExponentIndex) != 0:
            ExponentTemp[ExponentIndex] = 0
            MentisaTemp[ExponentIndex] = 0

    
    Exponent = ExponentTemp
    Mentisa = MentisaTemp
    
    return ToFloat(Exponent, Mentisa, MentisaBit)

def ToFloat(exponent, mentisa, mentisaBit):
    """
    Generate floating numbers.
    
    Parameters
    ---------------------------------------------------------------------------
    exponent : int, ndarray
               
    mentisa : int, ndarray
    
    mentisaBit : int
            
    
    Returns
    ---------------------------------------------------------------------------
    out : ndarray
    
    """
    # Convert integer input to an ndarray
    if type(exponent) != np.ndarray:
        exponent = np.array([exponent])
    
    if type(mentisa) != np.ndarray:
        mentisa = np.array([mentisa])
    
    expLen = len(exponent)
    menLen = len(mentisa)
    
    ## Sanity check
    if expLen == menLen:
        out = np.zeros(expLen)
    elif expLen != menLen:
        print(exponent, mentisa)
        print("Error: the number of exponent array and the number of metisa array is different.")
        
        return False
    
    if any(x >= 2**mentisaBit for x in mentisa):
        print("Error : the integer value of mentisa is out of range. Need to check mentisaBit.")
        return False
    
    # Caculate values
    zeroIndex = np.where(exponent + mentisa == 0)[0]
    
    # First calculate decimal point from 'mentisa'.
    DecimalPoint = 0
    for i in range(mentisaBit):
        DecimalPoint += ((mentisa >> i) % 2) * (1/(2**(mentisaBit - i)))
        
    out = (1 + DecimalPoint) * (2**exponent)
    
    if len(zeroIndex) > 0:
        out[zeroIndex] = 0
        
    return out

def FloatSplitToInt(Float, ExponentBit, MentisaBit):
    """
    
    """
    
    floatSplit = [str(x).split('.') for x in Float]
    
    a = np.array(floatSplit, dtype=np.int64).T[0]
    b = np.array(floatSplit, dtype=np.int64).T[1]
    bLen = np.array([len(x) for x in np.array(floatSplit).T[1]])
    
    Exponent = np.array([len("{0:b}".format(x))-1 for x in a])
    
    tempMentisa = np.zeros(len(Float))
    for i in range(MentisaBit):
        b = b * 2
        
        flag = (np.array((b / (10**bLen)), dtype=np.int64) == 1) + 0
        tempMentisa = tempMentisa + 2**(MentisaBit - i - 1) * flag
        b = b - (10**bLen) * flag
        
    Mentisa = (((a - 2**Exponent) << MentisaBit) + np.array(tempMentisa, dtype=np.int64)) >> Exponent
    
    # To remove negative mentisa when the float is smaller than 1.
    Mentisa[np.where(Mentisa < 0)[0]] = 0

    Error = Float - ToFloat(Exponent, Mentisa, MentisaBit)
    
    '''
    floatSplit = str(Float).split('.')
    
    a = int(floatSplit[0])
    b = int(floatSplit[1])
    bLen = len(floatSplit[1])
    
    Exponent = len("{0:b}".format(a)) - 1
    
    print(a, b, bLen, Exponent)
    tempMentisa = 0
    for i in range(MentisaBit):
        b = b * 2
        
        if int(b / (10**bLen)) == 1:
            tempMentisa += 2**(MentisaBit - i - 1)
        
            b -= (10**bLen)
            
    Mentisa = (((a - 2**Exponent) << MentisaBit) + tempMentisa) >> Exponent
    
    Error = Float - ToFloat(Exponent, Mentisa, MentisaBit)
    '''
    
    return [Exponent, Mentisa, Error]


def FloatSpacing(ExponentBit, MentisaBit, plot=False):
    """
    This generate all possible floating numbers can be expressed in given 
    floating point number system.
    
    Parameters
    ---------------------------------------------------------------------------
    ExponentBit : int
                  Bit budget of exponent in floating point number system.
    
    MentisaBit : int
                 Bit budget of mentisa in floating point number system.
    
    Returns
    ---------------------------------------------------------------------------
    out : ndarray
          An array of all floating point numbers that can be expressed in given
          floating point number system. 
          
    """

    i = np.repeat(np.linspace(0, 2**ExponentBit - 1, 2**ExponentBit, dtype=np.uint32), 2**MentisaBit)
    j = np.tile(np.linspace(0, 2**MentisaBit - 1, 2**MentisaBit, dtype=np.uint32), 2**ExponentBit)
    
    out = ToFloat(i, j, MentisaBit)
    
    if plot == True:
        plt.figure()
        plt.plot(out, 'b.')
        plt.show()

    return out