# Write the amended function here
# Open the text file
def count_words(filepath, words_list):
    with open(filepath) as file:
        text = file.read()

    n = 0
    for word in text.split():
        # Count the number of times the words in the list appear
        if word.lower() in words_list:
            n += 1

    print('Lewis Carroll uses the word "cat" {} times'.format(n))
    return n
##################
from textanalysis.textanalysis import count_words

# Count the number of positive words
nb_positive_words = count_words('hotel-reviews.txt' , ['good' , 'great'])

# Count the number of negative words
nb_negative_words = count_words('hotel-reviews.txt' , ['bad' , 'awful'])

print("{} positive words.".format(nb_positive_words))
print("{} negative words.".format(nb_negative_words))
########################
pyment -w -o numpydoc impyrial/length/core.py
###########
INCHES_PER_FOOT = 12.0  # 12 inches in a foot
INCHES_PER_YARD = INCHES_PER_FOOT * 3.0  # 3 feet in a yard

UNITS = ("in", "ft", "yd")


def inches_to_feet(x, reverse=False):
    """Convert lengths between inches and feet.

    Parameters
    ----------
    x : numpy.ndarray
        Lengths in feet.
    reverse : bool, optional
        If true this function converts from feet to inches 
        instead of the default behavior of inches to feet. 
        (Default value = False)

    Returns
    -------
    numpy.ndarray
    """
    if reverse:
        return x * INCHES_PER_FOOT
    else:
        return x / INCHES_PER_FOOT
###########################
"""
Conversions between inches and 
larger imperial length units
"""
INCHES_PER_FOOT = 12.0  # 12 inches in a foot
INCHES_PER_YARD = INCHES_PER_FOOT * 3.0  # 3 feet in a yard

UNITS = ("in", "ft", "yd")


def inches_to_feet(x, reverse=False):
    """Convert lengths between inches and feet.

    Parameters
    ----------
    x : numpy.ndarray
        Lengths in feet.
    reverse : bool, optional
        If true this function converts from feet to inches 
        instead of the default behavior of inches to feet. 
        (Default value = False)

    Returns
    -------
    numpy.ndarray
    """
    if reverse:
        return x * INCHES_PER_FOOT
    else:
        return x / INCHES_PER_FOOT
#############################
"""impyrial
========
A package for converting between imperial 
measurements of length and weight.
"""
####################
"""
impyrial.length
===============
Length conversion between imperial units.
"""
#################
"""User-facing functions."""
from impyrial.length.core import (
    inches_to_feet , 
    inches_to_yards ,
    UNITS
)

def convert_unit(x, from_unit, to_unit):
    """Convert from one length unit to another.

    Parameters
    ----------
    x : array_like
        Lengths to convert.
    from_unit : {'in', 'ft', 'yd'}
        Unit of the input lengths `x`
    to_unit : {'in', 'ft', 'yd'}
        Unit of the returned lengths

    Returns
    -------
    ndarray
        An array of converted lengths with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    # Convert length to inches
    if from_unit == "in":
        inches = x
    elif from_unit == "ft":
        inches = inches_to_feet(x, reverse=True)
    elif from_unit == "yd":
        inches = inches_to_yards(x, reverse=True)

    # Convert inches to desired units
    if to_unit == "in":
        value = inches
    elif to_unit == "ft":
        value = inches_to_feet(inches)
    elif to_unit == "yd":
        value = inches_to_yards(inches)

    return value
######################
from impyrial.length.api import convert_unit

result = convert_unit(10, 'in', 'yd')
print(result)
#######################
"""User-facing functions."""
# Import the check_units function
from impyrial.utils import check_units
from impyrial.length.core import (
    UNITS,
    inches_to_feet,
    inches_to_yards,
)


def convert_unit(x, from_unit, to_unit):
    """Convert from one length unit to another.

    Parameters
    ----------
    x : array_like
        Lengths to convert.
    from_unit : {'in', 'ft', 'yd'}
        Unit of the input lengths `x`
    to_unit : {'in', 'ft', 'yd'}
        Unit of the returned lengths

    Returns
    -------
    ndarray
        An array of converted lengths with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    # Check if units are valid length units
    check_units(from_unit, to_unit, UNITS)
    
    # convert length to inches
    if from_unit == "in":
        inches = x
    elif from_unit == "ft":
        inches = inches_to_feet(x, reverse=True)
    elif from_unit == "yd":
        inches = inches_to_yards(x, reverse=True)

    # Convert inches to desired units
    if to_unit == "in":
        value = inches
    elif to_unit == "ft":
        value = inches_to_feet(inches)
    elif to_unit == "yd":
        value = inches_to_yards(inches)

    return value

############################
# This is the top level __init__.py file

# Import the length subpackage
from . import length
###################
# This is the __init__.py file for the impyrial/length subpackage

# Import the convert_unit function from the api.py module
from ..api import convert_unit
##########################
# Import required functions
from setuptools import setup , find_packages

# Call setup function
setup(
    author = "NEGM_ELG3AN" ,
    description = "A package for converting impyrial lengths and weights." ,
    name = "impyrial" ,
    packages = find_packages( include = ['impyrial' , 'impyrial.*'] ) ,
    version = "0.1.0" ,
)
#################
pip install -e .
###########################
"""Conversions between ounces and larger imperial weight units"""
OUNCES_PER_POUND = 16.0  # 16 ounces in a pound
OUNCES_PER_STONE = OUNCES_PER_POUND * 14.0  # 14 pounds in a stone

UNITS = ("oz", "lb", "st")


def ounces_to_pounds(x, reverse=False):
    """Convert weights between ounces and pounds.

    Parameters
    ----------
    x : array_like
        Weights in pounds.
    reverse : bool, optional
        If this is set to true this function converts from pounds to ounces
        instead of the default behaviour of ounces to pounds.

    Returns
    -------
    ndarray
        An array of converted weights with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    if reverse:
        return x * OUNCES_PER_POUND
    else:
        return x / OUNCES_PER_POUND


def ounces_to_stone(x, reverse=False):
    """Convert weights between ounces and stone.

    Parameters
    ----------
    x : array_like
        Weights in stone.
    reverse : bool, optional
        If this is set to true this function converts from stone to ounces
        instead of the default behaviour of ounces to stone.

    Returns
    -------
    ndarray
        An array of converted weights with the same shape as `x`. If `x` is a
        0-d array, then a scalar is returned.
    """
    if reverse:
        return x * OUNCES_PER_STONE
    else:
        return x / OUNCES_PER_STONE
