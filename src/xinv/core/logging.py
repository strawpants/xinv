## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

# This file is part of the shxarray software which is licensed
# under the Apache License version 2.0 (see the LICENSE file in the main repository)
# Copyright Roelof Rietbroek (r.rietbroek@utwente.nl), 2023
#


import logging
# xinv wide logger
xinvlogger=logging.getLogger("xinv")

ch = logging.StreamHandler()

# create formatter
formatter = logging.Formatter('%(name)s-%(levelname)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
xinvlogger.addHandler(ch)


def debugging():
    return xinvlogger.getEffectiveLevel() == logging.DEBUG

def setInfoLevel():
    """Set logging level for both python and c++ to INFO severity"""
    xinvlogger.setLevel(logging.INFO)

def setDebugLevel():
    """Set logging level for both python and c++ to DEBUG severity"""
    xinvlogger.setLevel(logging.DEBUG)


def setWarningLevel():
    """Set logging level for both python and c++ to WARNING severity"""
    xinvlogger.setLevel(logging.WARNING)

def setErrorLevel():
    """Set logging level for both python and c++ to WARNING severity"""
    xinvlogger.setLevel(logging.ERROR)

setInfoLevel()
