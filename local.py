#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:15:15 2019

@author: Harshvardhan
"""
from entry_point import main
from pathlib import Path

if __name__ == '__main__':
    location = Path(__file__).stem
    main(location)
