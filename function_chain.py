#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:02:49 2019

@author: Harshvardhan
"""

from local_main import local_1, local_2
from remote_main import remote_1, remote_2


command_chain = {
    'local': {
        'start': local_1,
        'remote_1': local_2
    },
    'remote': {
        'local_1': remote_1,
        'local_2': remote_2
    }
}
