#!/usr/bin/bash

set -xe

python3 mainB_var.py explicit sq/6

python3 mainB_var.py implicit lin
python3 mainB_var.py implicit sq
python3 mainB_var.py implicit sq/6

python3 mainB_var.py nicolson lin
python3 mainB_var.py nicolson sq
python3 mainB_var.py nicolson sq/6

