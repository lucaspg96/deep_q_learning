#!/bin/bash
clear
play m.mp3 &
python2 pacman.py -p QAgent -n 10 --frameTime 0.05