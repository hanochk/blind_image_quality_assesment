#!/bin/bash
source /home/hanoch/.local/share/virtualenvs/blind_quality_36_venv/bin/activate
if [ -z $1 ] ; then
  python -u /home/hanoch/GIT/blind_quality_svm/merge_csv_holdouts.py
else
  python -u /home/hanoch/GIT/blind_quality_svm/merge_csv_holdouts.py --path "$1"
fi
