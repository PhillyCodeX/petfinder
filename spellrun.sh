#!/usr/bin/env bash
spell run -m uploads/petfinder_train:petfinder_train -t K80x8 --pip-req configs/requirements.txt "python main.py"
