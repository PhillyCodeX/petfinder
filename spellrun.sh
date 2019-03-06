#!/usr/bin/env bash
spell run -m uploads/petfinder:petfinder -t cpu --pip-req configs/requirements.txt "python src/submission_script.py ../petfinder/"
