#!/usr/bin/env bash
spell run -m petfinder:petfinder -t cpu-huge --pip-req requirements.txt "python src/submission_script.py ./petfinder"
