#!/bin/bash
pip install torch==2.0.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
pip install torch-geometric
pip install numpy>=1.24.1 pandas==2.0.3 scikit-learn==1.3.0 Flask==2.3.3 flask-cors==4.0.0 gunicorn==21.2.0 python-dateutil==2.8.2 joblib==1.2.0
