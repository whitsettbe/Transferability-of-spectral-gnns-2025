#!/bin/bash
pip install 'Markdown<3.4' 'setuptools<=68.0.0' # could probably do Markdown<3.5
mamba install tensorboard==2.1.1
pip install tensorflow-gpu==2.1.0 tensorflow-estimator==2.1.0
mamba install ogb==1.2.3