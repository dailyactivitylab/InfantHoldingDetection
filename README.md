# Infant holding Detection
This repository contains code to run an infant holding detection model for continuous acceleration data in naturalistic environments, as described in the following paper.

## Citation Information
X. Yao, T. Plötz, M. Johnson, and K. de Barbaro. 2019. Automated Detection of Infant Holding Using Wearable Sensing: Implications for Developmental Science and Intervention. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 3, 2, Article 64 (June 2019), 17 pages. https://doi.org/10.1145/3328935


## Models and Main Package Versions
Trained random forest model can be found at: https://utexas.box.com/shared/static/epbzxav40e25orn3c2liie2e62mo39sn.pkl

### Versions
python3/3.6.3  
scikit-learn==0.18.1   


# Code
There are two scripts: *get_features.py*, and *predict.py*

get_features.py obtains features from synchronized mother baby data streams into one single file final.csv

Input format example: (time in second, acc value) in css file
0.1 23.12
0.2 43.12

predict.py predicts crying at seconds level, then smooth the predictions, then make the predictions into minute level.

RUN get_features.py, then predict.py

