# Test machine learning models

## Run
 `$ cd mltest/`  
 `$ docker-compose up -d --build`  
 `$ docker-compose exec mltest bash`  
 `$ python opt/main.py`  

## Result
|                  |  Normal  | CrossValidation |
| ---------------- | -------- | --------------- |
| LinearRegression | 0.575462 |     0.576007    |
|       SVM        | 0.497062 |     0.515925    |
|   RandomForest   | 0.162012 |     0.439354    |
|      GBoot       | 0.462840 |     0.474575    |
|       MLP        | 0.468229 |     0.494212    |

## Dataset
[California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)