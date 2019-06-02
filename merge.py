# https://www.kaggle.com/bojackchill/hongfuliu?scriptVersionId=7987764
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
result1=pd.read_csv("../input/finaldata/submission_adjusted.csv")
result2=pd.read_csv("../input/fantasticdata/submission_mlp.csv")
result3=pd.read_csv("../input/fantasticdata/submission_pytorch.csv")

final_result=pd.merge(result1,result3, on="Id")
final_result["winPlacePerc"]=(final_result["winPlacePerc_x"]+final_result["winPlacePerc_y"])/2
final_result.drop("winPlacePerc_x", axis=1,inplace=True)
final_result.drop("winPlacePerc_y", axis=1,inplace=True)
final_result.to_csv("submission_adjusted.csv",index=False)
