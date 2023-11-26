import os
import pandas as pd


pathgm = "../resources/gm_eng"
pathdu = "../resources/gm_rus"

gmList = list()
gmids = list()
duList = list()
duids = list()

for filename in os.listdir(pathgm):
    gmList.append(filename)
    gmids.append(1)

for filename in os.listdir(pathdu):
    duList.append(filename)
    duids.append(0)

# otherList = list()
# otherIds = list()
# otherpath = "../resources/wallpapers"
#
# for filename in os.listdir(otherpath):
#     otherList.append(filename)
#     otherIds.append(2)


df = pd.DataFrame({'name': gmList + duList, 'class': gmids+duids})
df = df.sample(frac=1)
df.to_csv("../resources/annotations.csv", index=False)
print(df)