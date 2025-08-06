import numpy as np
import pandas as pd
import json
import itertools
from tqdm.auto import tqdm
import os
import glob

INDEX = '2023-08-03_03-44-08_ALLANSWER'
ROOT = '/scratch/generalvision/mowentao/ScanQA/outputs'
path = os.path.join(ROOT, INDEX, 'best_val_pred_answers*')

dfs = []
for p in tqdm(glob.glob(path)):
    df = pd.read_csv(p)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

for col in df.columns:
    if "pred_answer_scores" in col:
        df[col] = df[col].apply(lambda s: np.fromstring(s[1:-1], sep=","))

def create_id(row):
    return f"{row['scene_id']}-{row['question_id']}"
df["id"] = df.apply(create_id, axis=1)
df.set_index("id", inplace=True)

dset = pd.read_json("/scratch/generalvision/ScanQA-feature/qa/ScanQA_v1.0_val.json")
dset["question_id"] = dset["question_id"].apply(lambda s: int(s.split("-")[-1]))
dset["id"] = dset.apply(create_id, axis=1)
dset.set_index("id", inplace=True)

print(df.head(), df.shape)
print(dset.head(), dset.shape)
print(df.iloc[0], dset.iloc[0])

# df = df.merge(dset, on="question_id", how="left")
answer_counter = json.load(open(os.path.join(ROOT, INDEX, "answer_vocab.json")))
answer_counter = list(answer_counter)
atoi = {a: i for i, a in enumerate(answer_counter)}
itoa = {i: a for i, a in enumerate(answer_counter)} 

# for row in df.head(10).itertuples():
#     print(row.pred_answer_scores[:20])
#     print(row.pred_answer, row.pred_answer_idx, atoi[row.pred_answer], np.argmax(row.pred_answer_scores), len(row.pred_answer_scores))

# dset["answer_idx"] = dset["answer"].apply(lambda a: atoi[a])

# exit(0)

def get_answer(df):
    # get answer from dset
    def get_answer_from_dset(row):
        answers = dset.loc[row.name]["answers"]
        return answers
    
    df["answers"] = df.apply(get_answer_from_dset, axis=1)
    
    df["answer_idxs"] = df["answers"].apply(lambda answers: [atoi[a] for a in answers])

get_answer(df)

# df["correct"] = df.apply(lambda row: row["pred_answer"] in row["answers"], axis=1)
# print(df["correct"].mean())

df["pred_answer_scores_add"] = df["pred_answer_scores_2d"] + df["pred_answer_scores_scene"]
df["pred_answer_scores_expadd"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"]) + np.exp(row["pred_answer_scores_scene"]), axis=1)
df["pred_answer_scores_expadd2+1"] = df.apply(lambda row: 2 * np.exp(row["pred_answer_scores_2d"]) + np.exp(row["pred_answer_scores_scene"]), axis=1)
df["pred_answer_scores_expadd3+1"] = df.apply(lambda row: 3 * np.exp(row["pred_answer_scores_2d"]) + np.exp(row["pred_answer_scores_scene"]), axis=1)
df["pred_answer_scores_expadd1+2"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"]) + 2 * np.exp(row["pred_answer_scores_scene"]), axis=1)
# df["pred_answer_scores_expadd**2"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"]) ** 2 + np.exp(row["pred_answer_scores_scene"]) ** 2, axis=1)
# df["pred_answer_scores_expadd**0.5"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"]) ** 2 + np.exp(row["pred_answer_scores_scene"]) ** 2, axis=1)
df["pred_answer_scores_expadd/10"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] / 10)  + np.exp(row["pred_answer_scores_scene"] / 10), axis=1)
df["pred_answer_scores_expadd*2"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] * 2) + np.exp(row["pred_answer_scores_scene"] * 2), axis=1)
df["pred_answer_scores_expadd/2"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] / 2) + np.exp(row["pred_answer_scores_scene"] / 2), axis=1)
df["pred_answer_scores_expadd*2.5"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] * 2.5) + np.exp(row["pred_answer_scores_scene"] * 2.5), axis=1)
# df["pred_answer_scores_expadd*1.5"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] * 1.5) + np.exp(row["pred_answer_scores_scene"] * 1.5), axis=1)
# for i in np.linspace(0.5, 2, 20):
#     i = round(i, 2)
#     df[f"pred_answer_scores_expadd*{i}"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] * i) + np.exp(row["pred_answer_scores_scene"] * i), axis=1)




def cal_acc(df, score_name):
    print("Using score:", score_name)
    def get_pred(score):
        return np.argmax(score)

    df["pred_answer"] = df[score_name].apply(get_pred)
    
    df["correct"] = df.apply(lambda row: row["pred_answer"] in row["answer_idxs"], axis=1)
    return df["correct"].mean()
print("-" * 60)
for col in df.columns:
    if "pred_answer_scores" in col:
        print(col, cal_acc(df, col))
# print(cal_acc(df, "pred_answer_scores"))
# print(cal_acc(df, "pred_answer_scores_2d"))
# print(cal_acc(df, "pred_answer_scores_scene"))
# print(df.iloc[0])
# data = {}
# N = 20
# js = np.linspace(0.5, 2, N)
# # insert 1.0
# js = np.insert(js, 0, 1.0)
# js = np.insert(js, 0, 1.5)
# # resort 
# js = np.sort(js)
# for i, j in tqdm(itertools.product(js, js), total=(N+2)**2):
#     i = round(i, 3)
#     j = round(j, 3)
#     df[f"pred_answer_scores_expadd*{i}+{j}"] = df.apply(lambda row: np.exp(row["pred_answer_scores_2d"] * i) + np.exp(row["pred_answer_scores_scene"] * j), axis=1)
#     acc = cal_acc(df, f"pred_answer_scores_expadd*{i}+{j}")
#     print(f"pred_answer_scores_expadd*{i}+{j}: {acc}", )
#     data[(i, j)] = acc
# print(data)
# data = pd.DataFrame(list(data.items()), columns=['coords', 'value'])
# print(data)
# data[['2d', '3d']] = pd.DataFrame(data['coords'].tolist(), index=data.index)
# data.to_csv("heatmap.csv")
# data = data.pivot('3d', '2d', 'value')
# # save to file
# # find max value and its coordinates
# max_value = data.max().max()
# max_value_coords = data.unstack().index[data.unstack() == max_value].tolist()[0]
# print(f"Max value: {max_value} at {max_value_coords}")

# # plot heatmap
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 20))
# # sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu")
# sns.heatmap(data, cmap='viridis', annot=True, fmt=".4f")
# plt.show()
# plt.savefig("heatmap.png")
