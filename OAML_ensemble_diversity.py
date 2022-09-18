# Application script for automated River

import numpy as np
import pandas as pd
import arff
import sys

from gama import GamaClassifier
from gama.search_methods import AsyncEA
from gama.search_methods import RandomSearch
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.postprocessing import BestFitOnlinePostProcessing
from gama.utilities.ensemble import VotingPipeline

from river import metrics
from river.drift import EDDM
from river import evaluate
from river import stream
from river import ensemble

from skmultiflow import drift_detection
import matplotlib.pyplot as plt
import wandb

# Metrics
gama_metrics = {
  "acc": 'accuracy',
  "b_acc": "balanced_accuracy",
  "f1": "f1",
  "roc_auc": "roc_auc",
  "rmse": "rmse"
}

online_metrics = {
    "acc":      metrics.Accuracy(),
    "b_acc":    metrics.BalancedAccuracy(),
    "f1":       metrics.F1(),
    "roc_auc":  metrics.ROCAUC(),
    "rmse":     metrics.RMSE()
}

# Search algorithms
search_algs = {
    "random":       RandomSearch(),
    "evol":         AsyncEA(),
    "s_halving":    AsynchronousSuccessiveHalving()
}

# User parameters
print(sys.argv[0])                                                          # prints python_script.py
print(f"Data stream is {sys.argv[1]}.")                                     # prints dataset no
print(f"Initial batch size is {int(sys.argv[2])}.")                         # prints initial batch size
print(f"Sliding window size is {int(sys.argv[3])}.")                        # prints sliding window size
print(f"Gama performance metric is {gama_metrics[str(sys.argv[4])]}.")                         # prints gama performance metric
print(f"Online performance metric is {online_metrics[str(sys.argv[5])]}.")                       # prints online performance metric
print(f"Time budget for GAMA is {int(sys.argv[6])}.")                       # prints time budget for GAMA
print(f"Search algorithm for GAMA is {search_algs[str(sys.argv[7])]}.")                       # prints search algorithm for GAMA
print(f"Live plotting (wandb) is {eval(sys.argv[8])}.")  #
print(f"Search with diversity is {eval(sys.argv[9])}, diversity weight is {float(sys.argv[10])}")

data_loc = sys.argv[1]                              #needs to be arff
initial_batch = int(sys.argv[2])                    #initial set of samples to train automl
sliding_window = int(sys.argv[3])                   #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
gama_metric = gama_metrics[str(sys.argv[4])]        #gama metric to evaluate in pipeline search
online_metric  = online_metrics[str(sys.argv[5])]   #river metric to evaluate online learning
time_budget = int(sys.argv[6])                      #time budget for gama run
search_alg = search_algs[str(sys.argv[7])]
live_plot = eval(sys.argv[8])
diversity = eval(sys.argv[9])
diversity_phi = float(sys.argv[10])
drift_detector = drift_detection.EDDM()             #multiflow drift detector
#drift_detector = EDDM()                            #river drift detector - issues

# Plot initialization
if live_plot:
    wandb.init(
        project="Ensemble - demo",
        entity = "autoriver",
        config={
            "dataset": data_loc,
            "batch_size": initial_batch,
            "sliding_window": sliding_window,
            "gama_performance_metric": gama_metric,
            "online_performance_metric": online_metric,
            "time_budget_gama": time_budget,
            "search_algorithm": search_alg
        })

# Data
B = pd.DataFrame(arff.load(open(data_loc, 'r'), encode_nominal=True)["data"])

# Preprocessing of data: Drop NaNs, check for zero values
if pd.isnull(B.iloc[:, :]).any().any():
    print("Data X contains NaN values. The rows that contain NaN values will be dropped.")
    B.dropna(inplace=True)
if B[:].iloc[:,0:-1].eq(0).any().any():
    print("Data contains zero values. They are not removed but might cause issues with some River learners.")
for u in B.columns:
    if B[u].dtype == bool:
        B[u] = B[u].astype('int')

X = B[:].iloc[:,0:-1]
y = B[:].iloc[:,-1]

# Algorithm selection and hyperparameter tuning
Auto_pipeline = GamaClassifier(max_total_time=time_budget,
                               scoring=gama_metric,
                               search=search_alg,
                               online_learning=True,
                               post_processing=BestFitOnlinePostProcessing(),
                               store='nothing',
                               diversity=diversity,
                               diversity_phi=diversity_phi,
                               )

Auto_pipeline.fit(X.iloc[0:initial_batch], y[0:initial_batch])
print(f'Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}')
print("Online model is updated with latest AutoML pipeline.")

# Online learning
Online_model = VotingPipeline([Auto_pipeline.model])
exist_pipelines = [Auto_pipeline.model]

record_df = pd.DataFrame(columns=['idx', 'acc', 'ensemble_sim_cos'])

cos_sim_ensemble = 1.0
dis = 0.0
rho = 0.0
Q = 0.0
kappa = 0.0
cos_sim = 0.0

count_drift = 0
last_training_point = initial_batch
print(f'Test batch - 0 with 0')
for i in range(initial_batch+1, len(B)):
    # Test then train - by one
    y_pred = Online_model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y.iloc[i], y_pred)
    Online_model = Online_model.learn_one(X.iloc[i].to_dict(), int(y.iloc[i]))

    if i % 1000 == 0:
        print(f'Test batch - {i} with {online_metric}')
        record_df.loc[len(record_df.index)] = [
            i, online_metric.get(), cos_sim_ensemble
        ]
        record_df.to_csv(f'record_diversity_ensemble5_HYPER_{diversity}_{diversity_phi}_1.csv')
        if live_plot:
            wandb.log({"current_point": i, "Prequential performance": online_metric.get()})

    # Check for drift
    drift_detector.add_element(int(y_pred != y[i]))
    if (drift_detector.detected_change()) or ((i - last_training_point) > 20000):
        if i - last_training_point < 5000:
            continue
        if drift_detector.detected_change():
            print(f"Change detected at data point {i} and current performance is at {online_metric}")
            if live_plot:
                wandb.log({"drift_point": i, "current_point": i, "Prequential performance": online_metric.get()})
        if (i - last_training_point) > 20000:
            print(f"No drift but start a new pipeline searching at point {i} and current performance is at {online_metric}")
            if live_plot:
                wandb.log({"drift_point": i, "current_point": i, "Prequential performance": online_metric.get()})
        last_training_point = i

        # Sliding window at the time of drift
        X_sliding = X.iloc[(i-sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i-sliding_window):i].reset_index(drop=True)

        # Drop the pipeline if ensemble exceeding 5
        if len(Online_model.models) == 5:
            Online_model.models.pop(0)
            exist_pipelines.pop(0)

        # re-optimize pipelines with sliding window
        search_alg.clear_history()
        Auto_pipeline = GamaClassifier(max_total_time=time_budget,
                                       scoring=gama_metric,
                                       search=search_alg,
                                       online_learning=True,
                                       post_processing=BestFitOnlinePostProcessing(),
                                       store='nothing',
                                       diversity=diversity,
                                       exist_pipelines=exist_pipelines,
                                       diversity_phi=diversity_phi
                                       )
        Auto_pipeline.fit(X_sliding, y_sliding)
        new_model = Auto_pipeline.model

        # Calculate the accuracy and diversity between new pipeline and existing pipelines
        print("Ensemble already include {} pipelines.".format(len(exist_pipelines)))
        X_sample = X_sliding.iloc[-sliding_window//10:]
        y_sample = y_sliding.iloc[-sliding_window//10:]

        # cos_sim = 0.0
        # cos_sim_lis = []
        # cos_cnt = 0
        # n_11 = 0
        # n_10 = 0
        # n_01 = 0
        # n_00 = 0
        # for idx in range(len(X_sample)):
        #     x_i = X_sample.iloc[idx].to_dict()
        #     y_i = int(y_sample.iloc[idx])
        #     try:
        #         y_p1 = new_model.predict_proba_one(x_i)
        #         y_p2 = Online_model.predict_proba_one(x_i)
        #         v1 = np.array(list(y_p1.values()))
        #         v2 = np.array(list(y_p2.values()))
        #         numer = np.sum(v1 * v2)
        #         denom = np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2))
        #         cos_dis = numer / denom
        #         cos_sim_lis.append(cos_dis)
        #         if cos_dis < 0.95:
        #             cos_sim += cos_dis
        #             cos_cnt += 1
        #         y_p1 = new_model.predict_one(x_i)
        #         y_p2 = Online_model.predict_one(x_i)
        #         if y_p1 == y_p2 and y_p1 == y_i:
        #             n_11 += 1
        #         elif y_p1 == y_p2 and y_p1 != y_i:
        #             n_00 += 1
        #         elif y_p1 != y_p2 and y_p1 == y_i:
        #             n_10 += 1
        #         elif y_p1 != y_p2 and y_p1 != y_i:
        #             n_01 += 1
        #     except Exception as e:
        #         print(e)
        # # disagreement measure
        # dis = 1.0 * (n_01 + n_10) / len(X_sample)
        # # correlation coefficient
        # if (1.0 * (n_11 + n_10) * (n_11 + n_01) * (n_01 + n_00) * (n_10 + n_00)) ** 0.5 != 0:
        #     rho = (1.0 * n_11 * n_00 - 1.0 * n_01 * n_10) / (
        #                 (1.0 * (n_11 + n_10) * (n_11 + n_01) * (n_01 + n_00) * (n_10 + n_00)) ** 0.5)
        # # Q
        # if (n_11 * n_00 + n_10 * n_01) != 0:
        #     Q = 1.0 * (n_11 * n_00 - n_10 * n_01) / (n_11 * n_00 + n_10 * n_01)
        # # kappa
        # p1 = 1.0 * (n_11 + n_00) / len(X_sample)
        # p2 = (1.0 * (n_11 + n_10) * (n_11 + n_01) + 1.0 * (n_01 * n_00) * (n_10 * n_00)) / (
        #             (1.0 * len(X_sample)) ** 2)
        # kappa = (p1 - p2) / (1 - p2)
        # # Cosine similarity
        # if cos_cnt is not 0:
        #     cos_sim /= cos_cnt
        # else:
        #     cos_sim = 1.0
        #
        # cos_sim_percentage = [0 for _ in range(20)]
        # for sim in cos_sim_lis:
        #     for j in range(20):
        #         if 0.05 * j < sim < 0.05 * (j + 1):
        #             cos_sim_percentage[j] += 1
        # for j in range(20):
        #     print(f"Cosine similarity [{0.05*j}, {0.05*(j+1)}] account for {cos_sim_percentage[j]/len(cos_sim_lis)*100}%")
        #
        # print(f"Cosine similarity = {cos_sim}")
        # print(f"Disagreement measure = {dis}")
        # print(f"Correlation coefficient = {rho}")
        # print(f"Q statistics = {Q}")
        # print(f"Kappa = {kappa}")
        #

        # Ensemble update with new model
        Online_model.models.append(Auto_pipeline.model)
        exist_pipelines.append(Auto_pipeline.model)

        print(f'Current model is {Online_model} and hyperparameters are: {Online_model._get_params()}')

        # Calculate the similarity of ensemble
        cos_sim_ensemble = 0.0
        pre_result = [[] for _ in range(len(Online_model.models))]
        for idx_m, model in enumerate(Online_model.models):
            for idx in range(len(X_sample)):
                x_i = X_sample.iloc[idx].to_dict()
                y_i = int(y_sample.iloc[idx])
                y_p = model.predict_proba_one(x_i)
                v = np.array(list(y_p.values()))
                pre_result[idx_m].append(v)
        cos_cnt = 0
        for j in range(len(Online_model.models) - 1):
            for k in range(j+1, len(Online_model.models)):
                for idx in range(len(X_sample)):
                    v1 = pre_result[j][idx]
                    v2 = pre_result[k][idx]
                    numer = np.sum(v1 * v2)
                    denom = np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2))
                    cos_dis = numer / denom
                    if cos_dis < 0.95:
                        cos_sim_ensemble += cos_dis
                        cos_cnt += 1
        if cos_cnt is not 0:
            cos_sim_ensemble = cos_sim_ensemble / cos_cnt
        else:
            cos_sim_ensemble = 1.0

        print(f'Current ensemble cosine similarity is {cos_sim_ensemble}')

        record_df.loc[len(record_df.index)] = [
            i, online_metric.get(), cos_sim_ensemble
        ]
        record_df.to_csv(f'record_diversity_ensemble5_HYPER_{diversity}_{diversity_phi}_1.csv')

