import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
from mlflow.sklearn import log_model
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset


import optuna

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
)

# извлекаю датасет
data_fish = pd.read_csv(r"Fish Dataset for Clustering\fish_data.csv")

# кодирую таргет для кластеризации
labelEncoder = LabelEncoder()

data_fish["labels_encoder"] = labelEncoder.fit_transform(data_fish["species"])

# нормализую данные. В ходе эксперемента это дает большрй прирост ARI
scaler = StandardScaler()

data_fish[["length", "weight", "w_l_ratio"]] = scaler.fit_transform(
    data_fish[["length", "weight", "w_l_ratio"]]
)


# подключение к нужному эксперементу и задание рана
experiment_name = "Fish dataset clustering"
run_name = "AgglomerativeClustering StandardScaler"

mlflow.set_experiment(experiment_name)
mlflow.set_tracking_uri("http://localhost:5000")


def objective(trial):
    with mlflow.start_run(nested=True):

        # категориальные параметры
        linkage = trial.suggest_categorical(
            "linkage",
            ["ward", "complete", "average", "single"],
        )

        params = {
            "linkage": linkage,
        }

        # обучение кластеризации
        ac = AgglomerativeClustering(
            n_clusters=len(labelEncoder.classes_),
            linkage=params["linkage"],
        )

        ac.fit(data_fish[["length", "weight", "w_l_ratio"]])

        data_fish["labels"] = ac.labels_

        # подготовливаю датасет для загрузки в mlflow
        dataset: PandasDataset = mlflow.data.pandas_dataset.from_pandas(
            data_fish,
            targets="labels_encoder",
            name="fish_data",
            predictions="labels",
        )

        mlflow.log_input(dataset=dataset, context="training")

        # cчитаю метрики
        RI = rand_score(data_fish["labels_encoder"], data_fish["labels"])
        ARI = adjusted_rand_score(data_fish["labels_encoder"], data_fish["labels"])

        AMI = adjusted_mutual_info_score(
            data_fish["labels_encoder"], data_fish["labels"]
        )
        NMI = normalized_mutual_info_score(
            data_fish["labels_encoder"], data_fish["labels"]
        )

        homogeneity = homogeneity_score(
            data_fish["labels_encoder"], data_fish["labels"]
        )

        completeness = completeness_score(
            data_fish["labels_encoder"], data_fish["labels"]
        )
        v = v_measure_score(data_fish["labels_encoder"], data_fish["labels"])

        FMI = fowlkes_mallows_score(data_fish["labels_encoder"], data_fish["labels"])

        # сохраняем модель в mlflow
        signature = infer_signature(
            data_fish[["length", "weight", "w_l_ratio"]], data_fish["labels"], params
        )
        log_model(ac, run_name, signature=signature)

        # логирую параметры в mlflow
        mlflow.log_params(params)

        # логирую метрики в mlflow
        mlflow.log_metric("RI", RI)
        mlflow.log_metric("ARI", ARI)

        mlflow.log_metric("AMI", AMI)
        mlflow.log_metric("NMI", NMI)

        mlflow.log_metric("homogeneity", homogeneity)
        mlflow.log_metric("completeness", completeness)
        mlflow.log_metric("v", v)

        mlflow.log_metric("FMI", FMI)

    return ARI


with mlflow.start_run(run_name=run_name):

    # Создание основных графиков
    fish_dataset_pairplot = sns.pairplot(
        data=data_fish[["length", "weight", "w_l_ratio", "species"]], hue="species"
    ).figure
    mlflow.log_figure(fish_dataset_pairplot, "fish_dataset_pairplot.png")

    plt.clf()

    matrix = data_fish[["length", "weight", "w_l_ratio", "labels_encoder"]].corr()
    fish_dataset_heatmap = sns.heatmap(matrix, cmap="Greens", annot=True).figure
    mlflow.log_figure(fish_dataset_heatmap, "fish_dataset_heatmap.png")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_ARI = study.best_value

    best_ac = AgglomerativeClustering(
        n_clusters=len(labelEncoder.classes_),
        linkage=best_params["linkage"],
    )

    best_ac.fit(data_fish[["length", "weight", "w_l_ratio"]])

    data_fish["labels"] = best_ac.labels_

    # сохраняю график кластеризации
    fish_dataset_predict_pairplot = sns.pairplot(
        data=data_fish[["length", "weight", "w_l_ratio", "labels"]],
        hue="labels",
        palette="husl",
    ).figure

    mlflow.log_figure(
        fish_dataset_predict_pairplot, "fish_dataset_predict_pairplot.png"
    )

    mlflow.log_params(best_params)
    mlflow.log_metric("best_ARI", best_ARI)
