import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pts7.metrics import get_metrics
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from pts.model.time_grad import TimeGradEstimator
from pytorch_lightning import Trainer
from diffusers import DEISMultistepScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_electricity = get_dataset("electricity_nips", regenerate=False, prediction_length=15)
train_grouper_electricity = MultivariateGrouper(max_target_dim=min(2000, int(dataset_electricity.metadata.feat_static_cat[0].cardinality)))

test_grouper_electricity = MultivariateGrouper(num_test_dates=int(len(dataset_electricity.test)/len(dataset_electricity.train)), 
                                   max_target_dim=min(2000, int(dataset_electricity.metadata.feat_static_cat[0].cardinality)))


dataset_train_electricity = train_grouper_electricity(dataset_electricity.train)
dataset_test_electricity = test_grouper_electricity(dataset_electricity.test)

scheduler = DEISMultistepScheduler(
    num_train_timesteps=150,
    beta_end=0.1,
)
estimator = TimeGradEstimator(
    input_size=int(dataset_electricity.metadata.feat_static_cat[0].cardinality),
    hidden_size=64,
    num_layers=2,
    dropout_rate=0.1,
    lags_seq=[1],
    scheduler=scheduler,
    num_inference_steps=149,
    prediction_length=192,
    context_length=96, # input_length
    freq=dataset_electricity.metadata.freq,
    scaling="mean",
    trainer_kwargs=dict(max_epochs=1, accelerator="gpu", devices="1"),
)

# transformation = estimator.create_transformation()
# trainnet = estimator.create_lightning_module()
# trainnet.load_state_dict(torch.load('/vast/s224075134/timeseries/TimeGrad-re/pts7/lightning_logs/version_3/checkpoints/epoch=18-step=950.ckpt')['state_dict'])
# predictor=estimator.create_predictor(transformation, trainnet)
predictor = estimator.train(dataset_train_electricity, num_workers=8)


forecast_it_electricity, ts_it_electricity = make_evaluation_predictions(dataset=dataset_test_electricity,
                                                 predictor=predictor,
                                                 num_samples=50)

forecasts_electricity = list(forecast_it_electricity)
targets_electricity = list(ts_it_electricity)

evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})
targets = []
preds = []
for i in range(len(targets_electricity)):
    targets.append(torch.tensor(np.atleast_1d(np.squeeze(targets_electricity[i].loc[forecasts_electricity[i].index]))))
    preds.append(torch.tensor(forecasts_electricity[i].samples))

targets = torch.stack(targets)
preds = torch.stack(preds)

# metrics 1
metrics = get_metrics(preds, targets)
print(metrics)

# metrics 2
agg_metric_electricity, item_metrics_electricity = evaluator(targets_electricity, forecasts_electricity, num_series=len(dataset_test_electricity))
print("Electricity Results")
print("CRPS:", agg_metric_electricity["mean_wQuantileLoss"])
print("ND:", agg_metric_electricity["ND"])
print("NRMSE:", agg_metric_electricity["NRMSE"])
print("")
print("CRPS-Sum:", agg_metric_electricity["m_sum_mean_wQuantileLoss"])
print("ND-Sum:", agg_metric_electricity["m_sum_ND"])
print("NRMSE-Sum:", agg_metric_electricity["m_sum_NRMSE"])