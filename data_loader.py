import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
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

dataset_electricity = get_dataset("electricity_nips", regenerate=True, prediction_length=23)


from gluonts.dataset.util import to_pandas
entry = next(iter(dataset_electricity.train))
train_series = to_pandas(entry)

entry = next(iter(dataset_electricity.test))
test_series = to_pandas(entry)

print('test_series.loc[test_series.index > train_series.index[-1]].shape', test_series.loc[test_series.index > train_series.index[-1]].shape)