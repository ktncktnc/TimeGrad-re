import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import argparse
import torch
import os
from datetime import datetime

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation import MultivariateEvaluator
from gluonts.dataset.split import split
from gluonts.evaluation.backtest import _to_dataframe
from gluonts.evaluation import MultivariateEvaluator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast

from pts.model.time_grad import TimeGradEstimator
from pts7.metrics import get_metrics
from typing import Tuple, Iterator

from diffusers import DEISMultistepScheduler



def parse_args():
    parser = argparse.ArgumentParser(description='TimeGrad forecasting with configurable parameters')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='electricity_nips', 
                        help='Dataset name from GluonTS repository')
    parser.add_argument('--prediction_length', type=int, default=48,
                        help='Prediction horizon length')
    parser.add_argument('--context_length', type=int, default=96,
                        help='Context/input sequence length')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the model')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--lags_seq', type=str, default='[1]',
                        help='Lags sequence as a string representation of a list')
    
    # Scheduler parameters
    parser.add_argument('--num_train_timesteps', type=int, default=150,
                        help='Number of timesteps for the scheduler')
    parser.add_argument('--beta_end', type=float, default=0.1,
                        help='Beta end value for the scheduler')
    parser.add_argument('--num_inference_steps', type=int, default=149,
                        help='Number of inference steps')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Maximum number of training epochs')
    parser.add_argument('--accelerator', type=str, default='gpu',
                        help='Accelerator type for PyTorch Lightning')
    parser.add_argument('--devices', type=str, default='1',
                        help='Devices for PyTorch Lightning')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples for evaluation')
    parser.add_argument('--scaling', type=str, default='mean',
                        help='Scaling method')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    
    # Logging and checkpointing
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='timeseries-forecasting/timegrad',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity name')
    parser.add_argument('--log_dir', type=str, default='lightning_logs',
                        help='Directory for logs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--save_top_k', type=int, default=3,
                        help='Number of best checkpoints to save')
    parser.add_argument('--monitor', type=str, default='train_loss',
                        help='Metric to monitor for checkpointing')
    parser.add_argument('--save_last', action='store_true',
                        help='Save the last checkpoint')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint file to load for testing (skips training)')
    
    return parser.parse_args()

def calculate_percentage_offset(dataset, test_percentage):
    """Calculate offset based on percentage of time series length"""
    # Get a sample time series to determine length
    sample = next(iter(dataset))
    series_length = sample["target"].shape[1]
    
    # Calculate offset (negative value as required by split function)
    offset = int(series_length * test_percentage / 100)
    
    return offset

def make_evaluation_predictions(
    dataset: Dataset,
    predictor: PyTorchPredictor,
    num_samples: int = 100,
) -> Tuple[Iterator[Forecast], Iterator[pd.Series]]:
    
    window_length = predictor.prediction_length + predictor.lead_time
    offset = calculate_percentage_offset(dataset, test_percentage=20)
    _, test_template = split(dataset, offset=-offset)
    test_data = test_template.generate_instances(
        prediction_length=window_length,
        windows=int((offset-window_length)/predictor.prediction_net.model.context_length), # number of samples
        distance=predictor.prediction_net.model.context_length # stride
    )
    print('int((offset-window_length)/predictor.prediction_net.model.context_length)', int((offset-window_length)/predictor.prediction_net.model.context_length))
    print('lentest_data', len(test_data))

    return (
        predictor.predict(test_data.input, num_samples=num_samples),
        map(_to_dataframe, test_data),
    )


def main():
    args = parse_args()
    
    # Parse lags_seq from string to list
    try:
        lags_seq = eval(args.lags_seq)
        if not isinstance(lags_seq, list):
            print(f"Warning: lags_seq '{args.lags_seq}' did not evaluate to a list. Using default [1].")
            lags_seq = [1]
    except:
        print(f"Warning: Could not parse lags_seq '{args.lags_seq}'. Using default [1].")
        lags_seq = [1]
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset, regenerate=False, prediction_length=args.prediction_length)
    
    # Create groupers
    max_target_dim = min(2000, int(dataset.metadata.feat_static_cat[0].cardinality))
    train_grouper = MultivariateGrouper(max_target_dim=max_target_dim)
    
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test)/len(dataset.train)),
        max_target_dim=max_target_dim
    )
    
    # Apply groupers
    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)
    
    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Test dataset size: {len(dataset_test)}")
    
    # Create scheduler
    scheduler = DEISMultistepScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_end=args.beta_end,
    )
    
    # Setup logging and checkpointing
    run_name = f"timgrad_{args.dataset}_{args.context_length}_{args.prediction_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, run_name),
        filename='{epoch}-{train_loss:.4f}',
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        save_last=args.save_last,
        mode='min',
    )
    
    # Setup trainer kwargs
    trainer_kwargs = {
        'max_epochs': args.max_epochs,
        'accelerator': args.accelerator,
        'devices': args.devices,
        #'callbacks': [checkpoint_callback],
        'default_root_dir': args.log_dir,
    }
    
    # Add WandB logger if enabled
    if args.wandb:
        try:
            # Initialize WandB logger
            wandb_logger = WandbLogger(
                name=run_name,
                project=args.wandb_project,
                entity=args.wandb_entity,
                log_model=True,
            )
            
            # Add config to WandB
            config = vars(args)
            wandb_logger.log_hyperparams(config)
            
            # Add WandB logger to trainer kwargs
            trainer_kwargs['logger'] = wandb_logger
            print("WandB logging enabled")
        except ImportError:
            print("Warning: wandb not installed. Running without WandB logging.")
            args.wandb = False
    
    # Create estimator
    estimator = TimeGradEstimator(
        input_size=int(dataset.metadata.feat_static_cat[0].cardinality),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        lags_seq=lags_seq,
        scheduler=scheduler,
        num_inference_steps=args.num_inference_steps,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        freq=dataset.metadata.freq,
        scaling=args.scaling,
        trainer_kwargs=trainer_kwargs,
    )
    
    if args.checkpoint:
        transformation = estimator.create_transformation()
        trainnet = estimator.create_lightning_module()
        trainnet.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        predictor=estimator.create_predictor(transformation, trainnet)
    else:
        # Train model
        print("Training model...")
        predictor = estimator.train(dataset_train, num_workers=args.num_workers)
    
    # Make predictions
    print("Making predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=args.num_samples
    )
    
    forecasts = list(forecast_it)
    targets = list(ts_it)

    print('len targets', len(targets))
    
    # Evaluate
    print("Evaluating predictions...")
    evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})
    
    # Prepare for custom metrics
    target_tensors = []
    pred_tensors = []
    for i in range(len(targets)):
        target_tensors.append(torch.tensor(np.atleast_1d(np.squeeze(targets[i].loc[forecasts[i].index]))))
        pred_tensors.append(torch.tensor(forecasts[i].samples))
    
    target_tensors = torch.stack(target_tensors)
    pred_tensors = torch.stack(pred_tensors)
    
    print('pred_tensors', pred_tensors.shape)
    print('target_tensors', target_tensors.shape)
    # Calculate metrics
    custom_metrics = get_metrics(pred_tensors, target_tensors)

    # Calculate GluonTS metrics
    agg_metric, item_metrics = evaluator(targets, forecasts, num_series=len(dataset_test))
    
    # Prepare results
    results = {
        "CRPS": agg_metric["mean_wQuantileLoss"],
        "ND": agg_metric["ND"],
        "NRMSE": agg_metric["NRMSE"],
        "CRPS-Sum": agg_metric["m_sum_mean_wQuantileLoss"],
        "ND-Sum": agg_metric["m_sum_ND"],
        "NRMSE-Sum": agg_metric["m_sum_NRMSE"]
    }
    
    print('pred_tensors', pred_tensors.shape)
    print('target_tensors', target_tensors.shape)

    print("\nCustom Metrics:")
    for key, value in custom_metrics.items():
        print(f"{key}: {value}")
    
    # Print results
    print(f"\n{args.dataset.capitalize()} Results:")
    print("CRPS:", results["CRPS"])
    print("ND:", results["ND"])
    print("NRMSE:", results["NRMSE"])
    print("")
    print("CRPS-Sum:", results["CRPS-Sum"])
    print("ND-Sum:", results["ND-Sum"])
    print("NRMSE-Sum:", results["NRMSE-Sum"])
    
    # Log to wandb if enabled
    if args.wandb and 'wandb_logger' in locals():
        wandb_logger.log_metrics(results)
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()