import os
import tensorflow as tf
from mia_covid.models import resnet18, resnet50
from mia_covid.settings import run_settings, privacy, commons
from mia_covid.evaluation import compute_delta, compute_noise, evaluateModelPerformance, evaluateModelPrivacy
from mia_covid.training import compileModel, trainModel
from mia_covid.models import model_settings, resnet_builder
from mia_covid.datasets import Covid19RadiographyDataset, MnistDataset
from mia_covid.utils import set_random_seeds
from mia_covid.abstract_dataset_handler import AbstractDataset
from contextlib import nullcontext
from typing import Dict, Any, Optional, Callable


'''
Experiment workflow
'''


def runExperiment(arg_dict: Dict["str", Any]):
    print(f"Tensorflow Version: {tf.version.VERSION}")

    # Bool to activate and deactivate syncing to wandb
    # wandb setup uses environment variables
    USE_WANDB = arg_dict['wandb']
    if USE_WANDB:
        import wandb
        from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
        WANDB_API_KEY = os.environ['WANDB_API_KEY']
        WANDB_ENTITY = os.environ['WANDB_ENTITY']
        WANDB_PROJECT = os.environ['WANDB_PROJECT']
        wandb.login(key=WANDB_API_KEY)

    # Distribute work over all available gpus using MirroredStrategy
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices for training: {}\n".format(strategy.num_replicas_in_sync))

    model_setting: model_settings = None
    preprocessing_func: Optional[Callable[[float], tf.Tensor]] = None
    if arg_dict["model"] == "resnet18":
        model_setting = resnet18
        # TODO: keras does not come provide a resnet18 preprocessing input function -> we should maybe write our own?
    elif arg_dict["model"] == "resnet50":
        model_setting = resnet50
        preprocessing_func = tf.keras.applications.resnet50.preprocess_input

    dataset: AbstractDataset = None
    if arg_dict["dataset"] == "mnist":
        dataset = MnistDataset(model_img_shape=(28, 28, 3), builds_ds_info=True, batch_size=model_setting.batch_size, preprocessing_func=preprocessing_func)
    elif arg_dict["dataset"] == "covid":
        dataset = Covid19RadiographyDataset(model_img_shape=(224, 224, 3), dataset_path="data", builds_ds_info=True, batch_size=model_setting.batch_size, preprocessing_func=preprocessing_func)

    settings = run_settings(dataset=dataset, model_setting=model_setting, commons=commons(), privacy=privacy(target_eps=arg_dict["eps"]))

    if settings.privacy.target_eps and settings.dataset.dataset_name == "covid":  # dpsgd covid needs reduced batch size
        settings.model_setting.batch_size = settings.model_setting.batch_size_private_covid

    # increase global batch size by number of available distribution gpus
    settings.model_setting.batch_size = settings.model_setting.batch_size * strategy.num_replicas_in_sync

    print('Dataset: {}, Model: {}, Epsilon: {}'.format(settings.dataset.dataset_name, settings.model_setting.architecture, settings.privacy.target_eps))
    set_random_seeds(settings.commons.random_seed)

    # Load data into dataset here
    dataset.load_dataset()
    dataset.prepare_datasets()

    print(f"\nTotal images: {dataset.ds_info['total_count']}")
    print(f"Classes: {len(dataset.ds_info['class_counts'])} {dataset.ds_info['class_counts']}")
    print(f"Class weights: {dataset.ds_info['class_weights']}")
    print(f"Train: {dataset.ds_info['train_count']}, Val: {dataset.ds_info['val_count']}, Test: {dataset.ds_info['test_count']}\n")

    # Update delta and noise in privacy settings
    settings.privacy.delta = compute_delta(dataset.ds_info['train_count'])

    if settings.privacy.target_eps:
        settings.privacy.noise = compute_noise(
            dataset.ds_info['train_count'],
            settings.model_setting.batch_size,
            settings.privacy.target_eps,
            settings.commons.epochs,
            settings.privacy.delta,
        )
        print('\n')

    if dataset.variants:
        print('\nSelected variants:')
        print(f"- {settings.model_setting.architecture}")
        print(*[f"-- {variant['activation']}" + f" {variant['pretraining']}" for variant in dataset.variants], sep='\n')
        print('\n')

        # Runs - for each model variant create one run with own wandb init
        histories, performance_results, privacy_results = [], [], []
        for variant in dataset.variants:  # final perfomance results roundup
            set_random_seeds(settings.commons.random_seed)
            # Use distributed scope if multiple gpus - scope only needed in model creation and compilation
            with strategy.scope() if strategy.num_replicas_in_sync > 1 else nullcontext():
                # Get model and compile it #TODO lowercase layer names in models for saving
                model, model_info = resnet_builder(architecture=settings.model_setting.architecture,
                                                   activation=variant['activation'],
                                                   pretraining=variant['pretraining'],
                                                   weights_path=settings.commons.weights_path,
                                                   classes=len(dataset.ds_info['class_counts']),
                                                   img_shape=dataset.model_img_shape)
                model = compileModel(model, dataset, settings)

        # Weights and biases initial setup
        if USE_WANDB:
            l2_clip = settings.commons.l2_clip if settings.privacy.target_eps else None
            if len(settings.dataset.ds_info['class_counts']) == 2:
                performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score']
            elif len(settings.dataset.ds_info['class_counts']) > 2:
                performance_metrics = ['accuracy', 'precision (macro avg)', 'recall (macro avg)', 'f1-score (macro avg)']
            privacy_metrics = ['auc', 'attacker advantage (adv)', 'positive predictive value (ppv)']
            run_name = f"{settings.dataset.dataset_name}_eps-{settings.privacy.target_eps}_{model_info['model_name']}" if settings.privacy.target_eps else settings.dataset.dataset_name + '_eps-inf_' + model_info['model_name']

            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                config={
                    'dataset': settings.dataset.dataset_name,
                    'dataset_info': settings.dataset.ds_info,
                    'target_eps': settings.privacy.target_eps,
                    **model_info,
                    'random_seed': settings.commons.random_seed,
                    'epochs': settings.commons.epochs,
                    'learning_rate': settings.commons.learning_rate,
                    'mia_samplenb': settings.commons.mia_samplenb,
                    'batch_size': settings.model_setting.batch_size,
                    'delta': settings.privacy.delta,
                    'noise': settings.privacy.noise,
                    'l2_clip': l2_clip,
                    'performance_metrics': performance_metrics,
                    'privacy_metrics': privacy_metrics,
                })

        # Training with weights and biases loggers and save final weights as artifact
        callbacks = [WandbMetricsLogger(),
                     # WandbModelCheckpoint("wandb/artifacts/"+run_name, save_weights_only=True, save_freq=setting.epochs) #TODO save final weights
                     ] if USE_WANDB else []
        model, history = trainModel(model, dataset, settings, callbacks)
        histories.append(history)

        # Performance evaluation
        performance_result = evaluateModelPerformance(model, dataset)
        performance_results.append(performance_result)

        # Privacy evaluation
        privacy_result, _ = evaluateModelPrivacy(model, dataset, settings)
        privacy_results.append(privacy_result)

        # Weights and biases final logging
        if USE_WANDB:
            wandb.log({
                **performance_result,
                **privacy_result,
            })
            wandb.finish()

            # release RAM for next model run
            tf.keras.backend.clear_session()

        # final results roundup
        for variant, performance_result, privacy_result in zip(settings.dataset.variants, performance_results, privacy_results):  # final perfomance results roundup
            print(f"{settings.model_setting.architecture} {variant['activation']} {variant['pretraining']}")
            print(performance_result)
            print(privacy_result)
            print('\n')
