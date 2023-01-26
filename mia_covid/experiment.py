from mia_covid import settings
from mia_covid.utils import merge_dataclasses, set_random_seeds
from mia_covid.datasets import Dataset
from mia_covid.models import resnet_builder
from mia_covid.training import compileModel, trainModel
from mia_covid.evaluation import compute_delta, compute_noise, evaluateModelPerformance, evaluateModelPrivacy

import os
import tensorflow as tf
from dataclasses import dataclass
from contextlib import nullcontext


'''
Experiment workflow
'''
def runExperiment(inputs):
    # Bool to activate and deactivate syncing to wandb
    # wandb setup uses environment variables
    USE_WANDB = inputs['wandb']
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

    # Settings
    @dataclass(eq=True, frozen=False)
    class privacy():
        target_eps: float = inputs['eps']
        delta: float = None # placeholder to set in model compilation
        noise: float = 0.0 # update in model compilation if private

    setting = merge_dataclasses(
        getattr(settings, 'commons')(),
        getattr(settings, inputs['dataset'])(),
        getattr(settings, inputs['model'])(),
        privacy()
    )

    if setting.target_eps and setting.dataset_name == 'covid': # dpsgd covid needs reduced batch size
        setting.batch_size = setting.batch_size_private_covid
    # increase global batch size by number of available distribution gpus
    setting.batch_size = setting.batch_size * strategy.num_replicas_in_sync

    #print(setting.__dict__)
    
    print('Dataset: {}, Model: {}, Epsilon: {}'.format(setting.dataset_name, setting.architecture, setting.target_eps))
    set_random_seeds(setting.random_seed)

    # Data
    dataset = Dataset(setting)

    print('\nTotal images: {}'.format(dataset.info['total_count']))
    print('Classes: {} {}'.format(len(dataset.info['class_counts']), dataset.info['class_counts']))
    print('Class weights: {}'.format(dataset.info['class_weights']))
    print('Train: {}, Val: {}, Test: {}\n'.format(dataset.info['train_count'], dataset.info['val_count'], dataset.info['test_count']))

    # Update delta and noise in privacy settings
    setting.delta = compute_delta(dataset.info['train_count'])

    if setting.target_eps:
        setting.noise = compute_noise(
            dataset.info['train_count'],
            setting.batch_size,
            setting.target_eps,
            setting.epochs,
            setting.delta,
        )
        print('\n')
    
    print('\nSelected variants:')
    print('- ' + setting.architecture)
    print(*['-- ' + variant['activation'] + ' ' + str(variant['pretraining']) for variant in setting.variants], sep = '\n')
    print('\n')

    # Runs - for each model variant create one run with own wandb init
    histories, performance_results, privacy_results = [], [], []
    for variant in setting.variants: # final perfomance results roundup
        set_random_seeds(setting.random_seed)
        # Use distributed scope if multiple gpus - scope only needed in model creation and compilation
        with strategy.scope() if strategy.num_replicas_in_sync > 1 else nullcontext():
            # Get model and compile it #TODO lowercase layer names in models for saving
            model, model_info = resnet_builder(architecture=setting.architecture,
                                               activation=variant['activation'],
                                               pretraining=variant['pretraining'],
                                               weights_path=setting.weights_path,
                                               classes=len(dataset.info['class_counts']),
                                               img_shape=setting.img_shape)
            model = compileModel(model, dataset, setting)

        # Weights and biases initial setup
        if USE_WANDB:
            l2_clip = setting.l2_clip if setting.target_eps else None
            if len(dataset.info['class_counts']) == 2:
                performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score']
            elif len(dataset.info['class_counts']) > 2:
                performance_metrics = ['accuracy', 'precision (macro avg)', 'recall (macro avg)', 'f1-score (macro avg)']
            privacy_metrics = ['auc', 'attacker advantage (adv)', 'positive predictive value (ppv)']
            run_name = setting.dataset_name+'_eps-'+str(setting.target_eps)+'_'+model_info['model_name'] if setting.target_eps else setting.dataset_name+'_eps-inf_'+model_info['model_name']

            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                config = {
                    'dataset': setting.dataset_name,
                    'dataset_info': dataset.info,
                    'target_eps': setting.target_eps,
                    **model_info,
                    'random_seed': setting.random_seed,
                    'epochs': setting.epochs,
                    'learning_rate': setting.learning_rate,
                    'mia_samplenb': setting.mia_samplenb,
                    'batch_size': setting.batch_size,
                    'delta': setting.delta,
                    'noise': setting.noise,
                    'l2_clip': l2_clip,
                    'performance_metrics': performance_metrics,
                    'privacy_metrics': privacy_metrics,
                })

        # Training with weights and biases loggers and save final weights as artifact
        callbacks = [WandbMetricsLogger(),
                     # WandbModelCheckpoint("wandb/artifacts/"+run_name, save_weights_only=True, save_freq=setting.epochs) #TODO save final weights
                    ] if USE_WANDB else []
        model, history = trainModel(model, dataset, setting, callbacks)
        histories.append(history)

        # Performance evaluation
        performance_result = evaluateModelPerformance(model, dataset)
        performance_results.append(performance_result)

        # Privacy evaluation
        privacy_result, _ = evaluateModelPrivacy(model, dataset, setting)
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
    for variant, performance_result, privacy_result in zip(setting.variants, performance_results, privacy_results): # final perfomance results roundup
        print(setting.architecture + ' ' + variant['activation'] + ' ' + str(variant['pretraining']))
        print(performance_result)
        print(privacy_result)
        print('\n')

