import sys, os
import numpy as np
import scipy.stats as st
from collections import Counter
from contextlib import contextmanager
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import Precision, Recall

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise as tfp_computer_noise
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SingleAttackResult
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia

"""
Performance evaluation
"""

"""
Metrics for model evaluation (already needed at compile step)
"""
def getMetrics(ds_info):
    if len(ds_info['class_counts']) == 2:
        metrics = [
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            F1Score(num_classes=1, threshold=0.5)
        ]
    elif len(ds_info['class_counts']) > 2:
        metrics = [ # uses classification_report later
            'accuracy'
        ]

    return metrics

"""
Overall performance evaluation workflow
"""
def evaluateModelPerformance(model, dataset):
    if len(dataset.info['class_counts']) > 2:
        labels_eval = []
        for x, y in dataset.test_attack_data.as_numpy_iterator():
            labels_eval.append(y[0])

    performance_results = []
    print("Evaluating %s ..." % (model.name))
    eval_res = model.evaluate(dataset.test_batched, verbose=2)
    print('\n')

    performance_result =  {
            'loss': eval_res[0],
            'accuracy': eval_res[1],
    }

    if len(dataset.info['class_counts']) == 2:
        performance_result.update({
            'precision': eval_res[2],
            'recall': eval_res[3],
            'f1-score': eval_res[4][0],
        })
    elif len(dataset.info['class_counts']) > 2:
        y_pred = model.predict(dataset.test_attack_data)
        y_pred = tf.argmax(y_pred, axis=1)
        report_dictionary = classification_report(labels_eval, y_pred, output_dict=True)
        print(classification_report(labels_eval, y_pred, output_dict=False, digits=3))

        performance_result.update({
            'precision': report_dictionary['macro avg']['precision'],
            'recall': report_dictionary['macro avg']['recall'],
            'f1-score': report_dictionary['macro avg']['f1-score'],
        })

    return performance_result

"""
Privacy evaluation
"""

"""
Calculate noise for given training hyperparameters
"""
def compute_noise(n, batch_size, target_epsilon, epochs, delta, min_noise=1e-5):
    return tfp_computer_noise(n, batch_size, target_epsilon, epochs, delta, min_noise)

"""
Calculate Delta for given training dataset size n
"""
def compute_delta(n):
    # delta should be one magnitude lower than inverse of training set size: 1/n
    # e.g. 1e-5 for n=60.000
    # take 1e-x, were x is the magnitude of training set size
    delta = np.power(10, - float(len(str(n)))) # remove all trailing decimals
    return delta

"""
Construct MIA inputs
"""
def compute_attack_inputs(model, attack_data, ds_info):
    # labels
    labels = []
    for x, y in attack_data.as_numpy_iterator():
        labels.append(y[0])
    # predictions
    probs = model.predict(attack_data)
    # losses
    from_logits = False
    constant = tf.keras.backend.constant
    if len(ds_info['class_counts']) == 2:
        bc = tf.keras.backend.binary_crossentropy
        losses = np.array([x[0] for x in bc(constant([[y] for y in labels]), constant(probs), from_logits=from_logits).numpy()])
    elif len(ds_info['class_counts']) > 2:
        scc = tf.keras.backend.sparse_categorical_crossentropy
        losses = scc(constant([[y] for y in labels]), constant(probs), from_logits=from_logits).numpy()
    
    return (np.array(probs), np.array(losses), np.array(labels))

"""
Single MIA experiment
"""
def run_mia(model, train_attack_input, test_attack_input):
    # prepare attacks
    probs_train, loss_train, labels_train = train_attack_input
    probs_test, loss_test, labels_test = test_attack_input

    attack_input = AttackInputData(
        # logits_train = logits_train,
        # logits_test = logits_test,
        probs_train = probs_train,
        probs_test = probs_test,
        # loss_train = loss_train,
        # loss_test = loss_test,
        labels_train = labels_train,
        labels_test = labels_test
    )

    slicing_spec = SlicingSpec( # only evaluate on complete dataset
        entire_dataset = True,
        by_class = False,
        by_percentiles = False,
        by_classification_correctness = False
    )

    attack_types = [
        AttackType.LOGISTIC_REGRESSION,
        AttackType.MULTI_LAYERED_PERCEPTRON,
        # AttackType.RANDOM_FOREST, # error because of tree model deprciation
        AttackType.K_NEAREST_NEIGHBORS,
        AttackType.THRESHOLD_ATTACK,
        #AttackType.THRESHOLD_ENTROPY_ATTACK, # error because incompatible formatting for entropy
    ] 

    # run several attacks for different data slices
    attack_results = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=attack_types)

    # plot the ROC curve of the best classifier
    #plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve)
    #plt.show()

    # print a user-friendly summary of the attacks
    #print(attacks_result.summary(by_slices=False))
    
    # get best auc, attacker advantage, and positive predictive value
    max_auc = attack_results.get_result_with_max_auc().get_auc()
    max_adv = attack_results.get_result_with_max_attacker_advantage().get_attacker_advantage()
    max_ppv = attack_results.get_result_with_max_ppv().get_ppv()
    # collect best working attack types for attacker advantage
    adv_atk_type = str(attack_results.get_result_with_max_attacker_advantage().attack_type)

    return max_auc, max_adv, max_ppv, adv_atk_type

"""
Method for print suppression
"""
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

"""
Overall privacy evaluation workflow
"""
def evaluateModelPrivacy(model, dataset, setting):
    # Calculate spent pivacy loss epsilon
    eps, order = compute_dp_sgd_privacy(
        dataset.info['train_count'],
        setting.batch_size,
        setting.noise,
        setting.epochs,
        setting.delta
    )
    #print(eps, order)
    #print('Achieved epsilon = %.3f with delta = %f for noise scale = %.3f'%(eps, setting.delta, setting.noise))
    print('\n')

    print('Membership Inference Attacks on '+model.name+'...')
    train_attack_input = compute_attack_inputs(model, dataset.train_attack_data, dataset.info)
    test_attack_input = compute_attack_inputs(model, dataset.test_attack_data, dataset.info)

    aucs, advs, ppvs = [], [], []
    adv_atk_types = []
    for i in range(setting.mia_samplenb):
        with suppress_stdout():
            max_auc, max_adv, max_ppv, adv_atk_type = run_mia(model, train_attack_input, test_attack_input)
        aucs.append(max_auc)
        advs.append(max_adv)
        ppvs.append(max_ppv)
        adv_atk_types.append(adv_atk_type)
        if (i+1) % 10 == 0:
            print('MIA %i/%i...'%(i+1, setting.mia_samplenb))

    auc_low, auc_high = st.t.interval(0.95, len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs))
    adv_low, adv_high = st.t.interval(0.95, len(advs)-1, loc=np.mean(advs), scale=st.sem(advs))
    ppv_low, ppv_high = st.t.interval(0.95, len(ppvs)-1, loc=np.mean(ppvs), scale=st.sem(ppvs))
    auc_max, adv_max, ppv_max = np.max(aucs), np.max(advs), np.max(ppvs)
    
    print('\n')
    print('95%%-CI based on %i attack samples'%(setting.mia_samplenb))
    print('AUC: %0.2f-%0.2f , max: %0.2f'%(auc_low, auc_high, auc_max))
    print('Attacker advantage: %0.2f-%0.2f , max: %0.2f'%(adv_low, adv_high, adv_max))
    adv_mean = np.mean([adv_low*100, adv_high*100])
    adv_diff = adv_high*100 - adv_mean
    print('Attacker advantage: %2.1f+-%2.1f'%(adv_mean, adv_diff))
    print('Attack types having max advantage: ' + str(Counter(adv_atk_types)))
    print('Positive predictive value: %0.2f-%0.2f , max: %0.2f'%(ppv_low, ppv_high, ppv_max))
    print('\n')

    privacy_result = {
        'aucs': aucs,
        'auc95': (auc_low, auc_high),
        'auc_max': auc_max,
        'advs': advs,
        'adv95': (adv_low, adv_high),
        'adv95_perc': (adv_mean, adv_diff),
        'adv_max': adv_max,
        'adv_atk_types': Counter(adv_atk_types),
        'ppvs': ppvs,
        'ppvs95': (ppv_low, ppv_high),
        'ppv_max': ppv_max,
    }

    return privacy_result, eps





