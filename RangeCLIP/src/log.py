import os
import torch
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log

def log_input_settings(log_path,
                       n_batch=None,
                       n_height=None,
                       n_width=None):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    log('', log_path)

def log_network_settings(log_path,
                         # Network settings
                         model_architecture,
                         # Weight settings
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    log('Network settings:', log_path)
    log('model_architecture={}'.format(model_architecture),
        log_path)

    log('Weight settings:', log_path)
    log('n_parameter={}'.format(n_parameter),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          scheduler_type):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('scheduler type: ' + str(scheduler_type), log_path)
    log('scheduler: {}, learning_schedule=[{}]'.format(
        scheduler_type,
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates))),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_losses={},
                           w_weight_decay=None):

    w_losses_text = ''
    for idx, (key, value) in enumerate(w_losses.items()):

        if idx > 0 and idx % 3 == 0:
            w_losses_text = w_losses_text + '\n'

        w_losses_text = w_losses_text + '{}={:.1e}'.format(key, value) + '  '

    log('Loss function settings:', log_path)
    if len(w_losses_text) > 0:
        log(w_losses_text, log_path)

    if w_weight_decay is not None:
        log('w_weight_decay={:.1e}'.format(
            w_weight_decay),
            log_path)

    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_sample_per_summary=None,
                        validation_start_step=None,
                        restore_path_model=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_step_per_checkpoint), log_path)

        if validation_start_step is not None:
            log('validation_start_step={}'.format(validation_start_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'log_summary_frequency={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_sample_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_sample_per_summary={}'
        summary_settings_vars.append(n_sample_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_path_model is not None and restore_path_model != '':
        log('restore_path={}'.format(restore_path_model),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)