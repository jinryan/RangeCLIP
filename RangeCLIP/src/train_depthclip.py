from .factory import create_loss

# Training loop example
def train_multimodal_encoder(model, depth_maps, images, texts, optimizer, epochs=10):
    
    model.train()
    loss_fn = create_loss()
    for epoch in range(epochs):
        epoch_loss = 0
        for depth_batch, image_batch, text_batch in zip(depth_maps, images, texts):
            optimizer.zero_grad()
            
            depth_embeddings, image_embeddings, text_embeddings = model(
                depth_batch, image_batch, text_batch
            )
            
            loss = loss_fn(
                depth_embeddings, 
                image_embeddings, 
                text_embeddings,
                logit_scale=10
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss / len(depth_maps)}")


import os, sys, time, tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, os.getcwd())
import datasets
from unet_denoising_model import UNetDenoisingModel
import utils.src.data_utils as data_utils
import utils.src.eval_utils as eval_utils
from utils.src.log_utils import log


def train(train_image_path,
          val_image_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          random_noise_type,
          random_noise_spread,
          # Network settings
          model_architecture,
          # Training settings
          learning_rates,
          learning_schedule,
          # Photometric data augmentations
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_hue,
          augmentation_random_saturation,
          # Geometric data augmentations
          augmentation_random_flip_type,
          # Loss function settings
          w_losses,
          w_weight_decay,
          # Checkpoint settings
          checkpoint_path,
          n_step_per_checkpoint,
          n_step_per_summary,
          n_image_per_summary,
          validation_start_step,
          restore_path_model,
          # Hardware settings
          device='cuda',
          n_thread=8):

    # TODO: Set device to torch.device based on cuda, gpu or cpu
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_checkpoint_path = os.path.join(
        checkpoint_path,
        'checkpoints',
        '{}'.format(model_architecture) + '-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'tensorboard')

    os.makedirs(event_path, exist_ok=True)
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty
    }

    '''
    Set up training dataloader
    '''
    # TODO: Read training input paths using data_utils.read_paths
    train_image_paths = data_utils.read_paths(train_image_path)

    # TODO: Get number of training examples based on paths
    n_train_sample = len(train_image_paths)

    # TODO: Compute the total number of training steps based on last element of
    # training_schedule and n_batch (assume we drop any incomplete batch)
    n_train_step = learning_schedule[-1] * (n_train_sample // n_batch)

    # TODO: Setup training dataloader using torch.utils.data.DataLoader and datasets.ImageDenoisingDataset
    # Set batch_size=n_batch, shuffle=True, num_workers=n_thread, drop_last=True
    train_dataset = datasets.ImageDenoisingDataset(image_paths=train_image_paths,
                                                   resize_shape=(n_height, n_width),
                                                   random_noise_type=random_noise_type,
                                                   random_noise_spread=random_noise_spread,
                                                   augmentation_random_brightness=augmentation_random_brightness,
                                                   augmentation_random_contrast=augmentation_random_contrast,
                                                   augmentation_random_hue=augmentation_random_hue,
                                                   augmentation_random_flip_type=augmentation_random_flip_type)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=n_batch,
                                                   shuffle=True,
                                                   num_workers=n_thread,
                                                   drop_last=True)

    '''
    Set up validation dataloader
    '''
    if val_image_path is not None:

        # TODO: Read validation input paths using data_utils.read_paths
        val_image_paths = data_utils.read_paths(val_image_path)

        # TODO: Setup validation dataloader using torch.utils.data.DataLoader and datasets.ImageDenoisingDataset
        # Set batch_size=1, shuffle=False, num_workers=1, drop_last=False, do not include augmentations
        
        val_dataset = datasets.ImageDenoisingDataset(image_paths=val_image_paths,
                                                   resize_shape=(n_height, n_width),
                                                   random_noise_type=random_noise_type,
                                                   random_noise_spread=random_noise_spread,
                                                   augmentation_random_brightness=augmentation_random_brightness,
                                                   augmentation_random_contrast=augmentation_random_contrast,
                                                   augmentation_random_hue=augmentation_random_hue,
                                                   augmentation_random_flip_type=augmentation_random_flip_type)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=n_batch,
                                                   shuffle=True,
                                                   num_workers=n_thread,
                                                   drop_last=True)

    '''
    Set up the model
    '''
    # TODO: Instantiate model
    model = UNetDenoisingModel(model_architecture, device)

    # TODO: Get model parameters
    parameters_model = model.parameters()

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)
    log(train_image_path, log_path)

    if val_image_path is not None:
        log('Validation input paths:', log_path)
        log(val_image_path, log_path)

    log_input_settings(
        log_path,
        # Batch settings
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        random_noise_type=random_noise_type,
        random_noise_spread=random_noise_spread)

    log_network_settings(
        log_path,
        # Network settings
        model_architecture=model_architecture,
        # Weight settings
        parameters_model=parameters_model)

    log_training_settings(
        log_path,
        # Training settings
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Photometric data augmentations
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_hue=augmentation_random_hue,
        augmentation_random_saturation=augmentation_random_saturation,
        # Geometric data augmentations
        augmentation_random_flip_type=augmentation_random_flip_type)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        w_losses=w_losses,
        w_weight_decay=w_weight_decay)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        n_image_per_summary=n_image_per_summary,
        validation_start_step=validation_start_step,
        restore_path_model=restore_path_model,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Train model
    '''
    # Set up optimizer with starting learning rate
    n_epoch = learning_schedule[-1]

    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    # TODO: Set up optimizer with parameters of model, weight_decay, and learning rate
    optimizer = torch.optim.Adam(parameters_model,
                                lr=learning_rate,
                                weight_decay=w_weight_decay)

    # TODO: Restore model from restore path
    if restore_path_model is not None:
        start_step, optimizer = model.restore_model(restore_path_model, optimizer=optimizer)

        # TODO: Set optimizer learning rate in optimizer param_groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    else:
        start_step = 0

    # Intialize train step to start step
    train_step = start_step

    # TODO: Set model to train mode
    model.train()

    # Check if we are using multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # TODO: If so, enable data parallel
        model.data_parallel()

    time_start = time.time()

    log('Begin training...', log_path)

    for epoch in range(1, n_epoch + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # TODO: Update optimizer learning rates optimizer param_groups
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate


        for batch in tqdm.tqdm(train_dataloader, desc='{}/{}'.format(epoch, n_epoch)):

            train_step = train_step + 1

            # Move batch to device
            batch = [
                in_.to(device) for in_ in batch
            ]

            # Unpack batch
            noisy_image, ground_truth = batch

            # TODO: Forward through the network, set return_all_outputs=True
            output_image = model.forward(noisy_image, True)

            # TODO: Compute loss
            loss, loss_info = model.compute_loss(output_image, ground_truth, w_losses)

            # TODO: Zero gradient, backpropagate, and update weights with optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training summary on Tensorboard and validate
            if (train_step % n_step_per_summary) == 0:

                model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    input_image=noisy_image,
                    output_image=output_image[-1].detach().clone(),
                    ground_truth=ground_truth,
                    scalars=loss_info,
                    n_image_per_summary=min(n_batch, n_image_per_summary))

                with torch.no_grad():

                    # TODO: Set model to eval mode
                    model.eval()

                    # TODO: Perform validation using validation dataloader and keep track of best results
                    # Record results into validation summary writer and log path
                    best_results = validate(model, val_dataloader, train_step, best_results, device, val_summary_writer, n_image_per_summary, log_path)

                    # TODO: Set model back to train mode
                    model.train()


            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step + start_step) * time_elapse / (train_step - start_step)

                log('Step={:6}/{}  Loss={:.7f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain),
                    log_path)

                # TODO: Save checkpoint
                model.save_model(model_checkpoint_path.format(train_step), train_step, optimizer)


    # Log last step
    time_elapse = (time.time() - time_start) / 3600
    time_remain = (n_train_step - train_step + start_step) * time_elapse / (train_step - start_step)

    log('Step={:6}/{}  Loss={:.7f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
        train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain),
        log_path)

    # TODO: Save checkpoint
    model.save_model(model_checkpoint_path.format(train_step), train_step, optimizer)


def validate(model,
             dataloader,
             step,
             best_results,
             device,
             summary_writer,
             n_image_per_summary=4,
             log_path=None):

    n_sample = len(dataloader)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)

    input_image_summary = []
    output_image_summary = []
    ground_truth_summary = []

    for idx, batch in enumerate(dataloader):

        # Unpack batch
        noisy_image, ground_truth = batch

        # TODO: Move noisy image to device
        noisy_image = noisy_image.to(device)

        # TODO: Forward through the network, return_all_outputs=False
        output_image = model.forward(noisy_image, return_all_outputs=False)

        # Collect samples for Tensorboard display
        if (idx % (n_sample // n_image_per_summary)) == 0 and summary_writer is not None:
            input_image_summary.append(noisy_image)
            output_image_summary.append(output_image)
            ground_truth_summary.append(ground_truth)

        # TODO: Move output image to CPU and convert output and ground truth to numpy to validate
        output_image = output_image.to('cpu').numpy()
        ground_truth = ground_truth.numpy()

        # TODO: Compute evaluation metrics using eval_utils
        mae[idx] = eval_utils.mean_abs_err(output_image, ground_truth)
        rmse[idx] = eval_utils.root_mean_sq_err(output_image, ground_truth)

    # TODO: Compute mean scores
    mae = mae.mean()
    rmse = rmse.mean()

    # TODO: Log to lists of inputs, outputs, ground truth summaries to Tensorboard
    # Concatenate lists along batch dimension, create dictionary with metrics as keys and scores as value
    # for logging into Tensorboard
    if summary_writer is not None:
        model.log_summary(
            summary_writer=summary_writer,
            tag='val',
            step=step,
            input_image=torch.cat(input_image_summary, dim=0),
            output_image=torch.cat(output_image_summary, dim=0),
            ground_truth=torch.cat(ground_truth_summary, dim=0),
            scalars={'MAE': mae, 'RMSE': rmse},
            n_image_per_summary=n_image_per_summary
        )

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse),
        log_path)

    # TODO: If the model improves over best results on both metrics by 2nd decimal
    n_improve = int(best_results['mae'] - mae > 0.01) + int(best_results['rmse'] - rmse > 0.01)


    # Update best results
    if n_improve > 1:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse']), log_path)

    return best_results

'''
Helper functions for logging
'''
def log_input_settings(log_path,
                       n_batch=None,
                       n_height=None,
                       n_width=None,
                       random_noise_type='none',
                       random_noise_spread=-1):

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

    log('random_noise_type={}  random_noise_spread={}'.format(
        random_noise_type, random_noise_spread),
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
                          # Photometric data augmentations
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_hue,
                          augmentation_random_saturation,
                          # Geometric data augmentations
                          augmentation_random_flip_type,):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Photometric data augmentations:', log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_hue={}'.format(augmentation_random_hue),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)

    log('Geometric data augmentations:', log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
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
                        n_image_per_summary=None,
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

    if n_image_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_image_per_summary={}'
        summary_settings_vars.append(n_image_per_summary)

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
