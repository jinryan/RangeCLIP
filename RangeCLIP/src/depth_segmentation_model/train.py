import argparse
import sys
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from RangeCLIP.src.depth_segmentation_model.train_util import train_depth_clip_model

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank

def cleanup_ddp():
    dist.destroy_process_group()
    
parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--labeled_metadata_path',
    type=str, required=True, help='Path to labeled dataset metadata.csv')
parser.add_argument('--labels_path',
    type=str, required=True, help='Path to dataset labels')
parser.add_argument('--equivalence_dict_path',
    type=str, required=True, help='Path to equivalence dictionary CSV file')

# Batch parameters
parser.add_argument('--batch_size',
    type=int, default=16, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=128, help='Height of sample for resizing')
parser.add_argument('--n_width',
    type=int, default=128, help='Width of each sample for resizing')

# Network settings
parser.add_argument('--unet_architecture',
    type=str, required=True, help='UNet encoder architecture, e.g. resnet')

parser.add_argument('--clip_model_name',
    type=str, default='openai/clip-vit-base-patch32', help='CLIP model variant to use')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[2e-4, 1e-4, 5e-5, 1e-5], help='Space delimited list of learning rates')
parser.add_argument('--scheduler_type',
    type=str, default='multi_step', help='Options: multi_step, cosine_annealing, reduce_on_plateau')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[10, 20, 30, 35], help='Steps for changing learning rate')

# Loss settings
parser.add_argument('--w_weight_decay',
    type=float, default=0.0, help='Weight decay for regularization')

# Checkpointing and logging
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=5000, help='Steps per checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=1000, help='Steps per training summary')
parser.add_argument('--n_sample_per_summary',
    type=int, default=4, help='Number of samples per visualization')
parser.add_argument('--validation_start_step',
    type=int, default=5000, help='Start validation after this many steps')
parser.add_argument('--restore_path_model',
    type=str, default=None, help='Path to restore model from')
parser.add_argument('--restore_path_encoder',
    type=str, default=None, help='Path to separately restore encoder weights')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu or cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for data loading')

args = parser.parse_args()

if __name__ == '__main__':
    # Sanity check
    assert len(args.learning_rates) == len(args.learning_schedule), "Mismatch in learning rates and schedule lengths"

    device, local_rank = setup_ddp()

    train_depth_clip_model(
        labeled_metadata_path=args.labeled_metadata_path,
        labels_path=args.labels_path,
        equivalence_dict_path=args.equivalence_dict_path,
        batch_size=args.batch_size,
        n_height=args.n_height,
        n_width=args.n_width,
        unet_architecture=args.unet_architecture,
        learning_rates=args.learning_rates,
        learning_schedule=args.learning_schedule,
        scheduler_type=args.scheduler_type,
        w_weight_decay=args.w_weight_decay,
        checkpoint_path=args.checkpoint_path,
        n_step_per_checkpoint=args.n_step_per_checkpoint,
        n_step_per_summary=args.n_step_per_summary,
        n_sample_per_summary=args.n_sample_per_summary,
        validation_start_step=args.validation_start_step,
        restore_path_model=args.restore_path_model,
        restore_path_encoder=args.restore_path_encoder,
        clip_model_name=args.clip_model_name,
        device=device,
        n_thread=args.n_thread,
        local_rank=local_rank,
    )
    
    cleanup_ddp()
