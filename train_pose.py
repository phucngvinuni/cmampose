import os
import argparse
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from feeder.mmfi import make_dataset, make_dataloader
from utils import calulate_error, compute_pck_pckh, setup_seed

from model.omniwipose import CMambaPose


def kinematic_bone_loss(pred_pose, gt_pose):
    edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
             [12, 13], [8, 14], [14, 15], [15, 16]]
    loss_bone = 0.0
    for j1, j2 in edges:
        pred_bone_len = torch.sqrt(torch.sum((pred_pose[:, j1, :] - pred_pose[:, j2, :])**2, dim=-1) + 1e-8)
        gt_bone_len = torch.sqrt(torch.sum((gt_pose[:, j1, :] - gt_pose[:, j2, :])**2, dim=-1) + 1e-8)
        loss_bone += F.mse_loss(pred_bone_len, gt_bone_len)
    return loss_bone / len(edges)


def root_relative_loss(pred_pose, gt_pose):
    pred_rel = pred_pose - pred_pose[:, 0:1, :]
    gt_rel = gt_pose - gt_pose[:, 0:1, :]
    return F.mse_loss(pred_rel, gt_rel)


def absolute_loss(pred_pose, gt_pose):
    return F.mse_loss(pred_pose, gt_pose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="C-MambaPose Training & Evaluation Stage")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", required=True)
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    # 1. Device and Random Seed setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(config['seed'])
    
    # 2. Paths and logging setup
    weights_path = os.path.join(config['save_path'], config['dataset_name'], config['setting'], config['experiment_name'])
    log_path = os.path.join('logs', config['dataset_name'], config['setting'], config['experiment_name'])
    test_feature_path = os.path.join('test_feature', config['dataset_name'], config['setting'], config['experiment_name'])
    
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(test_feature_path, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_path)
    
    # 3. Create MMFi DataLoaders
    train_dataset, val_dataset = make_dataset(config['training_semi'], config['dataset_root'], config)
    
    rng_generator = torch.Generator()
    rng_generator.manual_seed(config['seed'])
    
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=config['batch_size'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=config['batch_size'])
    
    # 4. Initialize C-MambaPose SOTA Model
    model = CMambaPose(
        image_size=(114, 10),
        patch_size=(2, 2),
        emb_dim=config.get('emb_dim', 256),
        front_layer=config.get('front_layer', 2),
        front_head=4,
        input_dim=6,
        keypoints=17,
        coor_num=3,
        token_num=285,
        dataset=config['dataset_name'],
        num_person=config['num_person'],
        graformer_layers=config.get('graformer_layers', 4),
        graformer_dropout=0.4,
        dropout=0.3,
        graformer_hid_dim=config.get('graformer_hid_dim', 128),
        patchify_dim=config.get('patchify_dim', 128)
    ).to(device)
    
    # 5. Optimizer & LR Scheduler setup (Warmup + Cosine Annealing)
    optim = torch.optim.AdamW(model.parameters(), lr=config['base_learning_rate'], weight_decay=0.1)
    
    # Check if this setting uses static normalization / no warmup scheduler (e.g. Protocol 2 S3)
    if config['experiment_name'] == 'omniwipose_v17_static_norm_no_warmup':
        scheduler = CosineAnnealingLR(optim, T_max=config['total_epoch'], eta_min=1e-6)
    else:
        warmup = LinearLR(optim, start_factor=0.02, end_factor=1.0, total_iters=5)
        cosine = CosineAnnealingLR(optim, T_max=max(config['total_epoch'] - 5, 1), eta_min=1e-6)
        scheduler = SequentialLR(optim, schedulers=[warmup, cosine], milestones=[5])

    # 6. Training Loop
    best_val_mpjpe = float('inf')
    best_val_pampjpe = float('inf')
    best_val_pck = [0.0] * 5
    pck_order = ['50', '40', '30', '20', '10']
    
    for epoch in range(config['total_epoch']):
        model.train()
        train_losses = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            val_csi_phase = batch_data['input_wifi-csi_phase'].to(device)
            val_csi_amp = batch_data['input_wifi-csi_amp'].to(device)
            val_pose_gt = batch_data['output'].to(device)
            
            # Forward pass
            predicted_pose, _ = model(val_csi_phase, val_csi_amp)
            
            # Multi-task loss calculation (Absolute + Relative + Kinematic Bone Consistency)
            l_abs = absolute_loss(predicted_pose, val_pose_gt)
            l_rel = root_relative_loss(predicted_pose, val_pose_gt)
            l_bone = kinematic_bone_loss(predicted_pose, val_pose_gt)
            bone_weight = float(config.get('bone_weight', 0.1))
            
            loss = 0.2 * l_abs + 1.0 * l_rel + bone_weight * l_bone
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            train_losses.append(loss.item())
            
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Epoch {epoch}: Training Loss = {avg_train_loss:.6f}")
        
        # 7. Validation Loop
        model.eval()
        val_losses = []
        mpjpe_list = []
        pampjpe_list = []
        pck_iter = [[] for _ in range(5)]
        
        feature_list = []
        gt_list = []
        pred_list = []
        wifi_list = []
        label_list = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                val_csi_phase = batch_data['input_wifi-csi_phase'].to(device)
                val_csi_amp = batch_data['input_wifi-csi_amp'].to(device)
                val_pose_gt = batch_data['output'].to(device)
                
                predicted_val_pose, pred_fea = model(val_csi_phase, val_csi_amp)
                
                l_abs = absolute_loss(predicted_val_pose, val_pose_gt)
                l_rel = root_relative_loss(predicted_val_pose, val_pose_gt)
                l_bone = kinematic_bone_loss(predicted_val_pose, val_pose_gt)
                bone_weight = float(config.get('bone_weight', 0.1))
                
                loss = 0.2 * l_abs + 1.0 * l_rel + bone_weight * l_bone
                val_losses.append(loss.item())
                
                # Append lists for npz logging
                feature_list.append(pred_fea.cpu().numpy())
                gt_list.append(val_pose_gt.cpu().numpy())
                pred_list.append(predicted_val_pose.cpu().numpy())
                wifi_list.append(val_csi_phase.cpu().numpy())
                if 'label' in batch_data:
                    label_list.append(batch_data['label'].cpu().numpy())
                
                # Calculate metric errors
                for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(compute_pck_pckh(
                        predicted_val_pose.permute(0,2,1).cpu().numpy(),
                        val_pose_gt.permute(0,2,1).cpu().numpy(),
                        percentage,
                        align=False,
                        dataset=config['dataset_name']
                    ))
                mpjpe, pampjpe, _, _ = calulate_error(predicted_val_pose.cpu().numpy(), val_pose_gt.cpu().numpy(), align=False)
                mpjpe_list += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()
                
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
            avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)
            
            pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]
            
            gt_list = np.concatenate(gt_list, axis=0)
            pred_list = np.concatenate(pred_list, axis=0)
            feature_list = np.concatenate(feature_list, axis=0)
            wifi_list = np.concatenate(wifi_list, axis=0)
            if len(label_list) > 0:
                label_list = np.concatenate(label_list, axis=0)
                
        print(f"Validation loss: {avg_val_loss:.6f}")
        print(f"MPJPE: {avg_val_mpjpe:.2f} mm | PA-MPJPE: {avg_val_pampjpe:.2f} mm")
        print(f"PCK@50: {pck_overall[0]:.4f} | PCK@20: {pck_overall[3]:.4f}")
        
        # Save checkpoints for best performing states
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            print(f"New best MPJPE: {best_val_mpjpe:.2f} mm | Saving best checkpoint at Epoch {epoch}...")
            torch.save(model.state_dict(), os.path.join(weights_path, 'pose_mpjpe.pt'))
            
        if avg_val_pampjpe < best_val_pampjpe:
            best_val_pampjpe = avg_val_pampjpe
            torch.save(model.state_dict(), os.path.join(weights_path, 'pose_pampjpe.pt'))
            
        for idx, pck_value in enumerate(pck_overall):
            if pck_value > best_val_pck[idx]:
                best_val_pck[idx] = pck_value
                torch.save(model.state_dict(), os.path.join(weights_path, f'pose_pck{pck_order[idx]}.pt'))
                
        # Logging to TensorBoard
        writer.add_scalars('cls/loss', {'train': avg_train_loss, 'val': avg_val_loss}, global_step=epoch)
        if scheduler is not None:
            scheduler.step()
            
    print("*" * 80)
    print(f"Training completed successfully!")
    print(f"Best validation MPJPE: {best_val_mpjpe:.2f} mm")
    print(f"Best validation PA-MPJPE: {best_val_pampjpe:.2f} mm")
    print(f"Best PCK@50: {best_val_pck[0]:.4f} | Best PCK@20: {best_val_pck[3]:.4f}")
    print("*" * 80)
