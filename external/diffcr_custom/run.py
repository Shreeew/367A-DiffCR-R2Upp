import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

import torchvision.utils as vutils

import tifffile

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric

import sys


#for training
sys.argv = [
    "run.py",
    "--config", "config/solafune.json",
    "--phase", "train",
    "--gpu_ids", "0"
]

#sys.argv = [
#    "run.py",
#    "--config", "config/solafune_test.json",
#    "--phase", "test",
#    "--gpu_ids", "0"
#]



def tensor_to_rgb(tensor):
    """Convert 12-channel tensor to 3-channel RGB for visualization (normalizes values)."""
    # Pick Sentinel-2-like RGB: bands 4,3,2 → indices 3,2,1 (but your case seems 2,1,0)
    rgb = tensor[0, [2, 1, 0], :, :]  # shape: [3, H, W]
    rgb = rgb - rgb.min()
    rgb = rgb / (rgb.max() + 1e-8)
    return rgb.clamp(0, 1)



def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    print("▶ Number of training samples in dataset:", len(phase_loader.dataset))
    print("▶ Dataloader length (steps per epoch):", len(phase_loader))

    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    #metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']] # original instead of lines below
    metrics = []
    if opt['model'].get('which_metrics') is not None:
        metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]

    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )



    phase_logger.info('Begin model {}.'.format(opt['phase']))

    try:
        #if opt['phase'] == 'train':
        #    print("Calling model.train() now")
        #    model.train()
        #    print("model.train() returned successfully")
        if opt['phase'] == 'train':
            model.train()
            dataloader = phase_loader
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.get('lr', 1e-4))
            max_epoch = opt.get('epochs', 50)
            # === Resume logic ===
            start_epoch = 17
            resume_ckpt_path = os.path.join(opt['save_dir'], "model_epoch17.pth")  # change to latest available

            if os.path.exists(resume_ckpt_path):
                print(f"Resuming from {resume_ckpt_path}")
                model.load_state_dict(torch.load(resume_ckpt_path, map_location="cuda"))
                start_epoch = 17  # must match checkpoint epoch

            for epoch in range(start_epoch, max_epoch):
                print(f"\n[Epoch {epoch+1}/{max_epoch}]")
                for i, data in enumerate(dataloader):
                    gt = data['y0'].cuda(non_blocking=True)
                    cond = data['x'].cuda(non_blocking=True)

                    optimizer.zero_grad()
                    loss = model(gt, cond)
                    loss.backward()
                    optimizer.step()

                    print(f"  Iter {i:04d} | Loss: {loss.item():.6f}")

                checkpoint_dir = opt['save_dir']
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
        elif opt['phase'] == 'test':
            model.load_state_dict(torch.load("checkpoints/model_epoch17.pth"))
            model.eval()

            for i, data in enumerate(phase_loader):
                cond = data['x'].cuda()
                filename = f"sample_{i:03d}"
                print("before restoration")
                with torch.no_grad():
                    dummy_y0 = torch.zeros_like(cond)
                    output, _ = model.restoration(y_cond=cond, y_0=dummy_y0)  # shape: [1, 12, H, W]
                print("after restoration")
                # Save 12-channel .tif for future use
                out_np = output[0].cpu().numpy().astype('float32')  # [12, H, W]
                out_tif_path = os.path.join(opt['path']['result'], f"{filename}.tif")
                tifffile.imwrite(out_tif_path, out_np)
                print("saved 12 channel")
                # Save RGB visualization as PNG for visualization
                rgb_output = tensor_to_rgb(output)  # [3, H, W], float in [0, 1]
                out_png_path = os.path.join(opt['path']['result'], f"{filename}.png")
                vutils.save_image(rgb_output, out_png_path)
                print("saved 3 channel")
                print(f"Saved: {out_png_path}, {out_tif_path}")
                break # Remove this break to process all images



        else:
            #model.test()
            raise ValueError(f"Unsupported phase: {opt['phase']}")
    finally:
        print("Closing writer")
        phase_writer.close()

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)