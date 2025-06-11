import os
import torch
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms
from data_processing.collate_fn import collate_fn

from ops.distribute_utils import init_distributed_mode
from model.model_utils import load_model,save_checkpoint,save_model2path

from finetune.main_worker import configure_data_loader
from model.Finetune_Model_Head import Finetune_Model_Head
from model.pos_embed import interpolate_pos_embed_inputsize
from data_processing.predict_dataset import Finetune_Dataset
from model.pos_embed import expand_pos_embed_add_count_and_seq

def configure_inference_loader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def parse_text(config_file, data_dir):
        file_list = []
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    file_list.append(os.path.join(data_dir, line))
        return file_list

    file_list = parse_text(args.config_file, args.data_path)
    dataset = Finetune_Dataset(file_list, transform=transform,
                               window_height=args.input_row_size, window_width=args.input_col_size)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )

    return data_loader

def save_predictions(output_dir, filenames, predictions):
    os.makedirs(output_dir, exist_ok=True)
    for fname, pred in zip(filenames, predictions):
        save_path = os.path.join(output_dir, f"{fname}_prediction.npy")
        np.save(save_path, pred)

def load_trained_model(args, device='cuda'):
    patch_wise_size = (args.input_row_size // args.patch_size, args.input_col_size // args.patch_size)

    if args.sequence == 1:
        import model.Vision_Transformer_seq as Vision_Transformer
    else:
        import model.Vision_Transformer_count as Vision_Transformer
    
    vit_backbone = Vision_Transformer.VisionTransformer(
        img_size=(args.input_row_size, args.input_col_size),
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm
    )

    model = Finetune_Model_Head(
        vit_backbone,
        task=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=torch.nn.LayerNorm,
        pos_embed_size=patch_wise_size,
        use_sequence=(args.sequence == 1)
    )

    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        
        if args.sequence == 1:
            # Apply fix if needed
            if 'pos_embed' in checkpoint_model:
                checkpoint_model['pos_embed'] = expand_pos_embed_add_count_and_seq(
                    checkpoint_model['pos_embed'],
                    embed_dim=vit_backbone.embed_dim,
                    device='cpu'
                )

        interpolate_pos_embed_inputsize(vit_backbone, checkpoint_model, input_size=patch_wise_size, use_decoder=False)
        vit_backbone.load_state_dict(checkpoint_model, strict=False)

        interpolate_pos_embed_inputsize(model, checkpoint_model, input_size=patch_wise_size, use_decoder=True)
        model.load_state_dict(checkpoint_model, strict=False)
    else:
        print("Warning: Checkpoint file not found or invalid. Initializing model from scratch.")

    model.to(device)
    model.eval()
    return model

def predict_epoch(model, data_loader, device, output_dir):
    model.eval()
    all_filenames = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference"):
            input_matrix = batch[0].to(device)
            total_count = None #batch[1].to(device)
            sequence_input = batch[5].to(device)
            if (sequence_input == None):
                print("no sequence")
            filenames = batch[6]


            outputs = model(input_matrix, total_count, sequence_input)
            predictions = outputs.cpu().numpy()

            all_filenames.extend(filenames)
            all_predictions.extend(predictions)

    save_predictions(output_dir, all_filenames, all_predictions)
    print(f"Saved all predictions to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with HiCFoundation model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--config_file', type=str, required=True, help='Config text file listing data subdirectories')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save predictions')
    parser.add_argument('--input_row_size', type=int, default=224)
    parser.add_argument('--input_col_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sequence', type=int, default=1)
    parser.add_argument('--device', default='cuda', help='Device to run on')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_loader = configure_inference_loader(args)
    model = load_trained_model(args, device)

    predict_epoch(model, data_loader, device, args.output_dir)

if __name__ == "__main__":
    main()
