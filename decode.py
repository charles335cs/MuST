import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.fwm import FWM
from options import FWM_Options
import argparse

def compute_bit_accuracy(gt, pred):
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    pred_bin = (pred > 0.5).astype(np.float32)
    return np.mean(gt == pred_bin)

def load_from_checkpoint(net:FWM, checkpoint):
    net.enc_dec.encoder.load_state_dict(checkpoint['enc-model'], strict=False)
    net.enc_dec.decoder.load_state_dict(checkpoint['dec-model'], strict=False)
    net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    net.discriminator.load_state_dict(checkpoint['discrim-model'], strict=False)
    net.optimizer_dis.load_state_dict(checkpoint['discrim-optim'])
    net.localizer.load_state_dict(checkpoint['loc-model'], strict=False)
    net.optimizer_loc.load_state_dict(checkpoint['loc-optim'])

fwm_config = FWM_Options(
    H=512,
    W=512,
    message_length=30,           
    encoder_blocks=8, encoder_channels=64,
    decoder_blocks=9, decoder_channels=64,
    use_discriminator=True,
    discriminator_blocks=4, discriminator_channels=64,
    decoder_loss=2,
    encoder_loss=0.7,
    adversarial_loss=1e-3,
    localization_loss=1e-2,
    enable_fp16=True,
    resize_bound=[1,1],
    batchsize=1   
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0)  # [1,3,H,W]
    img = img * 2 - 1
    return img.to(device)  

def main(args):
    model = FWM(fwm_config, device, None, None).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_from_checkpoint(model, checkpoint)
    model.eval()

    image_files = [f for f in os.listdir(args.input_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sum_acc = 0

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(args.input_images, img_name)
        msg_path = os.path.join(args.input_messages, f"{img_name.split('.')[0]}.npy")

        if not os.path.exists(msg_path):
            print(f"Warning: message file {msg_path} not found, skipping {img_name}")
            continue

        wm_image = load_image(img_path)
        gt_message = torch.from_numpy(np.load(msg_path)).float().to(device)

        with torch.no_grad():
            pred_mask = torch.ones_like(wm_image[:, :1, :, :])
            decoded_message, _ = model.enc_dec.decoder(gt_message, pred_mask, wm_image, no_noise=True)
            acc = compute_bit_accuracy(gt_message, decoded_message)
            print(f"{img_name} -> Bit Accuracy: {acc:.4f}")
            sum_acc += acc

    if len(image_files) > 0:
        print(f"\nAverage Bit Accuracy over {len(image_files)} images: {sum_acc/len(image_files):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images", type=str, default="put your watermarked images path here")
    parser.add_argument("--input_messages", type=str, default="put your messages path here")
    parser.add_argument("--checkpoint", type=str, default="put your checkpoint path here")
    args = parser.parse_args()
    main(args)
