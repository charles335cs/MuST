import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.fwm import FWM
from options import FWM_Options
import argparse
import torchvision.utils as vutils

def compute_bit_accuracy(gt, pred):
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    pred_bin = (pred > 0.5).astype(np.float32)
    return np.mean(gt == pred_bin)

def load_from_checkpoint(net:FWM,checkpoint):
    net.enc_dec.encoder.load_state_dict(checkpoint['enc-model'],strict=False)
    net.enc_dec.decoder.load_state_dict(checkpoint['dec-model'],strict=False)
    net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    net.discriminator.load_state_dict(checkpoint['discrim-model'],strict=False)
    net.optimizer_dis.load_state_dict(checkpoint['discrim-optim'])
    net.localizer.load_state_dict(checkpoint['loc-model'],strict=False)
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
    batchsize= 1   
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transforms.ToTensor()(img).unsqueeze(0)  # [1,3,H,W]
    img = img * 2 - 1
    return img.to(device)  

def save_tensor_image(tensor, path):
    if tensor.dim() == 4 and tensor.size(1) == 1:  
        tensor = tensor.repeat(1,3,1,1)
    if tensor.dim() == 3:  
        tensor = tensor.unsqueeze(0)
    
    vutils.save_image(tensor, path, normalize=True, scale_each=True)

def main(args):
    model = FWM(fwm_config, device, None, None).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    load_from_checkpoint(model, checkpoint)
    model.eval()

    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_messages, exist_ok=True)

    image_files = [f for f in os.listdir(args.input_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sum_acc = 0
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(args.input_images, img_name)
        image = load_image(img_path)

        message = torch.randint(0, 2, (1, fwm_config.message_length)).float().to(device)
        with torch.no_grad():
            mask = torch.ones_like(image)
            encoded_images, input_enc, im_w, ori_img, origin_shape, mask_out = model.enc_dec.encoder(image, mask, message)
            encoded_img = im_w
            save_tensor_image(encoded_img, os.path.join(args.output_images, f"{img_name}"))
            wm_image = load_image(os.path.join(args.output_images, f"{img_name}"))
            pred_mask = torch.ones_like(wm_image[:, :1, :, :])
            decoded_message, _  = model.enc_dec.decoder(message, pred_mask, wm_image, no_noise=True)

            acc = compute_bit_accuracy(message, decoded_message)
            print(f"Bit Accuracy: {acc:.4f}")
            sum_acc += acc
        
        np.save(os.path.join(args.output_messages, f"{img_name.split('.')[0]}.npy"), message.cpu().numpy())

        print(f"Encoded {img_name} -> {args.output_images}")
    print(f"Average Bit Accuracy over {len(image_files)} images: {sum_acc/len(image_files):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_images", type=str, default='put your image path here')
    parser.add_argument("--checkpoint", type=str, default='put your checkpoint path here')
    parser.add_argument("--output_images", type=str, default="put your output images path here")
    parser.add_argument("--output_messages", type=str, default="put your output messages path here")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    main(args)
