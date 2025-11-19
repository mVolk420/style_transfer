import argparse
import torch
import torchvision.models as models
import os
from utils import load_image, save_image
from style_transfer import perform_style_transfer

def main():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True, help='Path to content image (in input/ folder)')
    parser.add_argument('--style', type=str, required=True, help='Path to style image or preset name')
    parser.add_argument('--output', type=str, default=None, help='Path to output image (default: output/ folder)')
    parser.add_argument('--steps', type=int, default=300, help='Number of optimization steps')
    parser.add_argument('--imsize', type=int, default=512 if torch.cuda.is_available() or torch.backends.mps.is_available() else 128, help='Image size')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # desired size of the output image
    image_size = args.imsize
    
    # Resolve content path
    content_path = args.content
    if not os.path.exists(content_path) and os.path.exists(os.path.join("input", content_path)):
        content_path = os.path.join("input", content_path)
    
    if not os.path.exists(content_path):
        print(f"Error: Content image not found at {content_path}")
        return

    # Check if style is a preset
    style_path = args.style
    style_name = os.path.splitext(os.path.basename(args.style))[0]
    
    if os.path.exists(os.path.join("styles", f"{args.style}.jpg")):
        style_path = os.path.join("styles", f"{args.style}.jpg")
        style_name = args.style
    elif os.path.exists(os.path.join("styles", f"{args.style}.png")):
        style_path = os.path.join("styles", f"{args.style}.png")
        style_name = args.style

    # Resolve output path
    if args.output is None:
        content_name = os.path.splitext(os.path.basename(content_path))[0]
        output_filename = f"{content_name}_{style_name}.jpg"
        output_path = os.path.join("output", output_filename)
    else:
        output_path = args.output

    print(f"Loading content image: {content_path}")
    content_img = load_image(content_path, image_size)
    
    # Resize style image to match content image dimensions
    content_shape = (content_img.shape[2], content_img.shape[3])
    
    print(f"Loading style image: {style_path}")
    style_img = load_image(style_path, image_size, shape=content_shape)
    
    print(f"Output will be saved to: {output_path}")

    print("Loading VGG19 model...")
    vgg_model = models.vgg19(pretrained=True).features.to(device).eval()

    # VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] 
    # and std=[0.229, 0.224, 0.225].
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    print("Starting Style Transfer...")
    output = perform_style_transfer(vgg_model, norm_mean, norm_std,
                                content_img, style_img, input_img, num_steps=args.steps)

    print(f"Saving output to {output_path}")
    save_image(output, output_path)
    print("Done!")

if __name__ == '__main__':
    main()
