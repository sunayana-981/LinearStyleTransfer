if __name__ == "__main__":

    import os
    import torch
    import cv2
    import argparse
    from libs.Loader import Dataset
    from libs.Matrix import MulLayer
    import torchvision.utils as vutils
    import torch.backends.cudnn as cudnn
    from libs.utils import print_options
    from libs.models import encoder3, encoder4, encoder5
    from libs.models import decoder3, decoder4, decoder5
    from tqdm import tqdm

    def generate_blur_dataset(style_dir, output_dir, sigma_levels=[5, 10, 15, 20,25,30,25,40,45,50]):
        """Generate a dataset of blurred images for different sigma levels."""
        os.makedirs(output_dir, exist_ok=True)

        for sigma in tqdm(sigma_levels, desc="Processing sigma levels"):
            sigma_dir = os.path.join(output_dir, f"sigma_{sigma}")
            os.makedirs(sigma_dir, exist_ok=True)

            for style_file in os.listdir(style_dir):
                if style_file.endswith((".jpg", ".png")):
                    style_path = os.path.join(style_dir, style_file)
                    image = cv2.imread(style_path)

                    # Apply Gaussian blur with the current sigma level
                    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)

                    # Save the blurred image in the corresponding sigma directory
                    output_path = os.path.join(sigma_dir, style_file)
                    cv2.imwrite(output_path, blurred_image)

    parser = argparse.ArgumentParser()
    parser.add_argument("--stylePath", default="data/style/",
                        help="path to style images")
    parser.add_argument("--outputPath", default="data/blurred_styles/",
                        help="path to save blurred images")

    ################# PREPARATIONS #################
    opt = parser.parse_args()

    # Generate blurred dataset
    generate_blur_dataset(opt.stylePath, opt.outputPath)
