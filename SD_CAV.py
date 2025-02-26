import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Import existing style transfer components
from libs.Loader import Dataset
from libs.Matrix import MulLayer
from libs.utils import print_options
from libs.models import encoder3, encoder4, decoder3, decoder4

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

class SDGuidedCAV:
    """
    Implements style control using Stable Diffusion's CLIP text and image encoders
    for more artistically-aware style understanding.
    """
    def __init__(self, vgg, matrix, decoder, sd_model_id="stabilityai/stable-diffusion-2-1", device='cuda'):
        self.device = device
        self.vgg = vgg
        self.matrix = matrix
        self.decoder = decoder
        
        # Load SD components
        self.pipeline = StableDiffusionPipeline.from_pretrained(sd_model_id)
        self.text_encoder = self.pipeline.text_encoder.to(device)
        self.vae = self.pipeline.vae.to(device)
        self.tokenizer = self.pipeline.tokenizer
        
        # Put models in eval mode
        self.text_encoder.eval()
        self.vae.eval()
        
        # Initialize alignment matrices
        self.sd_to_vgg_alignment = None
        
        logger.info("Initialized SD-guided CAV controller")
    
    def get_text_embeddings(self, prompt, negative_prompt=""):
        """
        Gets text embeddings from SD's CLIP text encoder.
        
        Args:
            prompt: Style description text
            negative_prompt: Optional negative prompt
        Returns:
            Text embeddings tensor
        """
        with torch.no_grad():
            # Tokenize text
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get text encoder hidden states
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]
            
            # Handle negative prompt if provided
            if negative_prompt:
                uncond_input = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=text_input.input_ids.shape[-1],
                    truncation=True,
                    return_tensors="pt"
                )
                uncond_embeddings = self.text_encoder(
                    uncond_input.input_ids.to(self.device)
                )[0]
                
                # Concatenate conditional and unconditional embeddings
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            return text_embeddings
    
    def get_image_embeddings(self, image):
        """
        Gets latent representations from SD's VAE encoder.
        
        Args:
            image: Input image tensor [1, 3, H, W]
        Returns:
            VAE latent representation
        """
        with torch.no_grad():
            # Ensure image is in correct format
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)
            else:
                # Convert PIL Image to tensor
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Scale image to [-1, 1]
            image = 2 * image - 1
            
            # Get VAE encoding
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            return latents
    
    def learn_space_alignment(self, style_images, style_prompts):
        """
        Learns alignment between SD latent space and VGG feature space.
        
        Args:
            style_images: List of style image tensors
            style_prompts: List of corresponding style descriptions
        """
        sd_embeddings = []
        vgg_features = []
        
        with torch.no_grad():
            for image, prompt in zip(style_images, style_prompts):
                # Get SD embeddings
                text_emb = self.get_text_embeddings(prompt)
                image_emb = self.get_image_embeddings(image)
                
                # Combine text and image embeddings
                sd_emb = torch.cat([
                    text_emb.mean(dim=1),
                    image_emb.view(image_emb.size(0), -1)
                ], dim=1)
                sd_embeddings.append(sd_emb)
                
                # Get VGG features
                vgg_feats = self.vgg(image)[self.layer_name]
                pooled = F.adaptive_avg_pool2d(vgg_feats, (1, 1)).squeeze()
                vgg_features.append(pooled)
            
            # Stack embeddings
            sd_matrix = torch.cat(sd_embeddings, dim=0)
            vgg_matrix = torch.cat(vgg_features, dim=0)
            
            # Learn alignment using CCA
            self.sd_to_vgg_alignment = self._compute_cca_alignment(
                sd_matrix, vgg_matrix
            )
            
        logger.info("Learned SD to VGG space alignment")
    
    def _compute_cca_alignment(self, X, Y):
        """Computes CCA alignment between two spaces."""
        # Center the data
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute correlation matrices
        Cxx = torch.mm(X.t(), X) + torch.eye(X.shape[1]).to(self.device) * 1e-6
        Cyy = torch.mm(Y.t(), Y) + torch.eye(Y.shape[1]).to(self.device) * 1e-6
        Cxy = torch.mm(X.t(), Y)
        
        # Compute alignment matrix
        A = torch.mm(torch.inverse(Cxx), Cxy)
        return A
    
    def get_style_cav(self, target_style, base_style="a painting"):
        """
        Computes CAV using SD's understanding of artistic styles.
        
        Args:
            target_style: Target style description
            base_style: Base style description
        Returns:
            CAV tensor in VGG feature space
        """
        # Get style embeddings from SD
        target_emb = self.get_text_embeddings(target_style)
        base_emb = self.get_text_embeddings(base_style)
        
        # Compute difference in SD space
        style_diff = target_emb.mean(dim=1) - base_emb.mean(dim=1)
        
        # Project to VGG space
        if self.sd_to_vgg_alignment is None:
            raise ValueError("Must call learn_space_alignment first!")
        
        cav = torch.mm(style_diff, self.sd_to_vgg_alignment)
        return F.normalize(cav, dim=-1)
    
    def apply_style_cav(self, content_image, style_image, target_style, strength=1.0):
        """
        Applies SD-guided style CAV during transfer.
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            target_style: Target style description
            strength: Scaling factor for style modification
        Returns:
            Tuple of (modified image, transformation matrix)
        """
        # Get style CAV
        cav = self.get_style_cav(target_style)
        
        with torch.no_grad():
            # Get current style features
            style_features = self.vgg(style_image)[self.layer_name]
            
            # Reshape CAV to match feature dimensions
            cav_resized = cav.view(*style_features.shape[1:])
            
            # Get SD's understanding of the style
            style_latents = self.get_image_embeddings(style_image)
            style_text_emb = self.get_text_embeddings(target_style)
            
            # Use SD's understanding to modulate CAV strength
            style_confidence = F.cosine_similarity(
                style_latents.view(1, -1),
                style_text_emb.mean(dim=1)
            )
            adaptive_strength = strength * style_confidence
            
            # Apply CAV modification
            modified_features = style_features + (adaptive_strength * cav_resized)
            
            # Generate result
            result = self.decoder(modified_features)
            
            return result.clamp(0, 1), None
        

def main():
    """Main function implementing both traditional and text-guided CAV style transfer."""
    parser = parse_args()
    parser = add_text_cav_args(parser)  # Add text CAV arguments
    opt = parser.parse_args()
    
    opt.cuda = torch.cuda.is_available()
    print_options(opt)
    
    # Create output directories
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.matrixOutf, exist_ok=True)
    
    # Initialize models and data loaders
    vgg = encoder4() if opt.layer == 'r41' else encoder3()
    dec = decoder4() if opt.layer == 'r41' else decoder3()
    matrix = MulLayer(opt.layer)
    
    # Load model weights
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    matrix.load_state_dict(torch.load(opt.matrixPath))
    
    if opt.cuda:
        vgg.cuda()
        dec.cuda()
        matrix.cuda()
    
    # Initialize data loaders
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    content_loader = torch.utils.data.DataLoader(
        dataset=content_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=0
    )
    
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    style_loader = torch.utils.data.DataLoader(
        dataset=style_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=0
    )
    
    # Create comparison directories
    comparison_dir = os.path.join(opt.outf, 'comparisons')
    text_comparison_dir = os.path.join(opt.outf, 'text_comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(text_comparison_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = CAVVisualizer(len(opt.cav_strengths))
    
    if opt.use_text_cav:
        # Initialize text-guided controller
        controller = TextGuidedStyleController(vgg, matrix, dec, opt.layer)
        logger.info("Processing with text-guided CAVs...")
        process_with_text_guidance(
            controller,
            content_loader,
            style_loader,
            opt.text_prompts,
            opt.cav_strengths,
            opt
        )
    else:
        # Initialize traditional CAV controller
        controller = StyleCAVController(vgg, matrix, dec, opt.layer)
        logger.info("Processing with traditional image-based CAVs...")
        
        # Process each content-style pair
        for content_idx, (content, content_name) in enumerate(content_loader):
            contentV = content.cuda() if opt.cuda else content
            content_num = content_idx + 1
            
            for style_idx, (style, style_name) in enumerate(style_loader):
                styleV = style.cuda() if opt.cuda else style
                style_num = style_idx + 1
                
                try:
                    # Get style directory
                    style_output_dir = os.path.join(
                        opt.data_dir, 
                        'styled_outputs', 
                        f'style_{style_num:02d}'
                    )
                    
                    # Learn CAV from blur variations
                    blur_cav = controller.learn_cav_from_directory(style_output_dir)
                    
                    transfers = []
                    # Apply different CAV strengths
                    for strength in opt.cav_strengths:
                        transfer, matrix = controller.apply_cav(
                            contentV, styleV, blur_cav, strength
                        )
                        transfers.append(transfer)
                        
                        # Save output image
                        output_filename = (
                            f'cav_content{content_num:02d}_'
                            f'style_{style_num:02d}_'
                            f'strength_{strength:.1f}.png'
                        )
                        output_path = os.path.join(opt.outf, output_filename)
                        vutils.save_image(transfer, output_path)
                        
                        # Save transformation matrix
                        matrix_filename = (
                            f'cav_content{content_num:02d}_'
                            f'style_{style_num:02d}_'
                            f'strength_{strength:.1f}_matrix.pth'
                        )
                        matrix_path = os.path.join(opt.matrixOutf, matrix_filename)
                        torch.save(matrix, matrix_path)
                    
                    # Create comparison visualization
                    visualizer.create_comparison_plot(
                        transfers,
                        opt.cav_strengths,
                        contentV,
                        styleV,
                        comparison_dir,
                        content_idx,
                        style_num
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error processing content {content_num} "
                        f"with style {style_num}: {str(e)}"
                    )
                    continue
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()