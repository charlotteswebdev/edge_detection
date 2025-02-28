# Standard library imports
import requests
from io import BytesIO

# Third party imports
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt

def load_image_from_url(url):
    """Load and convert image from URL to grayscale."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('L')
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def prepare_image_tensor(image):
    """Convert PIL image to normalized torch tensor."""
    to_tensor = tv.transforms.ToTensor()
    return to_tensor(image).unsqueeze(0) / 255.0

def create_edge_filters():
    """Create both vertical and horizontal edge detection filters."""
    vertical = torch.tensor([
        [1., 0, -1],
        [1., 0, -1],
        [1., 0, -1]
    ]).unsqueeze(0).unsqueeze(0)
    
    horizontal = torch.tensor([
        [1., 1., 1.],
        [0., 0., 0.],
        [-1., -1., -1.]
    ]).unsqueeze(0).unsqueeze(0)
    
    return vertical, horizontal

def detect_edges(image_tensor, filters):
    """Apply edge detection filters and return all edge maps."""
    vertical_map = F.conv2d(image_tensor, filters[0]).squeeze()
    horizontal_map = F.conv2d(image_tensor, filters[1]).squeeze()
    combined_map = torch.sqrt(vertical_map**2 + horizontal_map**2)
    return vertical_map, horizontal_map, combined_map

def plot_edge_maps(original, vertical, horizontal, combined):
    """Plot original and all edge maps in 2x2 grid."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(vertical, cmap='gray')
    ax2.set_title('Vertical Edges')
    ax2.axis('off')
    
    ax3.imshow(horizontal, cmap='gray')
    ax3.set_title('Horizontal Edges')
    ax3.axis('off')
    
    ax4.imshow(combined, cmap='gray')
    ax4.set_title('Combined Edges')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    url = 'https://media.istockphoto.com/vectors/chess-silhouettes-vector-id165635822?b=1&k=20&m=165635822&s=612x612&w=0&h=pmf6FVa--nzyWCKb0SyTkIi3xdaHaamJuaR-FIjw1iI='
    image = load_image_from_url(url)
    image_tensor = prepare_image_tensor(image)
    
    # Detect edges
    v_filter, h_filter = create_edge_filters()
    v_edges, h_edges, combined = detect_edges(image_tensor, (v_filter, h_filter))
    
    # Convert tensors to images for display
    to_pil = tv.transforms.ToPILImage()
    v_img = to_pil(v_edges)
    h_img = to_pil(h_edges)
    c_img = to_pil(combined)
    
    # Display results
    plot_edge_maps(image, v_img, h_img, c_img)