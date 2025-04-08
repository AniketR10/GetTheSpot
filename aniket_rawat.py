import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, filters, morphology

def process_img(fits_file):
    
    with fits.open(fits_file) as hdul:
        img_data = hdul[0].data
    normalized = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

    # gaussian blur
    blurred = ndimage.gaussian_filter(normalized, sigma=5)
    
    #thresholding initial
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    
   # cleaning
    binary = morphology.binary_closing(binary)
    binary = morphology.binary_opening(binary)
    cntrs = measure.find_contours(binary, 0.5)
    
    # this is the sun
    sun = max(cntrs, key=len)
    
    y,x= sun.T
    x_center = np.mean(x)
    y_center = np.mean(y)
    
    # dist from center and radius
    dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    radius = np.mean(dist)
    
    #circular mask for sun 
    x_idx, y_idx = np.ogrid[:img_data.shape[0], :img_data.shape[1]]
    dist_from_center = np.sqrt((x_idx - x_center)**2 + (y_idx - y_center)**2)
    sun_mask = dist_from_center <= radius * 0.95 
    
    masked_img = normalized * sun_mask
    
   # background
    bg = ndimage.median_filter(masked_img, size=25)
    sunspot = bg - masked_img
    
    # identify sunspot
    sunspot_threshold = np.mean(sunspot[sun_mask]) + 2 * np.std(sunspot[sun_mask])
    sunspots_binary = (sunspot > sunspot_threshold) & sun_mask
    
   # clean the sunspots
    sunspots_binary = morphology.binary_opening(sunspots_binary)
    sunspots_binary = morphology.binary_closing(sunspots_binary)
    sunspots_binary = sunspots_binary.astype(int)
    
    return { 'sun_center': (x_center, y_center),'sun_radius': radius,'sun_limb_binary': binary, 'sunspots_binary': sunspots_binary }

def img_results(img_path, results):
   
    with fits.open(img_path) as hdul:
        img_data = hdul[0].data
    normalized = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    fig,axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # original img with limb
    axes[0].imshow(normalized, cmap='gray')
    x_center, y_center = results['sun_center']
    radius = results['sun_radius']
    circle = plt.Circle((x_center, y_center), radius, color='r', fill=False)
    axes[0].add_patch(circle)
    axes[0].set_title('Original with Sun Limb')
    
    # display the center and radius
    info_text = f"Center: ({x_center:.2f}, {y_center:.2f})\nRadius: {radius:.2f} "
    axes[0].text(0.95, 1, info_text, transform=axes[0].transAxes,color='white', fontsize=12, horizontalalignment='right', verticalalignment='top')
    
    # sunlimb binary img
    axes[1].imshow(results['sun_limb_binary'], cmap='gray')
    axes[1].set_title('Sun Limb Binary')
    
    # sunspots binary img
    axes[2].imshow(results['sunspots_binary'], cmap='gray')
    axes[2].set_title('Sunspots Binary')
    
    plt.tight_layout()
    plt.show()
    
    # print in conso;le
    print(f"Sun center coordinates: ({x_center:.2f}, {y_center:.2f})")
    print(f"Sun radius: {radius:.2f} ")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1: # if you want to write the filename in front of the command in terminal
        fits_file = sys.argv[1]
    else:
        fits_file = "./SIP_USO_PRL/UDAI.FDGB.03062019.080021.864.fits" # putting the address of the filename here
    
    results = process_img(fits_file)
    img_results(fits_file, results)
