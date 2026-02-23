import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_map(tif_path, title, cmap, output_path):
    if not os.path.exists(tif_path):
        print(f"Error: {tif_path} does not exist.")
        return
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        # Handle nodata
        nodata = src.nodata
        if nodata is not None:
             data = np.ma.masked_equal(data, nodata)
        else:
             # Try common nodata values if not explicitly set
             data = np.ma.masked_less_equal(data, -9999)
             
        plt.figure(figsize=(10, 8))
        im = plt.imshow(data, cmap=cmap)
        plt.colorbar(im, label=title)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated {output_path}")

if __name__ == "__main__":
    os.makedirs('results/figures', exist_ok=True)
    plot_map('results/data/LSM_Map_Raw.tif', 'Landslide Susceptibility Map', 'viridis', 'results/figures/susceptibility_map.png')
    plot_map('results/data/LSM_Map_Classified.tif', 'Classified Susceptibility Map', 'inferno', 'results/figures/susceptibility_map_classified.png')
    plot_map('results/data/Uncertainty_Map_Raw.tif', 'Uncertainty Map', 'magma', 'results/figures/uncertainty_map.png')
    print("Map generation complete.")
