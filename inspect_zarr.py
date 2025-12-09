#!/usr/bin/env python3
"""
Script to inspect and visualize zarr dataset contents
"""
import zarr
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def inspect_zarr_group(group, indent=0):
    """Recursively inspect a zarr group and print its structure"""
    prefix = "  " * indent
    
    for key in group.keys():
        item = group[key]
        if isinstance(item, zarr.core.Array):
            print(f"{prefix}📊 {key}:")
            print(f"{prefix}   Shape: {item.shape}")
            print(f"{prefix}   Dtype: {item.dtype}")
            print(f"{prefix}   Chunks: {item.chunks}")
            print(f"{prefix}   Size: {item.nbytes / 1024:.2f} KB")
            
            # Show some statistics for numeric arrays
            if np.issubdtype(item.dtype, np.number):
                data = item[:]
                print(f"{prefix}   Min: {np.min(data):.4f}")
                print(f"{prefix}   Max: {np.max(data):.4f}")
                print(f"{prefix}   Mean: {np.mean(data):.4f}")
                print(f"{prefix}   Std: {np.std(data):.4f}")
            
            # Show first few values
            if len(item) > 0:
                preview = item[:min(3, len(item))]
                print(f"{prefix}   First values: {preview}")
            print()
        elif isinstance(item, zarr.hierarchy.Group):
            print(f"{prefix}📁 {key}/")
            inspect_zarr_group(item, indent + 1)


def visualize_images(store, output_dir=None, num_samples=5):
    """Visualize images from the zarr dataset"""
    if 'data' not in store or 'img' not in store['data']:
        print("⚠️  No images found in dataset")
        return
    
    img_array = store['data/img']
    print(f"\n🖼️  Visualizing images...")
    print(f"Image array shape: {img_array.shape}")
    print(f"Image dtype: {img_array.dtype}")
    
    # Get episode ends if available
    episode_ends = None
    if 'meta' in store and 'episode_ends' in store['meta']:
        episode_ends = store['meta/episode_ends'][:]
    
    # Determine number of samples to show
    total_frames = img_array.shape[0]
    num_samples = min(num_samples, total_frames)
    
    # Sample frames evenly across the dataset
    if episode_ends is not None and len(episode_ends) > 0:
        # Sample from different episodes
        num_episodes = len(episode_ends)
        sample_episodes = np.linspace(0, num_episodes - 1, min(num_samples, num_episodes), dtype=int)
        
        sample_indices = []
        for ep_idx in sample_episodes:
            ep_start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            ep_end = episode_ends[ep_idx]
            # Take middle frame of episode
            sample_indices.append((ep_start + ep_end) // 2)
    else:
        # Sample evenly across all frames
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=(15, 3 * ((num_samples + 2) // 3)))
    gs = GridSpec(((num_samples + 2) // 3), 3, figure=fig)
    
    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        
        # Load image
        img = img_array[idx]
        
        # Handle different image formats
        if len(img.shape) == 3:
            # Single image (H, W, C) or (C, H, W)
            if img.shape[0] in [1, 3, 4]:  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))
            
            # Normalize if needed
            if img.dtype == np.uint8:
                display_img = img
            else:
                # Assume float in [0, 1] or need normalization
                if img.max() <= 1.0:
                    display_img = (img * 255).astype(np.uint8)
                else:
                    display_img = img.astype(np.uint8)
            
            # Handle single channel
            if display_img.shape[-1] == 1:
                display_img = display_img.squeeze(-1)
                ax.imshow(display_img, cmap='gray')
            else:
                ax.imshow(display_img)
        else:
            # Unexpected shape
            print(f"Warning: Unexpected image shape at index {idx}: {img.shape}")
            continue
        
        # Find which episode this frame belongs to
        episode_num = 0
        if episode_ends is not None:
            episode_num = np.searchsorted(episode_ends, idx)
        
        ax.set_title(f'Frame {idx} (Episode {episode_num})', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = Path(output_dir) / 'zarr_images.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Images saved to: {output_path}")
    else:
        plt.savefig('/tmp/zarr_images.png', dpi=150, bbox_inches='tight')
        print(f"✅ Images saved to: /tmp/zarr_images.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Inspect zarr dataset')
    parser.add_argument('zarr_path', type=str, help='Path to zarr dataset')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize images from dataset')
    parser.add_argument('--num-samples', type=int, default=6, help='Number of image samples to visualize')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save visualizations')
    args = parser.parse_args()
    
    zarr_path = Path(args.zarr_path)
    
    if not zarr_path.exists():
        print(f"❌ Error: Path {zarr_path} does not exist")
        return
    
    print(f"🔍 Inspecting zarr dataset: {zarr_path}\n")
    print("=" * 80)
    
    # Open the zarr store
    store = zarr.open(str(zarr_path), mode='r')
    
    # Print general info
    print(f"\n📦 Dataset Type: {type(store)}")
    
    if isinstance(store, zarr.hierarchy.Group):
        print(f"🗂️  Groups and Arrays:\n")
        inspect_zarr_group(store)
        
        # If this looks like a diffusion policy dataset, show episode info
        if 'meta' in store and 'episode_ends' in store['meta']:
            print("\n" + "=" * 80)
            print("📈 Episode Information:\n")
            episode_ends = store['meta/episode_ends'][:]
            print(f"Number of episodes: {len(episode_ends)}")
            print(f"Episode ends: {episode_ends}")
            
            # Calculate episode lengths
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            episode_lengths = episode_ends - episode_starts
            print(f"\nEpisode lengths: {episode_lengths}")
            print(f"Min length: {np.min(episode_lengths)}")
            print(f"Max length: {np.max(episode_lengths)}")
            print(f"Mean length: {np.mean(episode_lengths):.2f}")
            print(f"Total timesteps: {episode_ends[-1]}")
            
        # Show action and state info if available
        if 'data' in store:
            print("\n" + "=" * 80)
            print("🎮 Data Summary:\n")
            
            data_group = store['data']
            if 'action' in data_group:
                action = data_group['action']
                print(f"Actions: shape={action.shape}, dtype={action.dtype}")
                
            if 'state' in data_group:
                state = data_group['state']
                print(f"States: shape={state.shape}, dtype={state.dtype}")
                
            if 'img' in data_group:
                img = data_group['img']
                print(f"Images: shape={img.shape}, dtype={img.dtype}")
                
            if 'time' in data_group:
                time = data_group['time']
                print(f"Time: shape={time.shape}, dtype={time.dtype}")
                
    elif isinstance(store, zarr.core.Array):
        print(f"\n📊 Single Array:")
        print(f"   Shape: {store.shape}")
        print(f"   Dtype: {store.dtype}")
        print(f"   Chunks: {store.chunks}")
        print(f"   Size: {store.nbytes / 1024:.2f} KB")
    
    print("\n" + "=" * 80)
    print("✅ Inspection complete!")
    
    # Visualize images if requested
    if args.visualize:
        print("\n" + "=" * 80)
        visualize_images(store, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
