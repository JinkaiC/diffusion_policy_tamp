#!/usr/bin/env python3
"""
Convert per-frame .h5 files + external images into diffusion-policy-standard zarr.

This script reads ee_states and gripper_control from .h5 files and merges them into a single
action vector compatible with diffusion-policy training.

Output structure:
  data/img: observations (T, H, W, 3) float32
  data/state: states (T, 7) float32 = [ee_states (7-dim)]
  data/action: actions (T, 8) float32 = [ee_states (7-dim) + gripper_control (1-dim)]
  meta/episode_ends: episode boundaries

Usage example:
    python utils/h5_to_zarr_ee_states.py \
        --record_dir datasets/records/banana_plate/20251201_154925_530/00000 \
        --out_zarr outputs/banana_plate_ee_states.zarr \
        --resize 96 96 --normalize 0-1 --layout array \
        --time_chunk 161 --overwrite --write_meta --compressor_name zstd --clevel 5

Notes:
- Prefers external images next to each .h5 (e.g., `0.jpg`, `0.png`, `0_depth.png`, `0_depth_render.jpg`).
- Skips frames missing images or missing ee_states/gripper_control keys.
- ee_states and gripper_control are automatically concatenated into a single action array.
"""
import argparse
import os
import glob
import h5py
import numpy as np
import zarr
import imageio.v2 as imageio
import cv2
from tqdm import tqdm


def get_store(out_zarr, overwrite=False):
    if hasattr(zarr, 'storage'):
        st = zarr.storage
        if hasattr(st, 'DirectoryStore'):
            store = st.DirectoryStore(out_zarr)
        elif hasattr(st, 'LocalStore'):
            store = st.LocalStore(out_zarr)
        else:
            if hasattr(zarr, 'DirectoryStore'):
                store = zarr.DirectoryStore(out_zarr)
            else:
                raise RuntimeError('Unsupported zarr storage API: no DirectoryStore/LocalStore found')
    elif hasattr(zarr, 'DirectoryStore'):
        store = zarr.DirectoryStore(out_zarr)
    else:
        raise RuntimeError('zarr DirectoryStore API not found; please install a compatible zarr version')
    root = zarr.group(store=store, overwrite=overwrite)
    return root


def collect_h5_files(ep_dir):
    files = [f for f in os.listdir(ep_dir) if f.endswith('.h5')]
    try:
        files_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    except Exception:
        files_sorted = sorted(files)
    return [os.path.join(ep_dir, f) for f in files_sorted]


IMG_CANDIDATES = ["{base}.jpg", "{base}.png", "{base}_depth_render.jpg", "{base}_depth_render.png", "{base}_depth.png", "{base}_depth.jpg"]


def read_image_for_frame(fpath):
    base = os.path.splitext(fpath)[0]
    for tmpl in IMG_CANDIDATES:
        p = tmpl.format(base=base)
        if os.path.exists(p):
            try:
                im = imageio.imread(p)
                return np.asarray(im)
            except Exception:
                continue
    return None


def convert(record_dir, out_zarr, pattern='*', overwrite=False, no_timestamps=False,
            resize=None, normalize='0-1', layout='array', time_chunk=161, write_meta=False,
            compressor_name='zstd', clevel=5):
    """
    Convert .h5 per-frame data + images into diffusion-policy-standard zarr.
    
    Merges ee_states and gripper_control into a single action vector.
    
    Args:
        record_dir: directory containing .h5 files (or subdirectories with timestamps/episodes)
        out_zarr: output zarr path
        pattern: unused (kept for consistency)
        overwrite: whether to overwrite existing zarr
        no_timestamps: if True, treat record_dir children as episodes directly; else look for timestamp subdirs
        resize: tuple (W, H) to resize images
        normalize: '0-1', '-1-1', or 'none' for image normalization
        layout: 'array' to concatenate episodes, 'group' to keep per-episode groups
        time_chunk: time dimension chunk size for zarr arrays
        write_meta: whether to write meta/episode_ends for array layout
        compressor_name: Blosc compressor name ('zstd', 'lz4', etc.)
        clevel: compression level
    """
    root = get_store(out_zarr, overwrite=overwrite)
    data = root.require_group('data')

    if not no_timestamps:
        # collect episode dirs under timestamps
        timestamps = sorted([os.path.join(record_dir, t) for t in os.listdir(record_dir)])
        episode_dirs = []
        for t in timestamps:
            if os.path.isdir(t):
                subs = sorted([os.path.join(t, s) for s in os.listdir(t)])
                episode_dirs.extend([p for p in subs if os.path.isdir(p)])
    else:
        episode_dirs = sorted([os.path.join(record_dir, x) for x in os.listdir(record_dir) if os.path.isdir(os.path.join(record_dir, x))])

    if len(episode_dirs) == 0:
        # treat record_dir itself as episode
        if any(f.endswith('.h5') for f in os.listdir(record_dir)):
            episode_dirs = [record_dir]

    # build compressor
    try:
        comp = zarr.codecs.Blosc(cname=compressor_name, clevel=clevel, shuffle=2)
    except Exception:
        try:
            import numcodecs
            comp = numcodecs.Blosc(cname=compressor_name, clevel=clevel, shuffle=2)
        except Exception:
            comp = None

    compressor = comp

    all_imgs = []
    all_states = []  # ee_states only
    all_actions = []  # merged ee_states + gripper_control
    episode_ends = []

    for ep_dir in tqdm(episode_dirs, desc='episodes'):
        h5_files = collect_h5_files(ep_dir)
        imgs = []
        states_list = []
        actions_list = []
        
        for fpath in h5_files:
            img = read_image_for_frame(fpath)
            
            with h5py.File(fpath, 'r') as hf:
                # if no external image, try common image keys inside h5
                if img is None:
                    for k in ['img', 'rgb', 'image']:
                        if k in hf:
                            try:
                                img = np.asarray(hf[k][()])
                                break
                            except Exception:
                                img = None

                if img is None:
                    #print(f"Warning: skipping {fpath} (no image)")
                    continue

                # prepare img (resize/normalize)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                # resize if requested
                if resize is not None:
                    w, h = resize
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                # normalize / dtype
                if normalize == '0-1':
                    img = img.astype(np.float32) / 255.0
                elif normalize == '-1-1':
                    img = img.astype(np.float32) / 127.5 - 1.0
                else:
                    # keep original dtype (likely uint8)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.01:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)

                # read ee_states
                if 'ee_states' not in hf:
                    print(f"Warning: skipping {fpath} (missing ee_states)")
                    continue
                ee_states = np.asarray(hf['ee_states'][()])
                ee_states = ee_states.reshape(-1)

                # read gripper_control
                if 'gripper_control' not in hf:
                    print(f"Warning: skipping {fpath} (missing gripper_control)")
                    continue
                gripper_control = np.asarray(hf['gripper_control'][()])
                gripper_control = gripper_control.reshape(-1)

                # merge ee_states and gripper_control into single action vector
                action = np.concatenate([ee_states, gripper_control], axis=0)

                # success: append all
                imgs.append(img)
                states_list.append(ee_states)
                actions_list.append(action)

        if len(imgs) == 0:
            print(f"No usable frames in episode {ep_dir}; skipping")
            continue

        imgs_np = np.stack(imgs, axis=0)

        # ensure all state vectors in this episode have the same length by padding
        max_len_state = max(s.shape[0] for s in states_list)
        states_padded = [np.pad(s, (0, max_len_state - s.shape[0]), mode='constant', constant_values=0) for s in states_list]
        states_np = np.stack(states_padded, axis=0)

        # ensure all action vectors in this episode have the same length by padding
        max_len_action = max(a.shape[0] for a in actions_list)
        actions_padded = [np.pad(a, (0, max_len_action - a.shape[0]), mode='constant', constant_values=0) for a in actions_list]
        actions_np = np.stack(actions_padded, axis=0)

        if layout == 'array':
            # accumulate across episodes
            start = len(all_imgs)
            all_imgs.append(imgs_np)
            all_states.append(states_np)
            all_actions.append(actions_np)
            episode_ends.append(start + imgs_np.shape[0])
        else:
            # per-episode group dataset
            ep_name = os.path.basename(os.path.normpath(ep_dir))
            if ep_name in data:
                try:
                    del data[ep_name]
                except Exception:
                    pass

            # create separate groups for per-episode layout
            ep_group = data.require_group(ep_name)
            
            img_ds = ep_group.create_dataset('img', shape=imgs_np.shape, dtype=imgs_np.dtype,
                                            chunks=(1,)+imgs_np.shape[1:], compressor=compressor)
            img_ds[:] = imgs_np
            
            state_ds = ep_group.create_dataset('state', shape=states_np.shape, dtype=np.float32,
                                               chunks=(min(time_chunk, states_np.shape[0]), states_np.shape[1]), compressor=compressor)
            state_ds[:] = states_np.astype(np.float32)
            
            action_ds = ep_group.create_dataset('action', shape=actions_np.shape, dtype=np.float32,
                                               chunks=(min(time_chunk, actions_np.shape[0]), actions_np.shape[1]), compressor=compressor)
            action_ds[:] = actions_np.astype(np.float32)

    # if array layout requested, concatenate and write as single arrays
    if layout == 'array' and len(all_imgs) > 0:
        imgs_cat = np.concatenate(all_imgs, axis=0)

        # ensure all episode-level state arrays have the same second-dimension
        max_len_state_global = max(arr.shape[1] for arr in all_states)
        all_states_padded = [np.pad(arr, ((0,0),(0,max_len_state_global - arr.shape[1])), mode='constant', constant_values=0) for arr in all_states]
        states_cat = np.concatenate(all_states_padded, axis=0)

        # ensure all episode-level action arrays have the same second-dimension
        max_len_action_global = max(arr.shape[1] for arr in all_actions)
        all_actions_padded = [np.pad(arr, ((0,0),(0,max_len_action_global - arr.shape[1])), mode='constant', constant_values=0) for arr in all_actions]
        actions_cat = np.concatenate(all_actions_padded, axis=0)

        # remove existing top-level arrays if overwriting
        for key in ['img', 'state', 'action']:
            if key in data:
                try:
                    del data[key]
                except Exception:
                    pass

        # compute chunks
        t = imgs_cat.shape[0]
        h = imgs_cat.shape[1]
        w = imgs_cat.shape[2]
        c = imgs_cat.shape[3]
        time_chunk_val = min(time_chunk, t)

        img_ds = data.create_dataset('img', shape=imgs_cat.shape, dtype=imgs_cat.dtype,
                                     chunks=(time_chunk_val, h, w, c), compressor=compressor)
        img_ds[:] = imgs_cat

        state_ds = data.create_dataset('state', shape=states_cat.shape, dtype=np.float32,
                                       chunks=(time_chunk_val, states_cat.shape[1]), compressor=compressor)
        state_ds[:] = states_cat.astype(np.float32)

        action_ds = data.create_dataset('action', shape=actions_cat.shape, dtype=np.float32,
                                        chunks=(time_chunk_val, actions_cat.shape[1]), compressor=compressor)
        action_ds[:] = actions_cat.astype(np.float32)

        # write meta/episode_ends if requested
        if write_meta:
            meta = root.require_group('meta')
            ep_ends_arr = np.array(episode_ends, dtype=np.int64)
            if 'episode_ends' in meta:
                del meta['episode_ends']
            ds = meta.create_dataset('episode_ends', shape=ep_ends_arr.shape, dtype=ep_ends_arr.dtype)
            ds[:] = ep_ends_arr

        print(f'Wrote zarr to {out_zarr}. Episodes (concatenated): {len(episode_ends)}, frames total: {t}')
        print(f'  data/img shape: {imgs_cat.shape}, dtype: {imgs_cat.dtype}')
        print(f'  data/state shape: {states_cat.shape}, dtype: float32')
        print(f'    └─ [ee_states (7-dim)]')
        print(f'  data/action shape: {actions_cat.shape}, dtype: float32')
        print(f'    └─ [ee_states (7-dim) + gripper_control (1-dim)]')
    else:
        print(f'Wrote zarr to {out_zarr}. Episodes (per-episode group count): {len(episode_dirs)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_dir', required=True, help='root directory with per-frame .h5 files')
    parser.add_argument('--out_zarr', required=True, help='output zarr path')
    parser.add_argument('--pattern', default='*', help='glob pattern for image files (unused)')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing zarr')
    parser.add_argument('--no_timestamps', action='store_true', help='do not treat record_dir children as timestamps')
    parser.add_argument('--resize', nargs=2, type=int, default=None, help='resize to W H (e.g. --resize 96 96)')
    parser.add_argument('--normalize', choices=['0-1','-1-1','none'], default='0-1', help='normalize image values')
    parser.add_argument('--layout', choices=['array','group'], default='array', help='write layout: array (concatenate) or group (per-episode)')
    parser.add_argument('--time_chunk', type=int, default=161, help='time chunk size for zarr arrays')
    parser.add_argument('--write_meta', action='store_true', help='write meta/episode_ends when using array layout')
    parser.add_argument('--compressor_name', default='zstd', help='Blosc compressor name (zstd or lz4)')
    parser.add_argument('--clevel', type=int, default=5, help='Blosc compression level')
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize is not None else None
    convert(args.record_dir, args.out_zarr, pattern=args.pattern,
            overwrite=args.overwrite, no_timestamps=args.no_timestamps,
            resize=resize, normalize=args.normalize, layout=args.layout,
            time_chunk=args.time_chunk, write_meta=args.write_meta,
            compressor_name=args.compressor_name, clevel=args.clevel)


if __name__ == '__main__':
    main()
