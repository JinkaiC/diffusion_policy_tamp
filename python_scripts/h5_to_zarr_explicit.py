#!/usr/bin/env python3
"""
Convert per-frame .h5 files + external images into a zarr with `data/img/<episode>` and `data/action/<episode>`.

This script uses explicit action keys discovered from `testh5.py` output. By default it will
concatenate `joint_control` and `gripper_control` into a fixed-size action vector, but you can
specify any comma-separated list of dataset keys present in each .h5 file.

Usage example:
    python utils/h5_to_zarr_explicit.py --record_dir /path/to/records --out_zarr ./output.zarr \
        --action_keys joint_control,gripper_control --resize 96 96 --normalize 0-1 --layout array \
        --time_chunk 161 --overwrite --write_meta --compressor_name zstd --clevel 5

Notes:
- Prefers external images next to each .h5 (e.g., `0.jpg`, `0.png`, `0_depth.png`, `0_depth_render.jpg`).
- Skips frames missing images or missing any specified action key (prints a warning(already ignored in line 147)).
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


def make_compressor():
    try:
        comp = zarr.codecs.Blosc(cname='lz4', clevel=5, shuffle=2)
    except Exception:
        try:
            import numcodecs
            comp = numcodecs.Blosc(cname='lz4', clevel=5, shuffle=2)
        except Exception:
            comp = None
    return comp


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


def convert(record_dir, out_zarr, action_keys, pattern='*', overwrite=False, no_timestamps=False,
            resize=None, normalize='0-1', layout='array', time_chunk=161, write_meta=False,
            compressor_name='zstd', clevel=5):
    root = get_store(out_zarr, overwrite=overwrite)
    data = root.require_group('data')
    # We'll either write data/img and data/action as arrays (layout='array')
    # or as groups with per-episode datasets (layout='group').
    img_group = data.require_group('img')
    action_group = data.require_group('action')

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

    # build compressor based on requested codec
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
    all_acts = []
    episode_ends = []

    for ep_dir in tqdm(episode_dirs, desc='episodes'):
        h5_files = collect_h5_files(ep_dir)
        imgs = []
        acts = []
        for fpath in h5_files:
            img = read_image_for_frame(fpath)
            with h5py.File(fpath, 'r') as hf:
                # if no external image, try common image keys inside h5
                if img is None:
                    # look for dataset keys likely to be image
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
                    # cv2.resize expects (width,height) as (cols,rows) order via (W,H)
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

                # collect action pieces
                pieces = []
                missing_key = False
                for k in action_keys:
                    if k not in hf:
                        missing_key = True
                        break
                    arr = np.asarray(hf[k][()])
                    arr = arr.reshape(-1)
                    pieces.append(arr)
                if missing_key:
                    print(f"Warning: skipping {fpath} (missing action key among {action_keys})")
                    continue

                action_vec = np.concatenate(pieces, axis=0)
                imgs.append(img)
                acts.append(action_vec)

        if len(imgs) == 0:
            print(f"No usable frames in episode {ep_dir}; skipping")
            continue

        imgs_np = np.stack(imgs, axis=0)

        # ensure all action vectors in this episode have the same length by padding
        max_len_ep = max(a.shape[0] for a in acts)
        acts_padded = [np.pad(a, (0, max_len_ep - a.shape[0]), mode='constant', constant_values=0) for a in acts]
        acts_np = np.stack(acts_padded, axis=0)

        if layout == 'array':
            # accumulate across episodes
            start = len(all_imgs)
            all_imgs.append(imgs_np)
            all_acts.append(acts_np)
            episode_ends.append(start + imgs_np.shape[0])
        else:
            # per-episode group dataset
            ep_name = os.path.basename(os.path.normpath(ep_dir))
            if ep_name in img_group:
                del img_group[ep_name]
            if ep_name in action_group:
                del action_group[ep_name]

            img_ds = img_group.create_dataset(ep_name, shape=imgs_np.shape, dtype=imgs_np.dtype,
                                              chunks=(1,)+imgs_np.shape[1:], compressor=compressor)
            img_ds[:] = imgs_np
            if acts_np.ndim == 1:
                acts_np = acts_np.reshape((-1, 1))
            action_ds = action_group.create_dataset(ep_name, shape=acts_np.shape, dtype=acts_np.dtype,
                                                    chunks=(min(time_chunk, acts_np.shape[0]),)+acts_np.shape[1:], compressor=compressor)
            action_ds[:] = acts_np

    # if array layout requested, concatenate and write as single arrays
    if layout == 'array' and len(all_imgs) > 0:
        imgs_cat = np.concatenate(all_imgs, axis=0)

        # ensure all episode-level action arrays have the same second-dimension
        max_len_global = max(arr.shape[1] for arr in all_acts)
        all_acts_padded = [np.pad(arr, ((0,0),(0,max_len_global - arr.shape[1])), mode='constant', constant_values=0) for arr in all_acts]
        acts_cat = np.concatenate(all_acts_padded, axis=0)

        # remove existing top-level arrays if overwriting
        if 'img' in data:
            try:
                del data['img']
            except Exception:
                pass
        if 'action' in data:
            try:
                del data['action']
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

        if acts_cat.ndim == 1:
            acts_cat = acts_cat.reshape((-1, 1))
        action_ds = data.create_dataset('action', shape=acts_cat.shape, dtype=acts_cat.dtype,
                                        chunks=(time_chunk_val, acts_cat.shape[1]), compressor=compressor)
        action_ds[:] = acts_cat

        # write meta/episode_ends if requested
        if write_meta:
            meta = root.require_group('meta')
            # episode_ends as cumulative lengths
            ep_ends_arr = np.array(episode_ends, dtype=np.int64)
            if 'episode_ends' in meta:
                del meta['episode_ends']
            meta.create_dataset('episode_ends', data=ep_ends_arr)

        print(f'Wrote zarr to {out_zarr}. Episodes (concatenated): {len(episode_ends)}, frames total: {t}')
    else:
        print(f'Wrote zarr to {out_zarr}. Episodes (per-episode group count): {len(episode_dirs)}')


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--record_dir', required=True)
        parser.add_argument('--out_zarr', required=True)
        parser.add_argument('--action_keys', default='joint_control,gripper_control',
                help='comma-separated action dataset keys to concatenate (default: joint_control,gripper_control)')
        parser.add_argument('--pattern', default='*', help='glob pattern for image files (unused for explicit reader)')
        parser.add_argument('--overwrite', action='store_true')
        parser.add_argument('--no_timestamps', action='store_true', help='do not treat record_dir children as timestamps')
        parser.add_argument('--resize', nargs=2, type=int, default=None, help='resize to W H (e.g. --resize 96 96)')
        parser.add_argument('--normalize', choices=['0-1','-1-1','none'], default='0-1', help='normalize image values')
        parser.add_argument('--layout', choices=['array','group'], default='array', help='write layout: array (concatenate) or group (per-episode)')
        parser.add_argument('--time_chunk', type=int, default=161, help='time chunk size for zarr arrays')
        parser.add_argument('--write_meta', action='store_true', help='write meta/episode_ends when using array layout')
        parser.add_argument('--compressor_name', default='zstd', help='Blosc compressor name (zstd or lz4)')
        parser.add_argument('--clevel', type=int, default=5, help='Blosc compression level')
        args = parser.parse_args()

        action_keys = [k.strip() for k in args.action_keys.split(',') if k.strip()]
        resize = tuple(args.resize) if args.resize is not None else None
        convert(args.record_dir, args.out_zarr, action_keys=action_keys, pattern=args.pattern,
            overwrite=args.overwrite, no_timestamps=args.no_timestamps,
            resize=resize, normalize=args.normalize, layout=args.layout,
            time_chunk=args.time_chunk, write_meta=args.write_meta,
            compressor_name=args.compressor_name, clevel=args.clevel)


if __name__ == '__main__':
    main()
