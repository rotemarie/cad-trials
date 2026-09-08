"""Run cadrille image-branch inference on arbitrary image files (PNG/JPG).

The repo's test.py only accepts .stl meshes (it renders them to images internally).
This script reuses cadrille's own `collate` + generation path, but feeds it a
dataset that reads image files directly.

Usage:
    python infer_image.py drawing1.png drawing2.png --num-samples 5 --temperature 0.8
"""

import os
import argparse
from functools import partial

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

from cadrille import Cadrille, collate


class ImageFolderDataset(Dataset):
    """Yields items shaped like cadrille's eval dataset, but from image files."""

    def __init__(self, paths, img_size=128, border=3, pad_color=(255, 255, 255), raw=False):
        self.paths = list(paths)
        self.img_size = img_size
        self.border = border
        self.pad_color = pad_color
        self.raw = raw  # if True, feed the image as-is (e.g. output of render_mesh.py)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if not self.raw:
            # letterbox to a square, then add the black border the model saw in training
            img = ImageOps.pad(img, (self.img_size, self.img_size), color=self.pad_color)
            img = ImageOps.expand(img, border=self.border, fill='black')
        stem = os.path.splitext(os.path.basename(self.paths[i]))[0]
        return {'video': [img], 'description': 'Generate cadquery code', 'file_name': stem}


def load_model(checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for attn in ('flash_attention_2', 'sdpa'):
        try:
            model = Cadrille.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn,
                device_map=device).eval()
            print(f'loaded {checkpoint_path} on {device} with attn={attn}')
            return model
        except (ImportError, ValueError) as e:
            print(f'attn={attn} unavailable ({e}); falling back')
    raise RuntimeError('could not load model')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+', help='image files to reconstruct')
    ap.add_argument('--checkpoint-path', default='maksimko123/cadrille')
    ap.add_argument('--py-path', default='./work_dirs/img_py')
    ap.add_argument('--num-samples', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--img-size', type=int, default=128)
    ap.add_argument('--raw', action='store_true',
                    help='feed images unmodified (use for render_mesh.py output)')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=768)
    args = ap.parse_args()

    os.makedirs(args.py_path, exist_ok=True)

    model = load_model(args.checkpoint_path)
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen2-VL-2B-Instruct',
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left')

    dataset = ImageFolderDataset(args.images, img_size=args.img_size, raw=args.raw)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    do_sample = args.temperature > 0
    for s in range(args.num_samples):
        for batch in loader:
            generated = model.generate(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                point_clouds=batch['point_clouds'].to(model.device),
                is_pc=batch['is_pc'].to(model.device),
                is_img=batch['is_img'].to(model.device),
                pixel_values_videos=batch['pixel_values_videos'].to(model.device)
                    if batch.get('pixel_values_videos') is not None else None,
                video_grid_thw=batch['video_grid_thw'].to(model.device)
                    if batch.get('video_grid_thw') is not None else None,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None)
            trimmed = [o[len(i):] for i, o in zip(batch['input_ids'], generated)]
            codes = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for stem, code in zip(batch['file_name'], codes):
                fn = os.path.join(args.py_path, f'{stem}+s{s}.py')
                with open(fn, 'w') as f:
                    f.write(code)
                print(f'wrote {fn}  ({len(code)} chars)')


if __name__ == '__main__':
    main()
