from PIL import Image, ImageDraw
import numpy as np


def _to_pil(img_uint8_3ch: np.ndarray) -> Image.Image:
    return Image.fromarray(img_uint8_3ch.astype(np.uint8))


def _draw_title(im: Image.Image, title: str) -> Image.Image:
    im = im.copy()
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, im.size[0], 26], fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return im


def make_2x3_mosaic(
    rgb_imgs, depth_rgb_imgs, csfnet_rgb_imgs, ours_rgb_imgs,
    titles_top, titles_bottom
) -> Image.Image:
    """Build a 2x3 mosaic:
    Top row (6 tiles): RGB(night,rain,snow) + Depth(night,rain,snow)
    Bottom row (6 tiles): CSFNet(night,rain,snow) + Ours(night,rain,snow)
    """
    assert len(rgb_imgs) == 3 and len(depth_rgb_imgs) == 3 and len(csfnet_rgb_imgs) == 3 and len(ours_rgb_imgs) == 3

    top = [*rgb_imgs, *depth_rgb_imgs]
    bot = [*csfnet_rgb_imgs, *ours_rgb_imgs]
    assert len(top) == 6 and len(bot) == 6

    pil_top = [_to_pil(x) for x in top]
    pil_bot = [_to_pil(x) for x in bot]

    tile_w, tile_h = pil_top[0].size
    pil_top = [im.resize((tile_w, tile_h), Image.BILINEAR) for im in pil_top]
    pil_bot = [im.resize((tile_w, tile_h), Image.BILINEAR) for im in pil_bot]

    pil_top = [_draw_title(im, titles_top[i]) for i, im in enumerate(pil_top)]
    pil_bot = [_draw_title(im, titles_bottom[i]) for i, im in enumerate(pil_bot)]

    canvas = Image.new("RGB", (tile_w * 6, tile_h * 2), (255, 255, 255))
    for i, im in enumerate(pil_top):
        canvas.paste(im, (i * tile_w, 0))
    for i, im in enumerate(pil_bot):
        canvas.paste(im, (i * tile_w, tile_h))
    return canvas
