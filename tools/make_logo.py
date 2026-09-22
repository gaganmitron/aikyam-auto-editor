"""Generates the PLACEHOLDER Aikyam wordmark (aikyam_video/assets/aikyam-logo.png). Replace the PNG with the real logo."""
from PIL import Image, ImageDraw, ImageFont
W, H = 600, 160
im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 92)
d.ellipse((6, 30, 106, 130), outline=(255, 190, 40, 235), width=8)        # simple lamp-flame glyph ring
d.polygon([(56, 42), (76, 82), (56, 118), (36, 82)], fill=(255, 190, 40, 235))
d.text((124, 22), "Aikyam", font=f, fill=(255, 255, 255, 240), stroke_width=3, stroke_fill=(0, 0, 0, 200))
im.save("aikyam_video/assets/aikyam-logo.png")
