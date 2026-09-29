"""
PWA Icon Generator for PLOS Predictor App.
Generates 192x192 and 512x512 PNG app icons with healthcare branding.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def create_pwa_icon(size=192, filename="static/images/icon-192.png"):
    # Create gradient/solid background
    img = Image.new("RGBA", (size, size), (15, 118, 110, 255)) # Teal #0f766e
    draw = ImageDraw.Draw(img)
    
    # Draw rounded background circle/square
    padding = int(size * 0.08)
    draw.rounded_rectangle(
        [(padding, padding), (size - padding, size - padding)],
        radius=int(size * 0.2),
        fill=(2, 132, 199, 255) # Secondary blue #0284c7
    )
    
    # Draw white medical cross symbol in center
    cx, cy = size // 2, size // 2
    arm_len = int(size * 0.24)
    thickness = int(size * 0.09)
    
    # Horizontal arm
    draw.rectangle(
        [(cx - arm_len, cy - thickness // 2), (cx + arm_len, cy + thickness // 2)],
        fill=(255, 255, 255, 255)
    )
    # Vertical arm
    draw.rectangle(
        [(cx - thickness // 2, cy - arm_len), (cx + thickness // 2, cy + arm_len)],
        fill=(255, 255, 255, 255)
    )
    
    # Draw inner circle accent
    circle_r = int(size * 0.06)
    draw.ellipse(
        [(cx - circle_r, cy - circle_r), (cx + circle_r, cy + circle_r)],
        fill=(15, 118, 110, 255)
    )
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename, "PNG")
    print(f"[+] Saved PWA icon ({size}x{size}) -> '{filename}'")

if __name__ == "__main__":
    create_pwa_icon(192, "static/images/icon-192.png")
    create_pwa_icon(512, "static/images/icon-512.png")
