from PIL import Image, ImageDraw
import os

def create_icon(size=1024):
    # Create a new image with a white background
    icon = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(icon)
    
    # Colors
    primary_color = (64, 156, 255)    # Blue for water/flux
    accent_color = (255, 128, 64)     # Orange for music/energy
    
    # Calculate dimensions
    padding = size // 8
    center = size // 2
    radius = (size - 2 * padding) // 2
    
    # Draw the main circular gradient (representing flux/flow)
    for r in range(radius, 0, -1):
        alpha = int(255 * (r / radius))
        color = (*primary_color, alpha)
        draw.ellipse(
            [center - r, center - r, center + r, center + r],
            fill=color
        )
    
    # Draw the music wave pattern
    wave_height = size // 4
    wave_width = size // 16
    wave_y = center + radius // 2
    
    # Draw three wave bars with varying heights
    heights = [0.8, 1.0, 0.6]  # Relative heights for visual interest
    for i, height in enumerate(heights):
        x = center + (i - 1) * (wave_width * 2)
        h = int(wave_height * height)
        draw.rounded_rectangle(
            [x - wave_width//2, wave_y - h//2,
             x + wave_width//2, wave_y + h//2],
            radius=wave_width//4,
            fill=(*accent_color, 200)
        )
    
    # Save in different sizes for macOS
    if not os.path.exists('resources'):
        os.makedirs('resources')
    
    # Save the main icon
    icon.save('resources/icon.png')
    
    # Create .icns compatible sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        resized = icon.resize((s, s), Image.Resampling.LANCZOS)
        resized.save(f'resources/icon_{s}x{s}.png')
    
    print("Icon files generated in resources/ directory")

if __name__ == '__main__':
    create_icon() 