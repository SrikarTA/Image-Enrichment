from PIL import Image, ImageDraw, ImageFont
import os
import random


def add_cation(prompt, img, path, out_path):
    # Open the image
    image = Image.open(path+img)  # Replace with the path to your image

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define text to be added
    text = prompt

    # Load a custom TrueType font with the desired size
    font_size = 36
    font_path = "Pacifico.ttf"  # Replace with the filename of your TTF font
    font = ImageFont.truetype(font_path, font_size)

    # Get text size to calculate text positioning
    text_width, text_height = draw.textsize(text, font=font)
    image_width, image_height = image.size
    x_position = (image_width - text_width) // 2
    y_position = int(image_height * 0.02)  # Placing text at the top 20% of the image

    # Set text color to white
    text_fill_color = (255, 255, 255)  # RGB values for white

    # Set text outline color to black
    text_outline_color = (0, 0, 0)  # RGB values for black

    # Create a black outline by drawing the text multiple times with a slight offset
    outline_width = 2  # Adjust the outline width as needed
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            draw.text((x_position + dx, y_position + dy), text, font=font, fill=text_outline_color)

    # Draw the central text in white on top of the outline
    draw.text((x_position, y_position), text, font=font, fill=text_fill_color)

    # Save the edited image
    image.save(out_path+img)  # Save to a different file to preserve the original
    
def add_logos(img,path,main_folder, out_path):
    # Set the path to the folder containing your logo images
    logo_folder = main_folder
    
    # List all files in the folder
    logo_files = [f for f in os.listdir(logo_folder)]
    
    # Randomly select up to 4 logos, or select all if there are fewer than 4
    selected_logos = logo_files if len(logo_files) <= 4 else random.sample(logo_files, 4)
    
    # Load your main image
    image_path = path+img  # Replace with your image file path (in PNG format)
    image = Image.open(image_path)
    
    # Get the dimensions of the main image
    image_width, image_height = image.size
    
    # Calculate the target width and height for the logos (10% of the image size)
    target_width = image_width // 10
    target_height = image_height // 10
    
    # Initialize the horizontal and vertical positions for the logos
    x_position = image_width - target_width
    y_position = image_height - (target_height * len(selected_logos))
    
    # Create a blank white background image
    background = Image.new("RGBA", (target_width, target_height), (255, 255, 255, 255))
    
    # Load and paste the selected logos onto the white background
    for logo_file in selected_logos:
        logo_path = os.path.join(logo_folder, logo_file)
        logo = Image.open(logo_path)
    
        # Convert the logo to RGBA mode if it's not already
        if logo.mode != "RGBA":
            logo = logo.convert("RGBA")
    
        # Resize the logo to the target size
        logo = logo.resize((target_width, target_height))
    
        # Create a blank white background image with the same size as the logo
        background = Image.new("RGBA", (target_width, target_height), (255, 255, 255, 255))
    
        # Paste the resized logo onto the white background
        background.paste(logo, (0, 0), logo)
    
        # Paste the white background with the logo onto the main image at the bottom-right
        image.paste(background, (x_position, y_position), background)
    
        # Update the vertical position for the next logo
        y_position += target_height
            
            
    # Add PURINA logo        
    # Load the logo image
    logo_image_path = "/home/suchitra/kellogs/nestle_purina_image_logo.png"  # Replace with your logo image file path
    logo_image = Image.open(logo_image_path)
    
    
    main_image = image
    # Get the dimensions of the main image
    main_image_width, main_image_height = main_image.size
    
    # Calculate the dimensions for the resized logo (10% of the main image)
    logo_width = int(main_image_width * 0.25)
    logo_height = int(main_image_height * 0.1)
    
    # Resize the logo image
    logo_image = logo_image.resize((logo_width, logo_height))
    
    # Create a new blank image with a white background
    background = Image.new("RGBA", (main_image_width, main_image_height), (255, 255, 255, 255))
    
    # Paste the main image onto the white background
    background.paste(main_image, (0, 0))
    
    # Calculate the position to paste the logo in the bottom-left corner
    logo_x = 0
    logo_y = main_image_height - logo_height
    
    # Paste the resized logo in the bottom-left corner
    background.paste(logo_image, (logo_x, logo_y))
    
    # Save the final image
    background.save(out_path+img)
