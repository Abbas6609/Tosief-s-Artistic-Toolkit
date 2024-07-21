# Necessary libraries and imports
# created by Tosief Abbas, Dec 2023
# For further suggestions/contact: Tosiefabbas@gmail.com

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL import ImageOps
import streamlit as st
from rembg import remove
import io
import base64
import random
import torch
import torch.nn as nn
from torchvision import transforms

# Initialize ONNX runtime session for the cartoonizer model
_sess_options = ort.SessionOptions()
_sess_options.intra_op_num_threads = os.cpu_count()
MODEL_SESS = ort.InferenceSession(
    "cartoonizer.onnx", _sess_options, providers=["CPUExecutionProvider"]
)

# Define normalization layer
norm_layer = nn.InstanceNorm2d

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# Define the Generator Model
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        # Residual blocks
        model2 = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out

# Load the models
model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))
model2.eval()


# Preprocess and inference functions for cartoonizing
def preprocess_image(image: Image) -> np.ndarray:
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert PIL Image to a NumPy array
    image = np.array(image)

    # Convert RGB to BGR format for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    image = image.astype(np.float32) / 127.5 - 1
    return np.expand_dims(image, axis=0)

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    beta = brightness
    alpha = contrast / 127.5 + 1  # Contrast adjustment factor
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img

def apply_blur_and_sharpen(img, blur_ksize=5, sharpen_ksize=3):
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    return sharpened

def fill_sketch(img):
    # Convert PIL Image to a NumPy array
    img_array = np.array(img)

    # Check if the image is already in grayscale
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        # The image is in grayscale, so we don't need to convert it
        gray_img = img_array
    else:
        # Convert image to grayscale
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Convert grayscale image to binary
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank colored image with the same dimensions
    colored_img = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)

    # Fill each contour with a random color
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(colored_img, [contour], -1, color, thickness=cv2.FILLED)

    return Image.fromarray(colored_img)

def enhanced_fill_sketch(img, use_original_colors=True, detail_enhancement=False):
    # Convert PIL Image to a NumPy array
    img_array = np.array(img)

    # Convert to grayscale if not already
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array.copy()

    # Enhance details if required
    if detail_enhancement:
        gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        edges = cv2.Canny(gray_img, 30, 100)
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        _, binary_img = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image with the same dimensions
    filled_img = np.zeros_like(img_array)

    # Fill each contour
    for contour in contours:
        if use_original_colors and len(img_array.shape) == 3:
            mask = np.zeros_like(gray_img, dtype=np.uint8)  # Ensure mask is of correct type
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mean_color = cv2.mean(img_array, mask=mask)
            color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.drawContours(filled_img, [contour], -1, color, thickness=cv2.FILLED)

    return Image.fromarray(filled_img)


def inference(image: np.ndarray, cartoon_intensity=1) -> Image:
    # Preprocessing
    image = preprocess_image(image)

    # Cartoonization with the ONNX model
    results = MODEL_SESS.run(None, {"input_photo:0": image})
    output = (np.squeeze(results[0]) + 1.0) * 127.5

    # Adjusting contrast and brightness based on intensity
    contrast_factor = 1.0 + (cartoon_intensity - 1) * 0.1  # Example calculation
    brightness_factor = 1.0 + (cartoon_intensity - 1) * 0.05  # Example calculation
    output = adjust_brightness_contrast(output, brightness=brightness_factor, contrast=contrast_factor)

    # Post-processing
    output = apply_blur_and_sharpen(output)
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return Image.fromarray(output)


# Sketch image function
def sketch_image_pil(pil_img, mode="pencil", intensity=13):
    # Ensure intensity is an odd number
    if intensity % 2 == 0:
        intensity += 1

    img = np.array(pil_img.convert('RGB'), dtype=np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_img_gray = cv2.bitwise_not(img_gray)
    blur_img = cv2.GaussianBlur(inv_img_gray, (intensity, intensity), 0)
    inv_blur_img = cv2.bitwise_not(blur_img)
    sketch = cv2.divide(img_gray, inv_blur_img, scale=256.0)
    
    if mode == "binary":
        _, sketch = cv2.threshold(sketch, 128, 255, cv2.THRESH_BINARY)
    elif mode == "invert":
        sketch = cv2.bitwise_not(sketch)

    pil_sketch = Image.fromarray(sketch)
    return pil_sketch

def adv_cartoon_effect(image: np.ndarray, intensity=1) -> np.ndarray:
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray_blurred = cv2.medianBlur(gray, 7)
    
    # Detect edges in the image
    edges = cv2.adaptiveThreshold(gray_blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # Convert back to color
    color = cv2.bilateralFilter(image, 9, 300, 300)
    
    # Combine edges and color
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

def sketch_image_pil_embossed(pil_img, mode="pencil"):
    img = np.array(pil_img.convert('RGB'), dtype=np.uint8)  # Convert PIL Image to NumPy array
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection (Canny)
    edges = cv2.Canny(img_gray, threshold1=30, threshold2=100)

    # Invert the edges
    inv_edges = cv2.bitwise_not(edges)

    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(inv_edges, (13, 13), 0)

    # Create an embossing kernel
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])

    # Apply the embossing kernel
    embossed_img = cv2.filter2D(blur_img, -1, kernel) + 128

    if mode == "binary":
        _, embossed_img = cv2.threshold(embossed_img, 128, 255, cv2.THRESH_BINARY)
    elif mode == "invert":
        embossed_img = cv2.bitwise_not(embossed_img)

    pil_sketch = Image.fromarray(embossed_img)
    return pil_sketch


def get_image_download_link(img, filename="processed_image.png", text="Download Image"):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    b64 = base64.b64encode(buffered.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

def convert_to_black_and_white(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

def resize_with_aspect_ratio(image, target_height, target_width):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


def simple_lines_effect(img):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 75, 150)

    # Convert edges to color
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges_colored

def complex_lines_effect(img):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with finer details
    edges = cv2.Canny(blurred, 30, 70)

    # Dilating the edges to make them more pronounced
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Convert edges to color
    dilated_colored = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    return dilated_colored


def model_predict(img, model):
    # Ensure the image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Capture the original size of the input image
    original_size = img.size

    # Transform and predict using the model
    transform = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC), 
                                    transforms.ToTensor()])
    img = transform(img)
    img = torch.unsqueeze(img, 0)

    with torch.no_grad():
        output = model(img)
    
    output = output.squeeze(0).detach()
    output_img = transforms.ToPILImage()(output)

    # Resize the output image back to the original size
    output_img = output_img.resize(original_size, Image.BICUBIC)
    
    return output_img

# Streamlit UI with integrated Background Removal
st.title("Tosief's Artistic Image Toolkit 🎨🖼️")

st.markdown("""
# Explore the Magic of AI-Powered Image Styling ✨
From Cartoons to Sketches and Line Art, transform your images into unique artistic creations! 🌟🖌️
""")
# st.title("Generate Cartoonized, Sketch, and Artistic Filled Images")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
    st.session_state.last_uploaded_file = None

original_image = None

if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
    original_image = Image.open(uploaded_file)
    st.session_state.processed_images = [('Original', original_image)]
    st.session_state.last_uploaded_file = uploaded_file

for idx, (label, image) in enumerate(st.session_state.processed_images):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f'{label} Image', use_column_width=True)

    if idx == len(st.session_state.processed_images) - 1:
        processed_image = None
        sketch_style = "pencil"  # Default sketch style

        if len(st.session_state.processed_images) > 1:
            image_choice_key = f'image-choice-{idx}'
            image_choice = st.radio("Choose an image to process", ["Original", "Last Processed"], key=image_choice_key)
            chosen_image = st.session_state.processed_images[0][1] if image_choice == "Original" else image
        else:
            chosen_image = image

        option_key = f'option-{idx}'
        # option = st.selectbox('Choose the style', ['Select', 'Cartoon', 'Adv_Cartoon', 'Sketch', 'Sketch Embossed', 'Fill Sketch', 'Remove Background', 'Black and White'], key=option_key)
        option = st.selectbox('Choose the style', 
                              ['Select', 'Avatar', 'Sketch', 'Sketch Simple', 'Sketch Complex', 'Fill Sketch', 
                               'Remove Background', 'Black and White'], 
                              key=option_key)
        
        cartoon_intensity = None
        adv_cartoon_intensity = None
        if option == 'Avatar':
            cartoon_intensity = 1
            # st.slider("Select Avatar Intensity", min_value=1, max_value=10, value=1, key=f'cartoon-intensity-{idx}')
        elif option == 'Adv_Avatar':
            adv_cartoon_intensity = st.slider("Select Advanced Avatar Intensity", min_value=1, max_value=10, value=1, key=f'adv-cartoon-intensity-{idx}')
        elif option == 'Sketch':
            sketch_intensity = st.slider("Select Sketch Intensity", min_value=1, max_value=25, value=25, step=2, key=f'slider-{idx}')
            radio_key = f'radio-{idx}'
            # sketch_style = st.radio("Choose Sketch Style", ["Pencil", "Binary", "Invert"], key=radio_key)
            sketch_style = st.radio("Choose Sketch Style", ["Pencil", "Invert"], key=radio_key)

        if option != 'Select':
            button_key = f'create-{option}-{idx}'
            if st.button(f'Create {option}', key=button_key):
                if chosen_image is not None:
                    with st.spinner(f'Generating {option} Image...'):
                        if option == 'Avatar' and cartoon_intensity is not None:
                            processed_image = inference(chosen_image, cartoon_intensity=cartoon_intensity)
                        elif option == 'Adv_Avatar' and adv_cartoon_intensity is not None:
                            processed_image = inference(chosen_image, cartoon_intensity=adv_cartoon_intensity)
                            # Ensure processed_image is a PIL Image before the next inference call
                            if isinstance(processed_image, np.ndarray):
                                processed_image = Image.fromarray(processed_image.astype(np.uint8))

                            processed_image = adv_cartoon_effect(np.array(processed_image.convert('RGB')), intensity=adv_cartoon_intensity)
                            processed_image = Image.fromarray(processed_image)

                        elif option == 'Sketch' and sketch_intensity is not None:
                            processed_image = sketch_image_pil(chosen_image, mode=sketch_style.lower(), intensity=sketch_intensity)
                        elif option == 'Sketch Embossed':
                            processed_image = sketch_image_pil_embossed(chosen_image, mode=sketch_style.lower())
                        elif option == 'Fill Sketch':
                            processed_image = fill_sketch(chosen_image)
                            # processed_image = enhanced_fill_sketch(chosen_image)
                        elif option == 'Remove Background':
                            img_byte_arr = io.BytesIO()
                            chosen_image.save(img_byte_arr, format='PNG')
                            img_byte_arr = img_byte_arr.getvalue()
                            output = remove(img_byte_arr)
                            processed_image = Image.open(io.BytesIO(output))
                        elif option == 'Black and White':
                            np_image = np.array(chosen_image.convert('RGB'))
                            processed_image = convert_to_black_and_white(np_image)
                            processed_image = Image.fromarray(processed_image)
                        elif option == 'Simple Lines':
                            processed_image = simple_lines_effect(np.array(chosen_image.convert('RGB')))
                            processed_image = Image.fromarray(processed_image)
                        elif option == 'Complex Lines':
                            # complex_lines_intensity slider no longer needed since dilate_iter is not a parameter
                            processed_image = complex_lines_effect(np.array(chosen_image.convert('RGB')))
                            processed_image = Image.fromarray(processed_image)
                        elif option == 'Sketch Simple':
                            processed_image = model_predict(chosen_image, model1)
                        elif option == 'Sketch Complex':
                            processed_image = model_predict(chosen_image, model2)

                        if processed_image is not None:
                            st.session_state.processed_images.append((f'{option}', processed_image))
                            st.success('Process is completed successfully')
                else:
                    st.error("No image selected. Please select an image to process.")