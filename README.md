# Neural Style Transfer

This project implements Neural Style Transfer using PyTorch. It allows you to apply the artistic style of one image to the content of another.

## Features

-   **Style Presets**: Use built-in styles like `anime`, `van_gogh`, and `cyberpunk`.
-   **Automatic Resizing**: Automatically handles images of different sizes and aspect ratios.
-   **MPS Support**: Optimized for Apple Silicon (M1/M2/M3) GPUs.
-   **CLI Interface**: Easy-to-use command line interface.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Place your content images in the `input/` folder.

### Basic Usage

```bash
python main.py --content my_photo.jpg --style anime
```

This will:
1.  Look for `my_photo.jpg` in `input/`.
2.  Apply the `anime` style.
3.  Save the result to `output/my_photo_anime.jpg`.

### Custom Style Image

You can also provide a path to a custom style image:

```bash
python main.py --content my_photo.jpg --style path/to/style.jpg
```

### Options

-   `--content`: Path to content image (filename in `input/` or full path).
-   `--style`: Style preset name (e.g., `anime`) or path to style image.
-   `--output`: (Optional) Path to save output. Defaults to `output/`.
-   `--steps`: Number of optimization steps (default: 300).
-   `--imsize`: Image size (default: 512 on GPU/MPS, 128 on CPU).

## Project Structure

-   `main.py`: Main entry point.
-   `style_transfer.py`: Core logic for style transfer.
-   `utils.py`: Image loading and processing utilities.
-   `styles/`: Directory containing style preset images.
-   `input/`: Default directory for content images.
-   `output/`: Default directory for output images.
-   `tests/`: Verification scripts.
