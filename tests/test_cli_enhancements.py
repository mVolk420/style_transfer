import os
import subprocess
import sys
from PIL import Image

def create_dummy_image(filename, color):
    img = Image.new('RGB', (256, 256), color=color)
    img.save(filename)

def test_cli_enhancements():
    # Setup directories
    if not os.path.exists('input'): os.makedirs('input')
    if not os.path.exists('output'): os.makedirs('output')

    print("Creating dummy content image in input/...")
    content_filename = 'test_auto_content.jpg'
    content_path = os.path.join('input', content_filename)
    create_dummy_image(content_path, 'yellow')

    # Get path to main.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    main_script = os.path.join(project_root, 'main.py')

    print("Testing auto-resolution of input and output...")
    # Command: python main.py --content test_auto_content.jpg --style anime --steps 10
    # Expected output: output/test_auto_content_anime.jpg
    cmd = [
        sys.executable, main_script,
        '--content', content_filename, # Note: just filename, not path
        '--style', 'anime',
        '--steps', '10',
        '--imsize', '128'
    ]
    
    expected_output = os.path.join('output', 'test_auto_content_anime.jpg')
    
    # Clean up previous run if exists
    if os.path.exists(expected_output):
        os.remove(expected_output)

    try:
        subprocess.check_call(cmd)
        if os.path.exists(expected_output):
            print(f"SUCCESS: Output generated at {expected_output}")
        else:
            print(f"FAILURE: Output not found at {expected_output}")
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: Process crashed with error {e}")
    finally:
        if os.path.exists(content_path): os.remove(content_path)
        if os.path.exists(expected_output): os.remove(expected_output)

if __name__ == '__main__':
    test_cli_enhancements()
