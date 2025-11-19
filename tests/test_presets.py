import os
import subprocess
import sys
from PIL import Image

def create_dummy_image(filename, color):
    img = Image.new('RGB', (256, 256), color=color)
    img.save(filename)

def test_presets():
    print("Creating dummy content image...")
    create_dummy_image('test_content_preset.jpg', 'green')

    # Get the path to main.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    main_script = os.path.join(project_root, 'main.py')

    print("Testing 'anime' preset...")
    cmd = [
        sys.executable, main_script,
        '--content', 'test_content_preset.jpg',
        '--style', 'anime',
        '--output', 'test_output_anime.jpg',
        '--steps', '10',
        '--imsize', '128'
    ]
    
    try:
        subprocess.check_call(cmd)
        if os.path.exists('test_output_anime.jpg'):
            print("SUCCESS: Anime preset worked.")
        else:
            print("FAILURE: Anime preset output not found.")
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: Process crashed with error {e}")
    finally:
        if os.path.exists('test_content_preset.jpg'): os.remove('test_content_preset.jpg')
        if os.path.exists('test_output_anime.jpg'): os.remove('test_output_anime.jpg')

if __name__ == '__main__':
    test_presets()
