import torch
from PIL import Image
import os
import subprocess
import sys

def create_dummy_image(filename, color):
    img = Image.new('RGB', (256, 256), color=color)
    img.save(filename)

def test_run():
    print("Creating dummy images...")
    create_dummy_image('test_content.jpg', 'blue')
    create_dummy_image('test_style.jpg', 'red')

    # Get the path to main.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    main_script = os.path.join(project_root, 'main.py')

    print("Running style transfer (short run)...")
    # Run for only 10 steps to verify it works without crashing
    cmd = [
        sys.executable, main_script,
        '--content', 'test_content.jpg',
        '--style', 'test_style.jpg',
        '--output', 'test_output.jpg',
        '--steps', '10',
        '--imsize', '128'
    ]
    
    try:
        subprocess.check_call(cmd)
        if os.path.exists('test_output.jpg'):
            print("SUCCESS: Output image generated.")
        else:
            print("FAILURE: Output image not found.")
    except subprocess.CalledProcessError as e:
        print(f"FAILURE: Process crashed with error {e}")
    finally:
        # Cleanup
        if os.path.exists('test_content.jpg'): os.remove('test_content.jpg')
        if os.path.exists('test_style.jpg'): os.remove('test_style.jpg')
        # Keep output for inspection if needed, or remove it. 
        # For this test, we might want to keep it or just check existence.
        if os.path.exists('test_output.jpg'): os.remove('test_output.jpg')

if __name__ == '__main__':
    test_run()
