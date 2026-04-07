import os
import sys
import subprocess

INPUT_FOLDER = "test_images"


def is_image(file_name):
    return file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder not found: {INPUT_FOLDER}")
        return

    files = os.listdir(INPUT_FOLDER)
    image_files = [f for f in files if is_image(f)]

    if len(image_files) == 0:
        print("No images found.")
        return

    print(f"Found {len(image_files)} images.\n")

    python_executable = sys.executable
    print(f"Using Python: {python_executable}\n")

    for idx, file_name in enumerate(image_files, start=1):
        full_path = os.path.join(INPUT_FOLDER, file_name)

        print("=" * 50)
        print(f"[{idx}/{len(image_files)}] Testing: {file_name}")

        result = subprocess.run(
            [python_executable, "test_image.py", full_path],
            cwd=os.getcwd()
        )

        if result.returncode != 0:
            print(f"Test failed for: {file_name} (code: {result.returncode})")


if __name__ == "__main__":
    main()