import shutil
import os

def move_pdf(source_dir, destination_dir, filename):
    # Check if the source file exists
    source_file = os.path.join(source_dir, filename)
    if not os.path.exists(source_file):
        print(f"The file '{filename}' does not exist in the source directory.")
        return

    # Check if the source file is a PDF
    if not filename.lower().endswith('.pdf'):
        print(f"The file '{filename}' is not a PDF file.")
        return

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move the file to the destination directory
    destination_file = os.path.join(destination_dir, filename)
    shutil.move(source_file, destination_file)
    print(f"The file '{filename}' has been successfully moved to '{destination_dir}'.")