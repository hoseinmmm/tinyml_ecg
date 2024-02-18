import os
import shutil
import argparse

def find_and_copy_files(input_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        #os.makedirs(output_dir)
        print("output dir is not found")
        return

    count = 0
    # Iterate over files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".hea"):
            file_path = os.path.join(input_dir, filename)
            
            # Check if the specified string is in the .hea file
            with open(file_path, 'r') as file:
                if "# Dx: 426783006" in file.read():
                    # Print log
                    print(f"String found in {filename}")
                    

                    
                    # Copy its .mat couple
                    mat_filename = filename.replace(".hea", ".mat")
                    mat_file_path = os.path.join(input_dir, mat_filename)
                    if os.path.exists(mat_file_path):
                        shutil.copy(mat_file_path, output_dir)
                        print(f"Copied: {mat_filename} to {output_dir}")
                        # Copy the .hea file
                        shutil.copy(file_path, output_dir)
                        print(f"Copied: {filename} to {output_dir}")
                        count += 1
                    else:
                        print(f"No .mat file found for {filename}")

    print(f"in total, {count} file couples are copied!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy .hea and .mat file pairs based on content search.")
    parser.add_argument("--input_dir", help="The directory to search for .hea and .mat files")
    parser.add_argument("--output_dir", help="The directory to copy matched files into")

    args = parser.parse_args()

    find_and_copy_files(args.input_dir, args.output_dir)
