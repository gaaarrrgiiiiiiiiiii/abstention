import kagglehub
import shutil
import os

print("Downloading IEEE-CIS Fraud Detection dataset using kagglehub...")
path = kagglehub.competition_download('ieee-fraud-detection')
print("Path to competition files:", path)

# Copy the CSV files to our data directory
data_dir = "c:/Users/HP/abstention/data/"
os.makedirs(data_dir, exist_ok=True)

files_to_copy = ['train_transaction.csv', 'train_identity.csv']

for f in files_to_copy:
    src = os.path.join(path, f)
    dst = os.path.join(data_dir, f)
    if os.path.exists(src):
        print(f"Copying {f} to {dst}")
        shutil.copy(src, dst)
    else:
        # Check if they are zipped
        src_zip = os.path.join(path, f + '.zip')
        if os.path.exists(src_zip):
            print(f"Extracting {f}.zip to {data_dir}")
            import zipfile
            with zipfile.ZipFile(src_zip, 'r') as zip_ref:
                zip_ref.extract(f, data_dir)
        else:
            print(f"Could not find {f} in {path}")

print("Dataset ready in data directory.")
