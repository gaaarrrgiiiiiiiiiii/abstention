import kagglehub

def download_ieee_dataset():
    print("Downloading IEEE-CIS Fraud Detection dataset using kagglehub...")
    path = kagglehub.competition_download('ieee-fraud-detection')
    print("Path to competition files:", path)

if __name__ == "__main__":
    download_ieee_dataset()
