import os
import subprocess
from google.colab import drive
from google.colab import auth
from google.colab import userdata
import torch


def mount_google_drive(google_drive_dir):
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Set up path to your data directory in Drive
    DATA_DIR = f'/content/drive/MyDrive/google_drive_dir'
    
    # Create directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    return DATA_DIR


def check_gpu():
    try:
        subprocess.run(['nvidia-smi'], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    else:
        print("Running in local environment. Ensure .env is configured.")


def configure_environment(environment, google_drive_dir):
    if environment == 'colab':
        # Retrieve GitHub and GCP credentials
        token = userdata.get('GITHUB_PAT')
        github_email = userdata.get('GITHUB_EMAIL')
        github_username = userdata.get('GITHUB_USER_NAME')
        project_id = userdata.get('GCP_EEG_PROJECT_ID')
        gcp_bucket_name = userdata.get('GCP_EEG_BUCKET_NAME')
        gcp_file_prefix = userdata.get('EEG_GCP_FILEPATH')

        # Configure Git with credentials
        os.system(f'git config --global user.email "{github_email}"')
        os.system(f'git config --global user.name "{github_username}"')
        os.system(f'gcloud config set project {project_id}')

        auth.authenticate_user()
        
        print(f"GCP Project Set")
        print("Git configured with your user data.")

        DATA_DIR = mount_google_drive(google_drive_dir)

        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        check_gpu()

    return gcp_bucket_name, gcp_file_prefix, project_id, DATA_DIR, device
