from dataclasses import dataclass
from typing import Optional
import os
import subprocess
import torch
from google.colab import userdata, auth, drive
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Configuration class to store all environment settings."""
    data_dir: Path
    device: torch.device
    gcp_bucket_name: Optional[str] = None
    gcp_file_prefix: Optional[str] = None
    project_id: Optional[str] = None


def mount_google_drive(google_drive_dir):
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Set up path to your data directory in Drive
    DATA_DIR = f'/content/drive/MyDrive/{google_drive_dir}'
    
    # Create directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    return DATA_DIR


def install_requirements():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
    except FileNotFoundError:
        print("requirements.txt file not found")


def check_gpu():
    try:
        subprocess.run(['nvidia-smi'], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    else:
        print("Running in local environment. Ensure .env is configured.")


def configure_environment(environment: str, google_drive_dir: str) -> EnvironmentConfig:
    """Configure the runtime environment with necessary settings.
    
    Args:
        environment: Type of environment ('local' or 'colab')
        google_drive_dir: Directory in Google Drive for data storage
    
    Returns:
        EnvironmentConfig object containing all configuration settings
    """
    if environment == 'colab':
        # Configure Git credentials
        github_email = userdata.get('GITHUB_EMAIL')
        github_username = userdata.get('GITHUB_USER_NAME')
        os.system(f'git config --global user.email "{github_email}"')
        os.system(f'git config --global user.name "{github_username}"')
        
        # Authenticate user
        auth.authenticate_user()
        
        # Mount Drive and get data directory
        data_dir = mount_google_drive(google_drive_dir)
        
        # Install required packages
        install_requirements()
        
        # Set up compute device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        check_gpu()
        
        # Try to get GCP configuration if available
        try:
            gcp_config = {
                'bucket_name': userdata.get('GCP_BUCKET_NAME'),
                'file_prefix': userdata.get('GCP_FILEPATH'),
                'project_id': userdata.get('GCP_PROJECT_ID')
            }
            if all(gcp_config.values()):
                os.system(f'gcloud config set project {gcp_config["project_id"]}')
                print("GCP Project Set")
            else:
                gcp_config = None
        except:
            gcp_config = None
        
        # Create and return configuration
        config = EnvironmentConfig(
            data_dir=data_dir,
            device=device,
            gcp_bucket_name=gcp_config['bucket_name'] if gcp_config else None,
            gcp_file_prefix=gcp_config['file_prefix'] if gcp_config else None,
            project_id=gcp_config['project_id'] if gcp_config else None
        )
        
        return config
    else:
        raise ValueError(f"Environment '{environment}' not supported")


