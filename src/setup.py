"""
Environment Configuration Module
------------------------------
This module handles environment setup for both local and cloud-based development,
including configuration for Google Colab, Google Cloud Platform, and GPU acceleration.

The main components include:
- Environment configuration management via EnvironmentConfig dataclass
- Google Drive mounting and data directory setup
- Package dependency management
- GPU/device configuration
- Git and GCP credentials setup
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import subprocess
import logging
from pathlib import Path

import torch
from google.colab import userdata, auth, drive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Stores and validates all environment configuration settings.
    
    Attributes:
        data_dir (Path): Path to data directory, either local or on Google Drive
        device (torch.device): Computing device (CPU/GPU) for model operations
        gcp_bucket_name (Optional[str]): GCP storage bucket name if using cloud storage
        gcp_file_prefix (Optional[str]): File prefix for GCP storage organization
        project_id (Optional[str]): Google Cloud Project identifier
    """
    data_dir: Path
    device: torch.device
    gcp_bucket_name: Optional[str] = None
    gcp_file_prefix: Optional[str] = None
    project_id: Optional[str] = None

    def is_using_gcp(self) -> bool:
        """Check if GCP configuration is active."""
        return all([self.gcp_bucket_name, self.gcp_file_prefix, self.project_id])


def mount_google_drive(google_drive_dir: str) -> Path:
    """Mount Google Drive and set up data directory.
    
    Args:
        google_drive_dir: Directory name within Google Drive for data storage
        
    Returns:
        Path: Configured data directory path
        
    Raises:
        RuntimeError: If drive mounting fails
    """
    try:
        drive.mount('/content/drive')
        data_dir = Path('/content/drive/MyDrive') / google_drive_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory configured at: {data_dir}")
        return data_dir
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        raise RuntimeError("Google Drive mount failed") from e


def install_requirements() -> None:
    """Install project dependencies from requirements.txt.
    
    Raises:
        FileNotFoundError: If requirements.txt is not found
        subprocess.CalledProcessError: If pip installation fails
    """
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        logger.info("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        raise
    except FileNotFoundError:
        logger.error("requirements.txt not found")
        raise


def check_gpu() -> bool:
    """Verify GPU availability and configuration.
    
    Returns:
        bool: True if GPU is available and properly configured
    """
    try:
        subprocess.run(['nvidia-smi'], check=True, capture_output=True)
        logger.info("GPU configuration verified")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("GPU not available, falling back to CPU")
        return False


def get_gcp_config() -> Optional[Dict[str, str]]:
    """Retrieve and validate GCP configuration from environment.
    
    Returns:
        Optional[Dict[str, str]]: GCP configuration if available, None otherwise
    """
    try:
        config = {
            'bucket_name': userdata.get('GCP_BUCKET_NAME'),
            'file_prefix': userdata.get('GCP_FILEPATH'),
            'project_id': userdata.get('GCP_PROJECT_ID')
        }
        if all(config.values()):
            return config
    except Exception as e:
        logger.warning(f"GCP configuration not available: {e}")
    return None


def configure_environment(environment: str, google_drive_dir: str) -> EnvironmentConfig:
    """Configure the runtime environment with necessary settings.
    
    This function handles the complete environment setup, including:
    - Git credential configuration
    - Google Drive mounting
    - Package installation
    - Device (CPU/GPU) setup
    - GCP configuration (if available)
    
    Args:
        environment: Type of environment ('local' or 'colab')
        google_drive_dir: Directory in Google Drive for data storage
    
    Returns:
        EnvironmentConfig: Complete environment configuration
        
    Raises:
        ValueError: If environment type is not supported
    """
    if environment != 'colab':
        raise ValueError(f"Environment '{environment}' not supported")
        
    # Configure Git credentials
    github_email = userdata.get('GITHUB_EMAIL')
    github_username = userdata.get('GITHUB_USER_NAME')
    os.system(f'git config --global user.email "{github_email}"')
    os.system(f'git config --global user.name "{github_username}"')
    
    # Set up environment components
    auth.authenticate_user()
    data_dir = mount_google_drive(google_drive_dir)
    install_requirements()
    
    # Configure compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    check_gpu()
    
    # Set up GCP if available
    gcp_config = get_gcp_config()
    if gcp_config:
        os.system(f'gcloud config set project {gcp_config["project_id"]}')
        logger.info("GCP Project configured")
    
    # Create and return configuration
    return EnvironmentConfig(
        data_dir=data_dir,
        device=device,
        gcp_bucket_name=gcp_config['bucket_name'] if gcp_config else None,
        gcp_file_prefix=gcp_config['file_prefix'] if gcp_config else None,
        project_id=gcp_config['project_id'] if gcp_config else None
    )