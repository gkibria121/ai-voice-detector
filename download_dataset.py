"""
Dataset download script for ASVspoof and other deepfake audio datasets.

Usage:
    python download_dataset.py --dataset 1  # Download Fake-or-Real dataset
"""

import argparse
import os
import kagglehub


def download_asvspoof2019():
    """Download ASVspoof 2019 dataset."""
    print("=" * 70)
    print("Downloading ASVspoof 2019 Dataset")
    print("=" * 70)
    
    awsaf49_asvpoof_2019_dataset_path = kagglehub.dataset_download('awsaf49/asvpoof-2019-dataset')
    print('✓ Data source download complete.')
    
    src = os.path.join(awsaf49_asvpoof_2019_dataset_path, "LA", "LA")  # real folder
    dst = "./LA"  # symlink to create in current directory
    
    # Remove old symlink or folder if exists
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            print(f"Warning: {dst} exists as a directory, not a symlink. Removing...")
            import shutil
            shutil.rmtree(dst)
    
    # Create symlink
    os.symlink(src, dst, target_is_directory=True)
    print(f"✓ Symlink created: {dst} → {src}")
    print("✓ ASVspoof 2019 dataset ready!")


def download_fake_or_real():
    """Download Fake-or-Real dataset."""
    print("=" * 70)
    print("Downloading Fake-or-Real Dataset")
    print("=" * 70)
    
    mohammedabdeldayem_the_fake_or_real_dataset_path = kagglehub.dataset_download(
        'mohammedabdeldayem/the-fake-or-real-dataset'
    )
    print('✓ Data source download complete.')
    
    src = mohammedabdeldayem_the_fake_or_real_dataset_path
    dst = "./fake_or_real"
    
    # Remove old symlink or folder if exists
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            print(f"Warning: {dst} exists as a directory, not a symlink. Removing...")
            import shutil
            shutil.rmtree(dst)
    
    # Create symlink
    os.symlink(src, dst, target_is_directory=True)
    print(f"✓ Symlink created: {dst} → {src}")
    print("✓ Fake-or-Real dataset ready!")


def download_scenefake():
    """Download SceneFake dataset."""
    print("=" * 70)
    print("Downloading SceneFake Dataset")
    print("=" * 70)
    
    mohammedabdeldayem_scenefake_path = kagglehub.dataset_download(
        'mohammedabdeldayem/scenefake'
    )
    print('✓ Data source download complete.')
    
    src = mohammedabdeldayem_scenefake_path
    dst = "./scenefake"
    
    # Remove old symlink or folder if exists
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            print(f"Warning: {dst} exists as a directory, not a symlink. Removing...")
            import shutil
            shutil.rmtree(dst)
    
    # Create symlink
    os.symlink(src, dst, target_is_directory=True)
    print(f"✓ Symlink created: {dst} → {src}")
    print("✓ SceneFake dataset ready!")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for audio deepfake detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Options:
    1 - Fake-or-Real (mohammedabdeldayem/the-fake-or-real-dataset)
            Binary classification dataset for fake vs real audio

Examples:
    python download_dataset.py --dataset 1    # Download Fake-or-Real
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=int,
        required=True,
        choices=[1],
        help='Dataset to download: 1=Fake-or-Real'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DATASET DOWNLOADER")
    print("=" * 70 + "\n")
    
    # In this trimmed repo Fake-or-Real is dataset id 1
    if args.dataset == 1:
        download_fake_or_real()
    
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()