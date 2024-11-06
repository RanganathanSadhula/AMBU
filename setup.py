import os
import sys
import yaml

def create_project_structure():
    """Create the necessary project structure and configuration files"""
    
    # Create directories
    directories = [
        'ambulance_dataset/images/train',
        'ambulance_dataset/images/val',
        'ambulance_dataset/labels/train',
        'ambulance_dataset/labels/val',
        'weights'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create YAML configuration
    yaml_config = {
        'path': './ambulance_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'ambulance'
        },
        'nc': 1
    }
    
    # Save YAML configuration
    with open('ambulance_data.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    print("Created YAML configuration file")

    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Place your training images in 'ambulance_dataset/images/train'")
    print("2. Place your validation images in 'ambulance_dataset/images/val'")
    print("3. Place corresponding label files in 'ambulance_dataset/labels/train' and 'labels/val'")
    print("4. Run train.py to start training")

if __name__ == "__main__":
    create_project_structure()