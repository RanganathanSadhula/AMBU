import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(data_dir, output_dir, test_size=0.2):
    """
    Prepare the dataset for YOLO training
    """
    # Create directory structure
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)
    
    # Get all image and label files
    image_files = [f for f in os.listdir(os.path.join(data_dir, 'images')) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Split dataset
    train_images, val_images = train_test_split(image_files, 
                                              test_size=test_size, 
                                              random_state=42)
    
    # Move files to respective directories
    for img_file in train_images:
        # Move image
        shutil.copy2(
            os.path.join(data_dir, 'images', img_file),
            os.path.join(output_dir, 'images/train', img_file)
        )
        # Move corresponding label
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(data_dir, 'labels', label_file)):
            shutil.copy2(
                os.path.join(data_dir, 'labels', label_file),
                os.path.join(output_dir, 'labels/train', label_file)
            )
    
    for img_file in val_images:
        # Move image
        shutil.copy2(
            os.path.join(data_dir, 'images', img_file),
            os.path.join(output_dir, 'images/val', img_file)
        )
        # Move corresponding label
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(data_dir, 'labels', label_file)):
            shutil.copy2(
                os.path.join(data_dir, 'labels', label_file),
                os.path.join(output_dir, 'labels/val', label_file)
            )

if __name__ == "__main__":
    prepare_dataset('raw_dataset', 'ambulance_dataset')