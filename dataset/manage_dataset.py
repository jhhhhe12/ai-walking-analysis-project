#!/usr/bin/env python3
"""
AI Dataset Management Tool
Manages images and labels for YOLO training
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

class DatasetManager:
    def __init__(self, base_path='/opt/ai/dataset'):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / 'images'
        self.labels_path = self.base_path / 'labels'
        
        # Splits
        self.splits = ['train', 'val', 'test']
        
    def get_stats(self):
        """Get comprehensive dataset statistics"""
        stats = {
            'images': {},
            'labels': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for split in self.splits:
            # Count images
            img_path = self.images_path / split
            if img_path.exists():
                images = list(img_path.glob('*.[jJ][pP][gG]')) + \
                        list(img_path.glob('*.[pP][nN][gG]')) + \
                        list(img_path.glob('*.[jJ][pP][eE][gG]'))
                stats['images'][split] = len(images)
            else:
                stats['images'][split] = 0
            
            # Count labels
            lbl_path = self.labels_path / split
            if lbl_path.exists():
                labels = list(lbl_path.glob('*.txt'))
                stats['labels'][split] = len(labels)
            else:
                stats['labels'][split] = 0
        
        # Totals
        stats['images']['total'] = sum(stats['images'].values())
        stats['labels']['total'] = sum(stats['labels'].values())
        
        return stats
    
    def print_stats(self):
        """Print dataset statistics in a formatted way"""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("📊 AI Dataset Statistics")
        print("="*70)
        print(f"📁 Base Path: {self.base_path}")
        print(f"⏰ Timestamp: {stats['timestamp']}")
        print("-"*70)
        
        print(f"\n{'Split':<15} {'Images':<20} {'Labels':<20}")
        print("-"*70)
        for split in self.splits:
            img_count = stats['images'].get(split, 0)
            lbl_count = stats['labels'].get(split, 0)
            print(f"{split.upper():<15} {img_count:<20} {lbl_count:<20}")
        
        print("-"*70)
        print(f"{'TOTAL':<15} {stats['images']['total']:<20} {stats['labels']['total']:<20}")
        print("="*70 + "\n")
        
        return stats
    
    def validate_dataset(self):
        """Validate that images have corresponding labels"""
        issues = []
        
        for split in self.splits:
            img_path = self.images_path / split
            lbl_path = self.labels_path / split
            
            if not img_path.exists() or not lbl_path.exists():
                continue
            
            # Get image files
            images = {f.stem: f for f in img_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
            labels = {f.stem: f for f in lbl_path.glob('*.txt')}
            
            # Find images without labels
            for img_stem in images:
                if img_stem not in labels:
                    issues.append({
                        'type': 'missing_label',
                        'split': split,
                        'file': images[img_stem].name
                    })
            
            # Find labels without images
            for lbl_stem in labels:
                if lbl_stem not in images:
                    issues.append({
                        'type': 'orphan_label',
                        'split': split,
                        'file': labels[lbl_stem].name
                    })
        
        return issues
    
    def create_yaml(self, output_path='dataset.yaml'):
        """Create YOLO dataset YAML configuration"""
        yaml_content = f"""# AI Dataset Configuration
# Generated: {datetime.now().isoformat()}

path: {self.base_path}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
names:
  0: car
  1: person
  2: object

# Number of classes
nc: 3
"""
        
        output_file = self.base_path / output_path
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Created YAML config: {output_file}")
        return output_file

def main():
    """Main function"""
    print("\n🤖 AI Dataset Manager")
    print("="*70)
    
    manager = DatasetManager()
    
    # Print statistics
    stats = manager.print_stats()
    
    # Validate dataset
    print("🔍 Validating dataset...")
    issues = manager.validate_dataset()
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - [{issue['type']}] {issue['split']}/{issue['file']}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✅ Dataset validation passed!")
    
    # Create YAML config
    print("\n📝 Creating dataset configuration...")
    manager.create_yaml()
    
    print("\n✨ Done!\n")

if __name__ == '__main__':
    main()
