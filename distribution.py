import os
import matplotlib.pyplot as plt

def plot_label_distribution(dataset_path='dataset/train'):
    """
    Counts the number of videos in each label directory and plots a bar chart.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Directory '{dataset_path}' not found.")
        return

    labels = []
    counts = []

    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        
        if os.path.isdir(label_path):
            video_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
            num_videos = len(video_files)
            
            labels.append(label)
            counts.append(num_videos)

    if not labels:
        print(f"No label directories found inside '{dataset_path}'.")
        return


    sorted_indices = sorted(range(len(labels)), key=lambda i: counts[i], reverse=True)
    labels = [labels[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    plt.figure(figsize=(15, 8))
    plt.bar(labels, counts, color='skyblue')
    
    plt.xlabel('Labels', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.title('Distribution of Videos per Label', fontsize=14)
    
    plt.xticks(rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    output_image = 'label_distribution.png'
    plt.savefig(output_image)
    print(f"Bar chart saved as '{output_image}'.")
    
    plt.show()

if __name__ == '__main__':
    plot_label_distribution('dataset/train')
