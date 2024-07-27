from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def auroc_confusion_matrix(args, labels, scores, result_file_path):

    roc_auc = roc_auc_score(labels, scores) * 100

    # confusion matrix
    normal_scores = [score for label, score in zip(labels, scores) if label == 0]
    normal_max_dist = max(normal_scores)
    abnormal_scores = [score for label, score in zip(labels, scores) if label == 1]
    abnormal_max_dist = min(abnormal_scores)
    
    threshold = sum(normal_scores) / len(normal_scores)
    predictions = [1 if score >= threshold else 0 for score in scores]
    cf_matrix = confusion_matrix(labels, predictions)
    
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    cf_labels = ["{0}\n{1}\n({2})".format(v1, v2, v3) for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    cf_labels = np.asarray(cf_labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=cf_labels, fmt='', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Abnormal'], yticklabels=['Actual Normal', 'Actual Abnormal'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Class {args.normal_class}')
    plt.savefig(f'result/{args.dataset}/Confusion_Matrix_{args.abnormal_class}.png')
    plt.close()

    normal_scores_np = np.array(normal_scores)
    abnormal_scores_np = np.array(abnormal_scores)

    with open(result_file_path, 'a') as file:
        file.write(f' {args.normal_class}    /    {args.abnormal_class} : {roc_auc:.2f}%\n')
        file.write(f'Label counts: {Counter(labels)}\n')
        file.write(f'Prediction counts: {Counter(predictions)}\n')
        file.write(f'Normal Scores - Min: {np.min(normal_scores_np):.2f}, Max: {np.max(normal_scores_np):.2f}, Mean: {np.mean(normal_scores_np):.2f}\n')
        file.write(f'Abnormal Scores - Min: {np.min(abnormal_scores_np):.2f}, Max: {np.max(abnormal_scores_np):.2f}, Mean: {np.mean(abnormal_scores_np):.2f}\n')


def top5_down5_visualization(args, indices, labels, scores, data):
    # 가장 이상적인 normal, abnormal class 5개 시각화 
    normal_indices_scores = [(idx, score) for idx, score, label in zip(indices, scores, labels) if label == 0]
    abnormal_indices_scores = [(idx, score) for idx, score, label in zip(indices, scores, labels) if label == 1]

    normal_indices_scores.sort(key=lambda x: x[1])  
    abnormal_indices_scores.sort(key=lambda x: x[1], reverse=True)  
    top_normal_images_scores = normal_indices_scores[:5]
    top_abnormal_images_scores = abnormal_indices_scores[:5]

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, (idx, score) in enumerate(top_normal_images_scores):
        img = data[1].dataset[idx][0]
        img = img.permute(1, 2, 0).cpu().numpy()  # Adjust for matplotlib
        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Normal\nScore: {score:.2f}')
        axs[0, i].axis('off')

    for i, (idx, score) in enumerate(top_abnormal_images_scores):
        img = data[1].dataset[idx][0]
        img = img.permute(1, 2, 0).cpu().numpy()  # Adjust for matplotlib
        axs[1, i].imshow(img)
        axs[1, i].set_title(f'Abnormal\nScore: {score:.2f}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f'result/{args.dataset}/Class_{args.abnormal_class}_visualization.png')
    plt.close(fig)

    print(f'Finished processing class {args.normal_class} / {args.abnormal_class}.')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def distribution_normal(normal_scores, result_dir):
    """
    Normal 점수 분포 시각화 및 저장

    Args:
        normal_scores (list or np.array): Normal 점수 목록
        result_dir (str): 결과를 저장할 디렉토리 경로
    """
    plt.figure(figsize=(12, 8))
    
    # KDE Plot
    sns.kdeplot(normal_scores, shade=True, color='blue', label='Normal')
    
    # 히스토그램
    plt.hist(normal_scores, bins=30, alpha=0.5, color='blue', edgecolor='black', density=True)
    
    plt.title('Distribution of Normal Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_dir}/normal_score_distribution.png')
    plt.close()

def distribution_abnormal(abnormal_scores, result_dir):
    """
    Abnormal 점수 분포 시각화 및 저장

    Args:
        abnormal_scores (list or np.array): Abnormal 점수 목록
        result_dir (str): 결과를 저장할 디렉토리 경로
    """
    plt.figure(figsize=(12, 8))
    
    # KDE Plot
    sns.kdeplot(abnormal_scores, shade=True, color='red', label='Abnormal')
    
    # 히스토그램
    plt.hist(abnormal_scores, bins=30, alpha=0.5, color='red', edgecolor='black', density=True)
    
    plt.title('Distribution of Abnormal Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_dir}/abnormal_score_distribution.png')
    plt.close()

def distribution_comparison(normal_scores, abnormal_scores, result_dir):
    """
    Normal 및 Abnormal 점수 분포 비교 시각화 및 저장

    Args:
        normal_scores (list or np.array): Normal 점수 목록
        abnormal_scores (list or np.array): Abnormal 점수 목록
        result_dir (str): 결과를 저장할 디렉토리 경로
    """
    plt.figure(figsize=(12, 8))
    
    # KDE Plot
    sns.kdeplot(normal_scores, shade=True, color='blue', label='Normal')
    sns.kdeplot(abnormal_scores, shade=True, color='red', label='Abnormal')
    
    plt.title('Comparison of Normal and Abnormal Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_dir}/score_distribution_comparison.png')
    plt.close()

