# Project Proposal: Cancer Detection Using Convolutional Neural Networks (CNNs)

## Description
This project aims to develop a deep learning-based system for detecting breast cancer and grading it based on histopathological images. By leveraging Convolutional Neural Networks (CNNs), we will focus on classifying images as either low-grade or high-grade cancer, with the goal of providing early detection and diagnostic support to medical professionals.

## Goals
1. **Accurate Classification**: Successfully classify histopathology images as low-grade, high-grade, or benign tissue.
2. **Generalization and Scalability**: Ensure the model performs well across images from different sources by training on one subset of images (e.g., low-grade cases) and testing on another (e.g., high-grade cases).

## Data Collection
We will use the [Breast Cancer Grade Dataset from Kaggle](https://www.kaggle.com/datasets/lesliedalton/breast-cancer-grade). This dataset includes histopathological images of breast cancer tissue, categorized into:
   - **High Grade**: High-risk cancer cells showing severe abnormalities.
   - **Low Grade**: Low-risk cancer cells with less aggressive features.
   - **Benign Tissue (Stroma)**: Non-cancerous tissue sections and empty regions representing normal, non-tumorous areas.

## Modeling Approach
1. **Preprocessing**: Preprocess images by resizing, normalizing, and augmenting to improve model performance and prevent overfitting. Techniques will include random rotation, horizontal/vertical flips, and color adjustments.
2. **Model Architecture**: 
   - Initially, use a pre-trained model (such as VGG16 or ResNet) and fine-tune it for our specific task. 
   - Later, experiment with custom CNN architectures tailored to the dataset characteristics for optimized performance.
3. **Cross-Domain Training**: Train on one subset (e.g., low grade) and validate on another (e.g., high grade) to ensure that the model can generalize across different image subsets, emulating real-world scenarios.

## Visualization
To understand and interpret the model’s decisions, we will use the following visualization techniques:
   - **Class Activation Maps (CAMs) or Heatmaps**: Highlight areas where the model focuses on detecting cancer cells.
   - **t-SNE Plots**: Represent high-dimensional features in a two-dimensional space to visualize clustering of features learned by the model.
   - **ROC Curves**: Evaluate classification performance, especially for differentiating between low and high-grade cancer tissue.

## Test Plan
1. **Data Split**: Divide the data into training and testing sets, with 80% for training and 20% for testing, maintaining a balanced class distribution.
2. **Cross-Validation**: Apply cross-validation to fine-tune hyperparameters, ensuring robustness and minimizing overfitting.
3. **Generalization Testing**: Train the model on one set (e.g., CHTN) and test on the other (e.g., TCGA), evaluating its performance in a cross-dataset setting. The primary evaluation metric will be the percentage of correctly classified images across all categories (high, low, and stroma).

## Hardware Capabilities
To handle the computational demands of training on high-resolution histopathological images, we will utilize either **Google Colab** or **Boston University’s Shared Computing Cluster**:
   - **Google Colab**: Allows us to leverage free GPU/TPU resources for quick prototyping and smaller-scale experiments.
   - **Boston University’s Shared Computing Cluster**: Provides access to a high-performance computing environment suitable for training on large datasets and conducting extended experiments, especially beneficial for projects that require more computational power.

These resources will enable us to scale our project effectively and manage complex computations, particularly when working with larger image sizes and experimenting with various CNN architectures.
