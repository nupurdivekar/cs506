# Midterm Report

**Preliminary Visualizations of Data**

**Sample Image Grids**

To understand the dataset's structure and appearance, we generated sample image grids for each tissue grade category (High, Low, Stroma). These grids visually display a subset of images, providing a comparative overview of the visual characteristics of each class.

display_image_grid(TCGA_PATH, categories=\['High', 'Low', 'Stroma'\], n_samples=5)

**Key Observations**:

- Images across different grades show notable differences in texture, color, and staining, which could help our model distinguish between categories.
- Visual inspection confirms that images are grouped correctly by grade, and we observed a reasonable diversity within each grade, beneficial for model training.

***RGB Intensity Histograms***

We created RGB intensity histograms for random images in each grade category, highlighting the pixel intensity distributions for each RGB channel (Red, Green, and Blue). This helps in analyzing color patterns and contrasts across the tissue images.

plot_intensity_histograms(TCGA_PATH, categories=\['High', 'Low', 'Stroma'\], n_samples=3)

**Key Observations**:

- The intensity histograms reveal noticeable variations in color intensity between high-grade, low-grade, and stroma tissue. High-grade images tend to show darker or more intense color patterns, which might be indicative of tissue characteristics.
- These color patterns are critical as they can help the CNN model learn color-based features, potentially improving classification accuracy.

Image Size Distributions

We examined the distribution of image dimensions (width and height) across categories by plotting histograms for each dimension. This ensures consistency in image size, which is essential for model training.

plot_image_size_distribution(TCGA_PATH, categories=\['High', 'Low', 'Stroma'\])

**Key Observations**:

- While images have some variation in size, they mostly fall within a manageable range. This makes it feasible to resize all images uniformly for input into a CNN without significant loss of detail.

These visualizations collectively provide a strong foundation, confirming that the dataset is well-organized and ready for further preprocessing and modeling.

**Detailed Description of Data Processing Done So Far**

To prepare the dataset for effective training, we implemented several data processing steps:

**Data Cleaning and Filtering**

We ensured that each directory contains only valid image files (e.g., .jpg, .png, .tif). Filtering out non-image files helps avoid runtime errors during training and ensures that only relevant data is processed.

**Data Augmentation and Normalization**

Data augmentation techniques were applied to enhance model generalization and prevent overfitting by artificially increasing the dataset's size. Here’s a breakdown of the augmentation techniques used:

- **Rotation Range**: ±20 degrees, which helps capture variations in tissue orientation.
- **Width and Height Shift**: ±20%, allowing the model to learn from slight shifts in tissue placement.
- **Horizontal and Vertical Flips**: Simulates real-world variations in tissue images.
- **Standardization**: All images were resized to 224x224 images for consistency and compatibility with the CNN model
- **Normalization**: Pixel values were rescaled to fall within the \[0, 1\] range, which aids in faster model convergence and consistency during training.

**Training and Validation Split**

We set a 70/30 split for training and validation to balance model training and evaluation. This split ensures that the model generalizes well by testing it on an unseen subset.

**Detailed Description of Data Modeling Methods Used So Far**

**Convolutional Neural Network (CNN) Model Choice**

We selected a CNN architecture as our primary model due to its strengths in image recognition tasks, such as capturing spatial hierarchies and texture patterns. We are considering either a custom CNN or using transfer learning with a pre-trained network (e.g., VGG, ResNet) if data complexity warrants it.

For our classification task, categorical class labels (High, Low, Stroma) were converted into numerical values using LabelEncoder. This encoding step prepares the labels for input into the model.

The ImageDataGenerator class from TensorFlow’s Keras library is configured to create batches of augmented images for training. This data generator will feed the CNN model during training, helping it learn to identify each category’s unique characteristics under varied conditions.

**Preliminary Data Exploration Results**

Our data exploration has revealed several promising observations:

1. **Visual Patterns**: Based on the RGB intensity histograms and sample images, there are clear visual differences between categories. This suggests that our CNN model may effectively learn distinguishing features based on these color and texture variations.
2. **Data Augmentation**: Preliminary data augmentation tests show that the transformations are visually consistent with real-world variations in histopathological images. This step is expected to boost the model's robustness and ability to generalize.
3. **Dataset Readiness**: The data preprocessing and organization steps have set up a well-structured dataset, ready for model training with reduced risk of issues like mislabeling or overfitting.

In summary, the data preparation and visualization stages have provided valuable insights and a robust starting point for CNN model training. Our next steps will focus on implementing the CNN architecture, training it on the augmented dataset, and evaluating its performance based on initial accuracy and classification metrics.

**Modeling and Results**

**Training Custom CNN from Scratch**

We trained a custom Convolutional Neural Network (CNN) model from scratch using a dataset downloaded from Kaggle, which contains images categorized into three tissue grades: High, Low, and Stroma. The training process involved:

\- We trained the model for 30 epochs, using a batch size of 32.

\- The model architecture was built using TensorFlow's **Functional API**, consisting of three convolutional layers followed by max-pooling layers. Specifically, the network includes layers with 32, 64, and 128 filters, followed by a **fully connected dense layer** with 128 neurons, and a **dropout layer** with a dropout rate of 0.5 to prevent overfitting.

\- **Early stopping** was used to prevent overfitting by monitoring the validation loss during training and halting training when performance on the validation set started to decline, thereby ensuring that the model did not over-optimize for the training data.

The trained CNN model was evaluated on the validation set, achieving promising results that indicate its ability to capture important spatial features distinguishing the different tissue types. The model was further used as a feature extractor, where features from the **'dense_1'** layer were extracted for further analysis. These intermediate features help provide a representation of the data that can be further leveraged for classification.

**Feature Extraction with Custom CNN**

To better understand the dataset and enhance the performance of the model, we used feature extraction with our trained CNN. We modified the output of our CNN model to capture features from an intermediate layer, which helps in analyzing the quality of the learned features and gaining insights into how well the model is distinguishing between classes. We then used these extracted features for further classification with a Random Forest classifier, which helps evaluate how effective the learned features are for classification tasks beyond the original CNN.

The extracted features were visualized using dimensionality reduction techniques such as t-SNE and **PCA**, providing insights into how the features from different tissue grades cluster in a lower-dimensional space. Both t-SNE and PCA showed notable separation between tissue grades, indicating that the extracted features have captured relevant information that could help the classifier distinguish between different tissue types effectively.

**t-SNE and PCA Visualization**

**\- t-SNE** was applied to visualize how well the extracted features cluster based on their respective tissue grades (High, Low, Stroma). The resulting t-SNE plot highlighted clear groupings of the tissue grades, with significant visual separation, particularly between the stroma and other grades. This indicates that our CNN model is capable of extracting meaningful features that can effectively differentiate between the classes, which is an important step in ensuring that subsequent classifiers can achieve high accuracy.

**\- PCA** was used to provide another view of the feature space. The PCA plot similarly displayed a distinct separation between high-grade, low-grade, and stroma tissues, supporting the insights from the t-SNE plot. Both methods reinforce that the model can learn representations that distinguish between categories based on key features. The ability to visualize the data in lower dimensions helps confirm that the features learned by the model are indeed informative and suitable for downstream tasks.

**Random Forest Classification on Extracted Features**

Following feature extraction, a Random Forest Classifier was trained to evaluate the effectiveness of these extracted features. We split the dataset into training and validation sets (80/20 split) and trained the classifier on the extracted features. The classifier achieved an accuracy of 69.71% on the validation set, with the following key metrics:

\- **Class 0 (Stroma)**: Precision: **0.74**, Recall: **0.71**, F1-score: **0.72**

\- **Class 1 (Low)**: Precision: **0.54**, Recall: **0.55**, F1-score: **0.54**

\- **Class 2 (High)**: Precision: **0.82**, Recall: **0.88**, F1-score: **0.85**

The overall classification report suggests that while **Class 2** is well-separated (high recall and precision), **Class 1** has the most overlap with the other classes, resulting in lower metrics for that class. This indicates that more feature engineering may be needed to fully capture the distinct characteristics of Class 1, as it seems to share features with the other tissue types.

**Confusion Matrix Analysis**

To further understand the model's performance, we plotted a confusion matrix for the classifier on the validation set. The matrix revealed:

\- **Class 0 (Stroma)** had relatively fewer misclassifications, mostly confused with Class 1, suggesting that the features for Class 0 are fairly distinct, but there are still some shared characteristics with Class 1.

\- **Class 1 (Low)** showed significant misclassification, often mistaken for Class 0 and Class 2. This suggests that the features extracted may not be fully differentiating Class 1 from the others, likely due to similarities in visual texture and patterns that overlap with those of other classes. Improving the model’s ability to distinguish these subtle differences may require further analysis and feature enhancement.

\- **Class 2 (High)** exhibited high recall with only a few instances being misclassified as Class 1, indicating that the model is able to effectively learn and differentiate the features of high-grade tissues. This strong performance for Class 2 could be attributed to more distinct visual patterns, making it easier for the model to identify.

**Preliminary Conclusions**

\- The feature extraction approach using our trained CNN has successfully captured discriminative features that enable separation between tissue grades. However, the Random Forest classifier results indicate that additional feature engineering or more sophisticated classification models may be necessary to fully capture the distinctions between all classes, particularly for Class 1, which shows considerable overlap with the other grades.

\- The confusion matrix and classification metrics highlight that while **Classes 0 and 2** perform reasonably well, further improvements are needed for **Class 1**, perhaps through additional feature extraction or specialized augmentation. Addressing the limitations for Class 1 will be crucial in improving the overall model performance and ensuring that all classes are equally well-represented.

**Next Steps**

\- **Model Refinement**: Train a deeper CNN model, either custom or through transfer learning with a model like **ResNet** or **VGG**, to potentially improve the accuracy for **Class 1**. Using a pre-trained model may allow us to leverage learned features from large datasets, which could be particularly beneficial for capturing the subtle differences between classes.

\- **Hyperparameter Tuning**: Conduct hyperparameter optimization for the Random Forest model or explore using more sophisticated classifiers, such as **Support Vector Machines (SVM** to improve classification performance. Optimizing the parameters could lead to better separation and classification accuracy, especially for the underperforming class.

\- **Additional Augmentation**: Implement additional augmentation strategies focused on **Class 1** to create more diverse and representative samples, reducing overlap and improving model differentiation. This could include transformations that specifically enhance features unique to Class 1, making it easier for the model to distinguish this category from others.

\- **Feature Engineering**: Experiment with different feature extraction methods or apply **Principal Component Analysis (PCA)** to select the most informative features, potentially reducing the overlap between classes. Additionally, applying **wavelet transformations** or **texture analysis** could provide new insights and lead to better classification results.

\- **Ensemble Methods**: Consider using ensemble methods, combining multiple models to leverage their individual strengths and potentially achieve better overall classification performance. An ensemble approach could help mitigate weaknesses observed in single classifiers and improve robustness, especially for difficult-to-classify categories like Class 1.

\- **Deeper Analysis of Misclassifications**: Conduct a more detailed analysis of the misclassified instances, particularly focusing on **Class 1**. By examining these instances visually, we may identify common patterns or characteristics that are confusing the model, which could provide insights into improving feature extraction or data augmentation techniques.

\- **Incorporate Additional Data Sources**: To further enhance the model's ability to distinguish between tissue grades, consider incorporating additional datasets. Using data from multiple sources may increase variability and provide the model with a broader understanding of the tissue types, leading to better generalization.

\- **Fine-Tuning Training Strategies**: Implement more advanced training strategies, such as **learning rate schedules** or **adaptive optimizers** like **AdamW**, to improve model convergence and performance. Adjusting the learning rate dynamically could help the model learn more effectively, particularly for harder-to-classify categories.

\- **Layer-Wise Analysis**: Perform an analysis of different layers of the CNN to understand which features are being learned at various stages of the network. This analysis can help identify if certain features are being underrepresented and guide modifications to the model architecture to enhance learning for those features.

\- **Data Augmentation Visualization**: Extend the augmentation visualization process to include more examples from each class. Visualizing the augmented images for **Class 1** can provide further insight into why it might be challenging to differentiate, and whether the augmentation is effectively creating diverse representations for this class.

\- **Transfer Learning Evaluation**: Experiment with initializing the CNN model using weights from networks pre-trained on medical image datasets, such as **ImageNet** or specific histopathological image collections. This could provide a stronger baseline and enhance the ability of the model to identify subtle features early in the training process.


___________________________________________________________________________________________________________________________________________________________________________________
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
