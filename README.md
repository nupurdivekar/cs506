# Final Report

## Dataset: Breast Cancer Grade Dataset from Kaggle

### Data Exploration and Feature Understanding
Before modeling, we performed thorough data exploration:

#### Dataset Structure & Classes
The dataset includes three categories: High, Low, and Stroma (benign) tissue images. Understanding class definitions was crucial:
- **High-grade**: More aggressive tumor cells, distinct morphological patterns.
- **Low-grade**: Less aggressive cells, subtler features.
- **Stroma**: Non-tumorous tissue or empty regions.

#### Visual Inspection
To understand the dataset's structure and appearance, we generated sample image grids for each tissue grade category (High, Low, Stroma). These grids visually display a subset of images, providing a comparative overview of the visual characteristics of each class:
- **High-grade images** often had darker staining patterns and more irregular cell clusters.
- **Low-grade images** looked intermediate, sometimes visually similar to both high and stroma.
- **Stroma images** were lighter and more uniform, often featuring fewer distinct cellular structures.

#### Dataset Justification
We decided to use the CHTN folder within the overall dataset since it was a richer data set, with higher quality images and had a greater number of images as well which helped with the training of our model.

#### Color Intensity & Distributions
We created RGB intensity histograms for random images in each grade category, highlighting the pixel intensity distributions for each RGB channel (Red, Green, and Blue). This helps in analyzing color patterns and contrasts across the tissue images. The intensity histograms reveal noticeable variations in color intensity between high-grade, low-grade, and stroma tissue:
- **High-grade images** tend to show darker or more intense color patterns, which might be indicative of tissue characteristics.
- These color patterns are critical as they can help the CNN model learn color-based features, potentially improving classification accuracy.

#### Image Sizes & Preprocessing
Most images were sufficiently large and of consistent quality. We decided to resize all images to a standard dimension (224x224) to ensure model compatibility. While resizing can lose some detail, this trade-off simplifies training and reduces memory usage.

### Key Insight from Exploration
The classes have some distinguishing features (color intensity, structure) that a CNN can learn. However, Low-grade is visually closer to both High and Stroma, suggesting that capturing subtle textural and morphological nuances would be critical.

---

## Detailed Description of Data Processing Done So Far

### Data Cleaning and Filtering
We ensured that each directory contains only valid image files (e.g., `.jpg`, `.png`, `.tif`). Filtering out non-image files helps avoid runtime errors during training and ensures that only relevant data is processed.

### Data Augmentation and Normalization
Data augmentation techniques were applied to enhance model generalization and prevent overfitting by artificially increasing the dataset's size. Here’s a breakdown of the augmentation techniques used:
- **Rotation Range**: ±20 degrees, which helps capture variations in tissue orientation.
- **Width and Height Shift**: ±20%, allowing the model to learn from slight shifts in tissue placement.
- **Horizontal and Vertical Flips**: Simulates real-world variations in tissue images.
- **Standardization**: All images were resized to 224x224 images for consistency and compatibility with the CNN model.
- **Normalization**: Pixel values were rescaled to fall within the [0, 1] range, which aids in faster model convergence and consistency during training.

### Training and Validation Split
We set an 80/20 split for training and validation to balance model training and evaluation. This split ensures that the model generalizes well by testing it on an unseen subset.

---

## Convolutional Neural Network (CNN) Model Choice

### Initial Attempts: Custom CNN from Scratch
#### What We Did:
As outlined in the midterm report, we trained a custom CNN from scratch to classify images into three tissue grades: High, Low, and Stroma, using 30 epochs, a batch size of 32, and TensorFlow’s Functional API. The architecture included three convolutional layers with increasing filters (32, 64, 128), max-pooling layers, a dense layer with 128 neurons, and dropout for overfitting prevention. Early stopping was applied based on validation loss, and the trained model demonstrated effective feature extraction from the dense layer for further classification. Dimensionality reduction techniques (t-SNE, PCA) visualized clear clustering of tissue grades, confirming the model’s ability to capture relevant features. Using these features, a Random Forest Classifier achieved 69.71% accuracy, with high performance for Class 2 (High) but lower precision and recall for Class 1 (Low), indicating overlapping features and the need for improved feature engineering.

#### Rationale:
Starting from scratch provided full control over the architecture and allowed us to experiment with varying complexity. However, histopathological images are inherently complex, requiring the network to learn low-level (e.g., edges) to high-level (e.g., patterns) features from scratch without any prior knowledge, making this approach challenging.

#### Reasoning on Results:
The lack of pretrained features meant the network had to learn foundational visual concepts in addition to domain-specific patterns. With a relatively small dataset, this proved insufficient to achieve high accuracy. This approach also lacked the generalizable feature representations available in pretrained networks.

## Model Tuning

### Attempt 2: VGG16 + Random Forest

#### What We Did
We utilized the VGG16 architecture, pretrained on ImageNet, as a feature extractor. The top layers were removed, and features from intermediate convolutional layers were flattened and passed to a Random Forest classifier. This approach allowed us to leverage VGG16's proven ability to capture edges, textures, and shapes in a hierarchical manner.

#### Rationale
VGG16’s simplicity and effectiveness made it a strong candidate for feature extraction. It has been widely validated for image classification tasks and offered a baseline for understanding the potential of pretrained feature hierarchies on our dataset.

#### Result
The VGG16 + Random Forest model achieved approximately 69-70% accuracy on the validation set. While the extracted features were moderately effective in separating the classes, the Random Forest classifier’s inability to optimize end-to-end limited its potential.

#### Key Takeaway
This attempt confirmed the value of pretrained feature extraction. However, the model’s accuracy plateaued due to the lack of domain-specific fine-tuning and end-to-end optimization.

---

### Attempt 3: EfficientNetB3 + Random Forest

#### What We Did
We transitioned to EfficientNetB3, a more advanced and modern architecture. EfficientNetB3’s compound scaling (balancing depth, width, and resolution) provided improved feature extraction capabilities. Features extracted from its convolutional layers were used as input to a Random Forest classifier.

#### Rationale
EfficientNetB3’s architecture is known for its efficiency and effectiveness, making it better suited for capturing subtle histological patterns. It also addressed some of the limitations of VGG16 by offering richer and more relevant features.

#### Result
The EfficientNetB3 + Random Forest model outperformed the VGG16-based approach, achieving approximately 80% accuracy on the validation set. The increased input size (224x224) and modern architecture allowed for better feature extraction. However, like VGG16, the Random Forest classifier restricted the model’s ability to fully adapt to task-specific nuances.

#### Key Takeaway
This attempt demonstrated that EfficientNetB3’s advanced feature extraction capabilities offered significant improvements. However, end-to-end fine-tuning was necessary to further boost performance.

---

### Attempt 4: Updated CNN Model

#### What We Did
We built a deeper and more advanced CNN from scratch, incorporating techniques such as batch normalization, dropout, and global average pooling. The model consisted of four convolutional blocks, each with multiple Conv2D layers followed by batch normalization and MaxPooling. A GlobalAveragePooling2D layer replaced the traditional Flatten operation to reduce feature dimensions, and a dense head with dropout layers was added for classification.

#### Rationale
This model was designed to address the limitations of the simpler scratch-built CNN by integrating techniques that stabilize training and improve generalization. The addition of batch normalization helped mitigate internal covariate shifts, and dropout reduced the risk of overfitting.

#### Result
The current CNN model achieved a validation accuracy of ~82%, with balanced precision, recall, and F1-scores across classes. Regularization techniques such as batch normalization and dropout proved instrumental in improving the model’s robustness. Global average pooling effectively reduced overfitting by minimizing the parameter count in the final layers.

#### Key Takeaway
The current CNN model outperformed earlier scratch-built architectures and ensemble models. However, it remained slightly less effective than fine-tuned pretrained models like EfficientNetB3, demonstrating the importance of leveraging pretrained knowledge for complex datasets.

### Attempt 5: MobileNetV2 + Ensemble

What We Did: MobileNetV2 was employed as a feature extractor. Features were flattened and passed to ensemble classifiers, including XGBoost, LightGBM, and CatBoost. A soft voting mechanism averaged the predictions from these classifiers to generate final outputs.  

Rationale: The ensemble approach aimed to leverage the strengths of different classifiers, each learning diverse decision boundaries. MobileNetV2’s lightweight architecture provided efficient feature extraction.  

Result: The MobileNetV2 + Ensemble approach achieved ~81% validation accuracy. While it outperformed earlier Random Forest-based methods, it fell short of the fine-tuned EfficientNetB3. This highlighted the importance of end-to-end optimization in tasks requiring nuanced feature adaptation.  

Key Takeaway: Ensemble models with MobileNetV2 features offered robustness but lacked the adaptability of end-to-end fine-tuning. This experiment reinforced the need for domain-specific refinement in pretrained networks.  

### Summary of Model Tuning

Through these attempts, we observed that pretrained models like EfficientNetB3 outperformed both scratch-built CNNs and VGG16-based feature extraction. While ensembles provided modest improvements, end-to-end fine-tuning was crucial for achieving state-of-the-art results. Fine-tuning EfficientNetB3 yielded an accuracy of ~86% and balanced performance across all metrics, making it the optimal approach for this task.

### Final Model Description and Justification

Our final model employs EfficientNetB3, a state-of-the-art convolutional neural network architecture that balances computational efficiency and accuracy. This model was chosen for its ability to generalize well while maintaining computational efficiency, making it suitable for medical image classification tasks.  

The architecture leverages transfer learning with pre-trained ImageNet weights. The top fully connected layers were removed, and custom layers were added, including a Global Average Pooling (GAP) layer to reduce the feature map dimensions while retaining spatial information, a dense layer with 256 neurons and ReLU activation to learn complex patterns, and two dropout layers with a 50% dropout rate to prevent overfitting. The final output layer consists of three neurons with softmax activation, corresponding to the three classes: "High," "Low," and "Stroma."  

The training process was divided into two phases. Initially, all layers of the EfficientNetB3 backbone were frozen, and only the added layers were trained. This ensured the preservation of the pre-trained weights and allowed the model to adapt to the dataset. A relatively high learning rate of 0.001 was used, and aggressive data augmentation strategies, such as rotation, width/height shifts, zoom, and horizontal flips, were applied to improve generalization. Early stopping and learning rate reduction were employed to avoid overfitting.  

In the fine-tuning phase, the top 50 layers of the EfficientNetB3 backbone were unfrozen, enabling the model to refine features specific to the dataset. A lower learning rate of 0.0001 was used to ensure stable updates to the pre-trained weights. Callbacks, including ReduceLROnPlateau and EarlyStopping, were used to adjust the learning rate dynamically and halt training when performance plateaued.  

The final model achieved a validation accuracy of 87%, with strong performance across all evaluation metrics. Precision, recall, and F1-scores were high, reflecting a well-balanced model that minimizes both false positives and false negatives. The confusion matrix showed minimal misclassification, with most "High" and "Low" grade samples being correctly identified and only a small fraction misclassified as "Stroma." These metrics indicate that the model is reliable for medical diagnostics, where accurate classification is critical.  

EfficientNetB3 was selected as the final model due to its state-of-the-art architecture and ability to generalize effectively. Its compound scaling optimizes depth, width, and resolution, offering a balanced tradeoff between computational efficiency and accuracy. The use of pre-trained weights significantly reduced training time and overfitting risk, especially given the relatively small dataset. The combination of aggressive data augmentation, dropout layers, and selective fine-tuning allowed the model to adapt effectively while maintaining robust performance. This approach ensures the model is reliable and efficient, meeting the rigorous demands of medical image classification tasks.

## Comparing Results

The two graphs provide a comprehensive comparison of model performance across various architectures. The first graph highlights validation accuracy, showcasing EfficientNetB3 (Fine-Tuned) as the best-performing model with the highest accuracy, followed closely by EfficientNetB3 + Random Forest. Simpler models like VGG16 + Random Forest and Simple CNN fall behind, emphasizing the advantage of using advanced architectures and transfer learning. The second graph delves deeper, comparing precision, recall, and F1-scores across models. Again, EfficientNetB3 (Fine-Tuned) demonstrates a balanced and superior performance in all metrics, indicating its reliability for robust classification tasks. The comparison reinforces the value of state-of-the-art models like EfficientNet in achieving high accuracy and consistent performance across critical evaluation metrics.
![output](https://github.com/user-attachments/assets/901408c5-2850-469f-918f-f717192eb135)

![3d6bc6b9-e9cc-42fd-9f73-bf67c9b26576](https://github.com/user-attachments/assets/27b8c26c-2dbb-4cbc-a501-c8cb0aaadcb3)


## Visualization

The visualization of training and validation accuracy and loss reveals consistent improvement over epochs, with convergence indicating that the model generalizes well without overfitting. The gradual reduction in both training and validation loss highlights effective optimization, although occasional spikes in validation loss suggest sensitivity to specific validation samples. Overall, the plots demonstrate that the model’s learning process was stable and well-tuned.

The confusion matrix further underscores the model's strengths and challenges. High precision and recall were achieved for the "High" and "Stroma" classes, reflecting the model's ability to distinguish these patterns effectively. However, the "Low" class exhibited some misclassifications, particularly as "High," likely due to overlapping features in histopathological images. This points to the need for enhanced feature extraction or additional training data to improve differentiation.

ROC curves and AUC scores provide a comprehensive view of the model’s performance across all classes. With AUC scores of ~0.98 for "High" and "Stroma" and ~0.95 for "Low," the model demonstrates robust classification capabilities. However, the slightly lower AUC for the "Low" class suggests areas for refinement to better capture subtle variations.

Visualizing the model architecture highlights a well-balanced design with EfficientNetB3 as the backbone. Dropout layers and a dense head mitigate overfitting while maintaining computational efficiency. This architecture strikes a balance between complexity and generalizability, making it effective for this dataset.

The visualization of filters and feature maps provides insights into the model’s learning process. Filters in the first convolutional layer capture basic features like edges and textures, while deeper layers abstract these into more complex patterns. Feature maps from sample images demonstrate the model's ability to focus on relevant regions, confirming its effectiveness in extracting meaningful features from histopathological slides.

Overall, these visualizations and metrics validate the model's robustness and effectiveness in classifying histopathological images. The insights gained from these analyses also highlight areas for future improvement, particularly in enhancing the differentiation of closely related classes like "Low" and "High."

---

## Conclusions

The model tuning process highlighted the progression from basic feature extraction to advanced architectures. VGG16 and EfficientNetB3 with Random Forest achieved 69% and 80% accuracy, respectively, leveraging pretrained features but lacking end-to-end optimization. A scratch-built CNN reached 68% accuracy, limited by data size and pretrained knowledge. The MobileNetV2 ensemble improved to 81%, showcasing robustness but falling short in adaptability. The current CNN model achieved 82% accuracy with balanced metrics, benefiting from modern techniques like batch normalization and dropout. However, the fine-tuned EfficientNetB3 achieved the highest accuracy of 88%, demonstrating the importance of domain-specific fine-tuning for complex datasets like histopathological images.

---

## Displaying the Final Model

- **Interactive Web App**: To simplify the process we made an interacting web application using Vite as a framework to allow users to upload their own image and test it with the model.
- **Web-side Image Preprocessing**: Since the original images in the data set have a dimension of 1x1 we made a cropping interface that let users select the area of interest on their image and prompt them with a preview before running the image through the model.
- **Using the Model in the Web App**: To use the model we export it as a Keras file and create an API endpoint using FastAPI so that the user can obtain the model and run the app locally.
- **Displaying Results**: The end result displayed on the webpage contains the following:
  - The primary classification (High, Low, or Stroma).
  - A breakdown of probabilities for each category, visualized through a bar chart built with Recharts.

Here are a few sample images of what our front end looks like:

![Cancer Cell Detection Analysis](https://github.com/user-attachments/assets/4b2555a0-1079-413c-ba8b-dc4ea0f7b128)

![AD_4nXf_M33j3LY3ycbSlXwfN1i1XKopkqqGfsrIcXEKqjZmi25rytWFHeSpa5R2zQnSTluITGLR4NfG8200MGoABMRDBgMRSbEe83M1SvoCUfmNgZMuro7gBLEk](https://github.com/user-attachments/assets/1ba28e70-43d7-44d6-9bda-a99dc7ace8f4)


![AD_4nXd3SH__oDKokdo3Guz8Qc4oTh6J9wlery4Cr7rHFmDEVuHfJ-lBsBArudtGQcQBX4B31FwGPXq2o1rqbnvuqWMh44p5_o0qJkK5b_doRP18cW1kFXN5Wa9h](https://github.com/user-attachments/assets/5638b37e-0e91-4293-98cb-3ac169cc959c)


## Installation and running the web interface
This web application provides an interface for analyzing microscopy images to detect and classify cancer cells. The application uses a TensorFlow model to classify cells into three categories: High, Low, and Stroma. Users can upload and crop images, which are then analyzed by the model.

## Prerequisites
- Python 3.10 or higher
- Node.js 16 or higher
- npm (usually comes with Node.js)
- Git

## Installation

### Setting Up the Project Using Make (Recommended)
```bash
# Clone the repository
git clone https://github.com/Kermit3T/cs506-final-project-front
cd your-repo-name
```

# Install all dependencies and set up environment
```bash
make setup
```

# Download the model
```bash
make install-model
```

# Run both frontend and backend
```bash
make run
```
The frontend will run on http://localhost:5173
The backend will run on http://localhost:8000

Available make commands:
```bash
make setup - Set up frontend and backend dependencies
make run - Start both servers
make clean - Remove virtual environment and node_modules
make install-model - Download model file
make help - Show all available commands
```

### Setting Up the Project Manually

### 1. Clone the Repository
```bash
git clone https://github.com/nupurdivekar/cs506
cd your-repo-name
```

### 2. Set Up Python Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
pip install tensorflow fastapi uvicorn python-multipart pillow numpy
```

### 4. Install Node.js Dependencies
```bash
npm install
```

### 5. Model Setup
Download the model file from this link:
```bash
https://drive.google.com/file/d/1MCeY5EMv3xC6g4uvenP8JB1pcfAhvhhA/view
```

Alternatively, you can run:
```bash
make install-model
````
## Running the Application
```

### 1. Start the Backend Server
In one terminal window (with virtual environment activated):
```bash
# Windows
.\venv\Scripts\activate
uvicorn api.app:app --reload

# Mac/Linux
source venv/bin/activate
uvicorn api.app:app --reload
```
The backend will run on http://localhost:8000

### 2. Start the Frontend Server
In a new terminal window:
```bash
npm run dev
```
The frontend will run on http://localhost:5173

## Usage
1. Open http://localhost:5173 in your web browser
2. Click the upload box to select an image (minimum size 244x244 pixels)
3. Use the cropping tool to select the area for analysis
4. Click "Crop Image" and then "Test Image"
5. View the analysis results, including:
   - Classification
   - Confidence score
   - Detailed probability breakdown for each class

## Troubleshooting

### Common Issues and Solutions

1. **Backend Won't Start**
   - Ensure Python virtual environment is activated
   - Check if port 8000 is available
   - Verify model file exists in correct location

2. **Frontend Won't Start**
   - Run `npm install` to ensure all dependencies are installed
   - Check if port 5173 is available
   - Clear npm cache with `npm cache clean --force`

3. **Image Upload Issues**
   - Ensure image meets minimum size requirements (244x244)
   - Check browser console for specific error messages
   - Verify file format is supported (JPG, PNG)

4. **Analysis Fails**
   - Check backend console for error messages
   - Verify backend server is running
   - Ensure image is properly cropped before analysis

### Still Having Issues?
Create an issue on the GitHub repository with:
- Steps to reproduce the problem
- Error messages
- Screenshots if applicable
- Your system information (OS, Python version, Node.js version)

## Important Notes
- This tool is for research and preliminary screening purposes only
- Not intended for medical diagnosis
- Always consult healthcare professionals for medical advice

