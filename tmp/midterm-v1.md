Here are the detailed midterm assignment problems for Topics 2 through 10, designed in my capacity as a Harvard professor specializing in Deep Learning. Each mini-project is structured to build upon foundational concepts, consistent with the academic rigor of Mini-Project 1.

---

### **Mini-Project 2 - Structured Data Multi-class Classification**

You will work with the Palmer Penguins dataset, a modern alternative to the classic Iris dataset. This dataset contains measurements for three penguin species observed in the Palmer Archipelago, Antarctica.

Your task is to design and train a deep learning model to classify a penguin into one of the three species (_Adelie, Chinstrap, or Gentoo_) based on its physical measurements.

- **Dataset:** [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) (Available via the `palmerpenguins` Python library or directly as a CSV). The dataset contains 344 rows with features like bill length, bill depth, flipper length, body mass, and sex.
- **Task Definition:** Multi-class classification.
- **Learning Objectives:**
  - Demonstrate data preprocessing for a real-world dataset, including handling missing values and encoding categorical features.
  - Implement a neural network for multi-class classification.
  - Utilize a `softmax` activation function in the output layer and `categorical_crossentropy` as the loss function.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script with your full implementation.
  - Presentation slides summarizing your methodology, model architecture, and results.
- **Evaluation Criteria:**
  - **Accuracy:** The primary metric will be classification accuracy on the test set. A successful model should achieve an accuracy of over 95%.
  - **Confusion Matrix:** Provide a confusion matrix to analyze the model's performance for each species.

---

### **Mini-Project 3 - Structured Data Regression**

For this project, you will use the California Housing dataset, which contains data from the 1990 California census.

Your goal is to build a deep learning regression model to predict the median house value for California districts, based on various features.

- **Dataset:** [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (Available in Scikit-learn via `sklearn.datasets.fetch_california_housing`). The dataset includes features like median income, house age, average number of rooms, and location.
- **Task Definition:** Regression.
- **Learning Objectives:**
  - Understand and implement a regression model using a neural network.
  - Apply feature scaling, a critical step for regression tasks.
  - Select an appropriate activation function for the output layer (e.g., `linear`) and a suitable loss function (e.g., Mean Squared Error).
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script showing data preparation, model training, and evaluation.
  - Presentation slides detailing your approach, results, and a discussion on predicted vs. actual values.
- **Evaluation Criteria:**
  - **Mean Absolute Error (MAE):** Your primary goal is to minimize the MAE.
  - **R-squared (R²):** Report the R² value to measure how well your model explains the variance in the data.
  - **Visualization:** Plot predicted vs. actual house values to visually assess model performance.

---

### **Mini-Project 4 - Collaborative Filtering Recommendation System**

You will create a recommendation system using the MovieLens 100k dataset. This dataset contains 100,000 ratings from 943 users on 1,682 movies.

The task is to build a collaborative filtering model that predicts how a user would rate a movie they have not yet seen.

- **Dataset:** [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/). It contains user IDs, movie IDs, and ratings (1-5).[1]
- **Task Definition:** Rating prediction (Regression/Recommendation).
- **Learning Objectives:**
  - Implement a classic matrix factorization model using neural network embeddings.
  - Create separate embedding layers for users and movies.
  - Combine the embeddings using a dot product or concatenation to predict a rating.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script with your implementation.
  - Presentation slides explaining the concept of collaborative filtering, your model architecture, and its performance.
- **Evaluation Criteria:**
  - **Root Mean Squared Error (RMSE):** The primary metric will be RMSE on the test set ratings. A good model should achieve an RMSE below 1.0.
  - **Inference:** Demonstrate how your model can be used to predict a rating for a specific user-movie pair.

---

### **Mini-Project 5 - Content-Based Filtering Recommendation System**

Using a richer movie dataset, your task is to build a content-based recommendation system. This approach recommends items based on their properties rather than user interactions.

You will build a model that recommends movies to a user based on the textual content and genres of movies they have previously rated highly.

- **Dataset:** [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle. You should use `movies_metadata.csv`. This file includes movie genres, overviews (plot summaries), and other metadata.
- **Task Definition:** Recommendation/Ranking.
- **Learning Objectives:**
  - Process and vectorize textual data (movie overviews) using techniques like TF-IDF.
  - Create item profiles based on their content (genres, keywords).
  - Build a model that takes a user's liked movies and finds other movies with similar content.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script that implements the content-based logic.
  - Presentation slides explaining your feature engineering process and demonstrating recommendations for a sample user.
- **Evaluation Criteria:**
  - **Qualitative Assessment:** The evaluation will be based on the quality and relevance of the recommendations generated for a few sample users.
  - **Methodology:** Clarity and correctness of the feature extraction and similarity computation process.

---

### **Mini-Project 6 - Neural Network Embedding Recommendation System (Hybrid)**

This project combines collaborative and content-based filtering into a single, powerful hybrid model. You will use the MovieLens 1M dataset, which includes user ratings, movie metadata (genres), and user information.

Your task is to build a "two-tower" neural network that learns from both user-item interactions and their features.

- **Dataset:** [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/). This dataset contains 1 million ratings from 6,000 users on 4,000 movies.
- **Task Definition:** Hybrid Recommendation.
- **Learning Objectives:**
  - Implement a two-tower architecture: one tower for processing user features (user ID, etc.) and another for item features (movie ID, genres).
  - Learn to combine multiple types of features into a single recommendation model.
  - Understand the power of deep learning for learning complex user preferences and item characteristics.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script detailing the two-tower model architecture, training, and evaluation.
  - Presentation slides explaining the hybrid approach and comparing its performance to simpler models.
- **Evaluation Criteria:**
  - **RMSE/MAE:** Evaluate the model's rating prediction accuracy.
  - **Precision/Recall@K:** Evaluate the model's ability to rank relevant items highly for a user.

---

### **Mini-Project 7 - Convolutional Neural Network (CNN) for Image Classification**

This is your introduction to Convolutional Neural Networks. You will work with the CIFAR-10 dataset, a benchmark for image classification.

Your task is to build and train a CNN to classify 32x32 color images into 10 distinct classes (e.g., airplane, automobile, bird, cat).

- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Available in Keras via `keras.datasets.cifar10`). The dataset has 60,000 images split into 50,000 for training and 10,000 for testing.
- **Task Definition:** Multi-class image classification.
- **Learning Objectives:**
  - Construct a basic CNN from scratch using `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
  - Understand the role of convolutional filters in feature extraction.
    another for item features (movie ID, genres).
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script with your CNN implementation.
  - Presentation slides showing your model architecture, training curves (accuracy/loss), and results.
- **Evaluation Criteria:**
  - **Test Accuracy:** Your model should achieve a minimum of 70% accuracy on the test set.
  - **Confusion Matrix:** Analyze which classes your model confuses the most.

---

### **Mini-Project 8 - CNN with Transfer Learning**

Instead of training a CNN from scratch, you will leverage a pre-trained model to solve a new image classification problem. This technique, known as transfer learning, is highly effective and widely used.

Your task is to classify images as either "cat" or "dog" using a pre-trained model like VGG16 or MobileNetV2.

- **Dataset:** [Cats vs. Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) (Available via `tensorflow_datasets`). The dataset contains over 23,000 high-resolution images.
- **Task Definition:** Binary image classification.
- **Learning Objectives:**
  - Understand the concept of transfer learning (feature extraction and fine-tuning).
  - Load a pre-trained CNN and freeze its base layers.
  - Add a new classification head on top of the pre-trained base and train it on the new dataset.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script implementing the transfer learning workflow.
  - Presentation slides explaining your choice of pre-trained model and comparing the results of feature extraction vs. fine-tuning.
- **Evaluation Criteria:**
  - **Binary Accuracy:** Achieve a test accuracy of over 90%.
  - **Efficiency:** Discuss the benefits of transfer learning in terms of training time and data requirements compared to training from scratch.

---

### **Mini-Project 9 - CNN with Data Augmentation to Combat Overfitting**

Real-world datasets are often small, leading to overfitting. In this project, you will learn to mitigate overfitting by artificially expanding your dataset using data augmentation.

Your task is to build a robust CNN to classify images of natural scenes, paying close attention to the training and validation performance.

- **Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (Buildings, Forest, Glacier, Mountain, Sea, Street). The dataset contains ~14,000 training images and ~3,000 test images.
- **Task Definition:** Multi-class image classification.
- **Learning Objectives:**
  - Identify overfitting by observing training and validation loss/accuracy curves.
  - Implement data augmentation techniques (`ImageDataGenerator` in Keras or `torchvision.transforms`) such as rotation, shearing, zooming, and flipping.
  - Utilize regularization techniques like Dropout.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script that clearly shows the impact of data augmentation.
  - Presentation slides that plot and compare the training/validation curves of models with and without augmentation and dropout.
- **Evaluation Criteria:**
  - **Generalization:** The primary goal is to build a model that generalizes well, meaning the gap between training and validation accuracy is small.
    has 60,000 images split into 50,000 for training and 10,000 for testing.
- **Deliverables:**
  - **Test Accuracy:** Achieve a test accuracy of over 85% on the validation set.

---

### **Mini-Project 10 - Image Denoising with Convolutional Autoencoders**

This project introduces you to a different application of CNNs: unsupervised representation learning. You will build a convolutional autoencoder to remove noise from images.

Your task is to take a noisy image from the Fashion-MNIST dataset and reconstruct its clean, original version.

- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Available in Keras via `keras.datasets.fashion_mnist`). You will manually add Gaussian noise to the dataset images to create the noisy inputs.
- **Task Definition:** Image-to-Image translation (denoising).
- **Learning Objectives:**
  - Understand and implement an autoencoder architecture, consisting of an encoder and a decoder.
  - Use `Conv2D` layers for the encoder and `Conv2DTranspose` or `UpSampling2D` layers for the decoder.
  - Train a model in an unsupervised manner to learn a compressed representation of the data.
- **Deliverables:**
  - A Jupyter Notebook (`.ipynb`) or Python script with your autoencoder implementation.
  - Presentation slides explaining the autoencoder architecture and showing a visual comparison of original, noisy, and denoised images.
- **Evaluation Criteria:**
  - **Reconstruction Loss:** Use Mean Squared Error (MSE) between the reconstructed and original images as your loss function and report its final value.
  - **Visual Quality:** The primary evaluation will be a visual inspection of the denoised images to confirm that the model successfully removed the noise while preserving the underlying structure.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/45307183/f9611dce-8b09-4a1d-bb6a-9349562dc68d/62FIT4ATI-Assignment-Instruction.docx)
