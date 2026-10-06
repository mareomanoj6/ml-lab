## Experiment 1
Implement linear regression with one variable on the California Housing dataset to predict housing prices based on a single feature (e.g., the average number of rooms per dwelling).

1. Load the dataset and select a feature.
2. Initialize weights and bias.
3. For gradient descent: compute prediction, calculate error, update weights.
4. For normal equation: use the closed-form formula to find weights.
5. Calculate MSE and R-squared.
6. Plot the fitted line.

## Experiment 2
Implement polynomial regression on the Auto MPG dataset to predict miles per gallon (MPG) based on engine displacement. Compare polynomial regression results with linear regression.

1. Load the dataset.
2. Transform the input feature into polynomial features of degree N.
3. Fit a linear regression model to these transformed features.
4. Repeat for different degrees of N.
5. Compare performance using MSE and R-squared.
6. Plot the resulting curves.

## Experiment 3
Implement Ridge and Lasso regression on the Diabetes dataset. Compare the performance of these regularized models with standard linear regression.

1. Load the dataset.
2. Define a penalty term (L2 for Ridge, L1 for Lasso).
3. Add the penalty to the linear regression cost function.
4. Tune the regularization strength using cross-validation.
5. Fit the model and compare it with standard linear regression.
6. Evaluate using MSE and R-squared.

## Experiment 4
Implement a logistic regression model to predict the likelihood of a disease using the Pima Indians Diabetes dataset. Compare the performance with and without feature scaling.

1. Load the dataset.
2. Pass linear combination of inputs through a sigmoid function.
3. Use binary cross-entropy as the loss function.
4. Update weights using gradient descent.
5. Evaluate accuracy, precision, recall, and F1-score.
6. Compare results with and without feature scaling.

## Experiment 5
Implement a Naïve Bayes classifier to categorize text documents into topics using the 20 Newsgroups dataset. Compare the performance of Multinomial Naïve Bayes with Bernoulli Naïve Bayes.

1. Preprocess text into word counts or binary indicators.
2. Calculate prior probabilities for each class.
3. Calculate conditional probabilities for each word given the class.
4. For a new document, multiply prior by conditional probabilities.
5. Assign the class with the highest probability.
6. Compare Multinomial vs Bernoulli variants.

## Experiment 6
Estimate the parameters of a logistic regression model using MLE and MAP on the Breast Cancer Wisconsin dataset. Compare the results and discuss the effects of regularization.

1. Load the dataset.
2. For MLE: Maximize the likelihood function of the observed data.
3. For MAP: Maximize the posterior (likelihood times a prior).
4. Use L1 or L2 priors for MAP regularization.
5. Estimate parameters using optimization.
6. Compare parameter weights and model performance.

## Experiment 7
Use MLE and MAP to estimate the parameters of a multinomial distribution on the 20Newsgroups dataset. Explore the impact of different priors on the estimation.

1. Load the text dataset.
2. For MLE: Estimate probabilities based on relative word frequencies.
3. For MAP: Incorporate a prior (e.g., Dirichlet) to smooth probabilities.
4. Calculate parameters for the multinomial distribution.
5. Compare how different priors affect the estimates.
6. Evaluate the resulting distribution.

## Experiment 8
Implement the K-Nearest Neighbors (KNN) algorithm for image classification using the Fashion MNIST dataset. Experiment with different values of K and analyze their impact on model performance.

1. Store all training data points.
2. For a new image, calculate distance to all training points.
3. Find the K closest neighbors.
4. Take a majority vote of the neighbors' classes.
5. Assign the most frequent class to the image.
6. Test different values of K to optimize accuracy.

## Experiment 9
Implement a Decision Tree classifier using the ID3 algorithm to segment customers based on their purchasing behavior using the Online Retail dataset. Analyze the tree structure and discuss the feature importance.

1. Start with all data at the root node.
2. Calculate information gain for every available feature.
3. Split the data using the feature with the highest gain.
4. Create child nodes for each possible value of that feature.
5. Repeat recursively until nodes are pure or no features remain.
6. Assign the majority class to leaf nodes.

## Experiment 10
Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset. Visualize the decision boundary and discuss how the margin is determined.

1. Load the data and define two classes.
2. Find the hyperplane that maximizes the margin between classes.
3. Identify support vectors that lie closest to the boundary.
4. Solve the optimization problem to find the optimal weights.
5. Classify new points based on which side of the hyperplane they fall.
6. Visualize the boundary and margin.

## Experiment 11
Implement and compare the performance of SVM classifiers with linear, polynomial, and RBF kernels on the Fashion MNIST dataset. Analyze the advantages and disadvantages of each kernel type.

1. Load the dataset.
2. Use a kernel function to map data to a higher-dimensional space.
3. Apply Linear, Polynomial, and RBF kernels.
4. Find the optimal separating hyperplane in the transformed space.
5. Predict classes based on the kernel-mapped boundary.
6. Compare accuracy across different kernels.

## Experiment 12
Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine Quality dataset. Experiment with different numbers of hidden layers and neurons, and discuss how these choices affect the network's performance.

1. Initialize weights and biases for input, hidden, and output layers.
2. Perform forward propagation: compute weighted sums and apply activations.
3. Calculate the error at the output layer.
4. Perform backward propagation to calculate gradients for all weights.
5. Update weights using an optimizer (e.g., SGD).
6. Experiment with different layer sizes and depths.

## Experiment 13
Implement and compare the performance of a neural network using different activation functions (Sigmoid, ReLU, Tanh) on the MNIST dataset. Analyze how each activation function affects the training process and classification accuracy.

1. Load the dataset and build a neural network.
2. Replace the activation function with Sigmoid, ReLU, or Tanh.
3. Train the network and track the loss over epochs.
4. Evaluate classification accuracy.
5. Compare training speed and convergence rates.
6. Analyze which function performs best for the task.

## Experiment 14
Implement and compare hierarchical (agglomerative) and partitional (K-means) clustering algorithms on the Mall Customers dataset. Discuss the strengths and weaknesses of each method based on clustering results and evaluation metrics.

1. Preprocess the dataset.
2. For K-means: Randomly initialize centroids, assign points, and update centroids.
3. For Hierarchical: Start with each point as a cluster and merge closest pairs.
4. Evaluate using inertia and silhouette scores.
5. Visualize the clusters.
6. Compare the grouping results of both methods.

## Experiment 15
Implement and apply K-means clustering to the Digits dataset. Experiment with different numbers of clusters and evaluate the clustering results using metrics such as inertia and silhouette score. Analyze how the choice of K affects clustering performance.

1. Choose the number of clusters K.
2. Randomly initialize K cluster centers.
3. Assign each data point to the nearest center.
4. Update centers by calculating the mean of assigned points.
5. Repeat assignment and update until centers stabilize.
6. Test different K values and use the elbow method.

## Experiment 16
Implement bootstrapping and cross-validation on the Iris dataset. Compare the model performance metrics (e.g., accuracy, F1-score) obtained using these resampling methods. Discuss the advantages and disadvantages of each method.

1. Load the dataset.
2. For Bootstrapping: Create multiple samples by sampling with replacement.
3. Train and evaluate the model on each bootstrap sample.
4. For Cross-validation: Split data into K folds.
5. Train on K-1 folds and test on the remaining fold, repeating K times.
6. Compare stability and accuracy of both methods.

## Experiment 17
Implement bagging and boosting ensemble methods on the Titanic dataset. Compare the performance of both methods in terms of accuracy, precision, recall, and F1-score. Discuss how each method improves model performance and their respective strengths and weaknesses.

1. Preprocess the dataset.
2. For Bagging: Train multiple base models on bootstrap samples and average results.
3. For Boosting: Train models sequentially, focusing on previous errors.
4. Combine weak learners into a strong ensemble.
5. Evaluate using accuracy and F1-score.
6. Compare the variance reduction of bagging vs bias reduction of boosting.

## Experiment 18
Investigate the bias-variance tradeoff using polynomial regression on the Boston Housing dataset. Plot the training and validation errors for various polynomial degrees and discuss the tradeoff between bias and variance.

1. Load the dataset.
2. Fit polynomial regression models of increasing degrees.
3. Split data into training and validation sets.
4. Calculate error for both sets for each degree.
5. Plot errors to find the point where validation error increases.
6. Analyze where the model shifts from underfitting to overfitting.

## Experiment 19
Implement and compare Logistic Regression and Decision Trees on the Adult Income dataset for predicting income levels. Evaluate both models based on performance metrics and interpretability.

1. Preprocess the dataset.
2. Train a logistic regression model using a sigmoid output.
3. Train a decision tree model using recursive splitting.
4. Evaluate both using precision, recall, and F1-score.
5. Analyze the decision boundaries of both models.
6. Compare model interpretability versus predictive power.

## Experiment 20
Implement and perform hyperparameter tuning for a neural network on the Fashion MNIST dataset. Experiment with different learning rates, batch sizes, and epochs, and discuss the impact on model performance.

1. Load the dataset and define a network architecture.
2. Select a range of values for learning rate, batch size, and epochs.
3. Train the network for each combination of hyperparameters.
4. Evaluate performance on a validation set.
5. Identify the combination that minimizes error.
6. Analyze how each parameter affects convergence and accuracy.
