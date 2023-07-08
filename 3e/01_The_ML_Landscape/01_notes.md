# Chapter 1 - The Machine Learning Landscape

## What is Machine Learning?

ML is the science and art of programming computers to learn from data.

## Why use Machine Learning?

Using ML to dig into large amounts of data to discover hidden patterns is called *data mining*.

ML is great for:

- Problems that require lots of fine-tuning or rules.
- Complex problems which traditional approaches don't work well.
- Fluctuating environments with lots of new data.
- Getting insights about complex problems and large amounts of data.

## Examples of Applications

- Prediction.
- Classification.
- Anomaly Detection.
- Natural Language Processing.
- Segmentation and Clustering.
- Data Visualization and Dimensionality Reduction.
- Recommendation Systems.
- Intelligent Agents / Bots.

This list is not exhaustive.

## Types of Machine Learning Systems

We can classify ML systems into broad categories using the following criteria:

1. How they are supervised during training.
   - Supervised, unsupervised, semi-supervised, self-supervised, and others...
2. Whether they can learning incrementally or not.
3. Whether they work by comparing new data to known data, or by detecting patterns and building a predictive model.
   - Instance-based vs. model based.

These criteria are not exclusive, you can combine them in any way.

### Training Supervision

We can classify ML systems by the amount and type of supervision during training.

1. Supervised learning
2. Unsuperivsed learning
3. Semi-supervised learning
4. Self-supervised learning
5. Reinforcement learning

#### Supervised Learning

The training data set contains the desired solutions, called labels or targets.

Typical supervised learning tasks are either classification or prediction (regression).

#### Unsupervised Learning

The training data in unlabeled.

Some unsupervised learning tasks are:

- Clustering: to determine groups.
- Data visualization.
- Dimensionality reduction to simplify data without loose info.
  - Feature extraction is a type of dimensionality reduction and the process of merging correlated features / attributes into one.
- Anomaly Detection / Novelty Detection
- Association rule learning
  - Dig into large amounts of data and discover relationships.

#### Semi-supervised Learning

Partially labeled training data:

- Lots of unlabeled data.
- Some labeled data.

Most semi-supervised learning algorithms are a combination of unsupervised and supervised algorithms.

Ex. First perform clustering, and then label the clusters with the labeled data.

#### Self-supervised Learning

Generate a fully labeled dataset from a fully unlabeled set. Once fully labeled, a supervised algorithm can be used.

#### Reinforcement Learning

Reinforcement Learning is very different from the other categories.

Reinforcement Learning involces an agent that can observe the environment, select and perform actions, and get rewards in return (or penalties).

It must learn the best strategy, called a *policy*.

Ex. Learning to walk or play Go.

### Batch vs. Online (Incremental) Learning

Whether or not a ML system can learn incrementally from a stream of incoming data.

#### Batch Learning

Batch learning is not done incrementally and must be trained using all data and then retrained when needed on old and new data.

#### Online (Incremental) Learning

Incremental learning is done by feeding the system data sequentially in small groups called mini-batches. Each learning step is fast and cheap.

Useful for systems that need to adapt to change extremely rapidly.

Also used to train models on huge datasets that cannot fit in one machine's main memory.

- Called "out of core" learning.

The **learning rate** is the parameter used to set how fast system should adapt to changing data.

- A high learning rate will adapt to new data quickly, but will forget the old data.
- A low learning rate will give the system more inertia, but will also be less sensitive to noise in the new data.

A big challenge is not to feed incremental / online systems bad data, as that will lead to a performance decline.

- Monitor systems closely and potentially turn off or decrease the learning rate if we detect a drop in performance.

### Instance-Based vs. Model-Based Learning

#### Instance-based Lerning

The system learns examples "by heart" and then generalizes to new cases using a *measure of similarity*.

#### Model-based Learning

The system is trained to build a model of the examples and then use the model to make predictions.

We use a *utility function / fitness function* to measure how good our model is, or a *cost function* to measure how bad your model is.

## Main Challenges of Machine Learning

What can go wrong when selecting a model and training it?

1. Bad Model
2. Bad Data

### Insufficient Quantity of Training Data

ML Models need lots (LOTS) of data to work properly.

"The Unreasonable Effectiveness of Data"

- Almost all models perform well on a complex problem once given enough data.

But it's not cheap or easy to get lots of data, so don't abandon algorithms just yet!

### Nonrepresentative Training Data

Training data needs to be representative of all cases you want to generalize to.

Models trained on nonrepresentative training data will be less accurate.

Sampling bias can lead to nonrepresentative data. Sampling bias arises when the sampling method favors or excludes specific characteristics or groups.

Sampling noise is the variability or randomness introduced to a data set due to the inherent nature of sampling. When data is collected, it is often not feasible to include evey possible example from the target population, and your selected sample may not perfectled represent the entire population.

### Poor-Quality Data

Poor-quality data leads to poor performance.

Data scientists must clean up their training data:

- Disard errors and outliers.
- Fill in missing data with medians, or ignore it.

### Irrelevant Features

Training data must contain useful and relevant features to train on, or else: "Garbage in, garbage out".

Coming up with good features is called *Feature Engineering* which involves:

- Feature Selection (Selecting the most useful features to train on.)
- Feature Extraction (Combining existing features into more useful ones, dimensionality reduction algorithms can help.)
- Creating new features by gathering more data.

### Overfitting the Training Data

Indicated by models performing well on the training data but failing to generalize well.

Overfitting happens when the model is too complex relative to the amount of noisiness of the training data. Here are possible solutions.

- Simplify the model used, reduce the number of attributes in the training data, or constrain the model (regularization).
- Gather more training data.
- Reduce the noise in the training data.

Constraining a model to make it simpler and reduce overfitting is called *regularization*.

The amount of regularization to apply during learning can be controlled by a *hyperparameter*

- A hyperparameter is a parameter of a learning algorithm, not a model.
- Tuning hyperparameters is an important part of building a ML system.

### Underfitting the Training Data

Occurs when the model is too simple to learn the structure of the data.

Fix by:

- Select a more powerful model with more parameters.
- Use better features (Feature engineering)
- Reduce constraints / regularization

## Testing and Validating

In order to know how well your model works, you need to test it on cases after training.

Typically we split our data set into a training and testing set with a 80% / 20% ratio.

The error we measure on new cases is called the *generalization error*.

If the training error is low, but the generalization error is high, we know our model is overfitting the data.

### Hyperparameter Tuning and Model Selection

To evaluate between different types of models we further take part of our training data set to create a validation set (dev set) to test different candidate models, as well as different hyperparameters.

Once we are happy with a particular candidate model, we then train on all the training data to create the final model, and then evaluate on the test set.

- Take care that the validation set isn't too small or we could select a sub-optimal model.
- Take care that the validation set isn't too big either as that reduces the size of the training set.
  - One solution is to use cross-validation.

### Data Mismatch

Sometimes you may have a lot of training data but it is not representative of the data used in production.

You can still train the model using this data, but we must make sure the validation set and test set only contain instances of data that are representative of production data.

Then, we hold out another subset of the training set, called a "training dev set".

First we evaluate the error of the training dev set.

- If it is low, we then test on the validation and test sets.
  - If the error on the validation and test sets are high, we know we have nonrepresentative data and need to make the appropriate adjustments (improve data quality).
  - If the error on the validation and test sets are low, we are good to go.
- If it is high, we know our model is overfitting the training data and we need to make appropriate adjustments (simplify the model, get more data, regularization)

## No Free Lunch Theorem

If you make absolutely no assumptions about your data, then there is no reason to prefer one model over another. There is no model that is gaurenteed to work better than another.

The only way to know, is to test them all.

In practice, this is not possible, and as such we do make reasonable assumptions about our data.
