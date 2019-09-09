### Table of Contents
- <a href="#goals">Goals and Purpose</a>
- <a href="#types">Types</a>
- <a href="#terms">AI Related Terms</a>
- <a href="#process">AI and ML Process (High-level)</a>
- <a href="#theory">Theory</a>
- <a href="#tradeoffs">AI and ML Concepts, Tradeoffs, Considerations, and Constraints (High-level)</a>
- <a href="#model_selection">Model Selection</a>
- <a href="#model_performance">Model Performance and Potential Issues</a>
- <a href="#model_training">Model Training, Learning, and Execution</a>
- <a href="#model_tuning">Model Validation, Tuning, Complexity Reduction, and Optimization</a>
- <a href="#data">Data, Data Sources, and Data Preparation</a>
- <a href="#requirements">Computing and Infrastructure Requirements and Performance</a>
- <a href="#real-world">Real-World AI and Machine Learning</a>
- <a href="#stats">Statistics</a>
- <a href="#future">AI Current and Future</a>
- <a href="#costs">AI and Machine Learning Costs</a>

<h2><a name="goals">Goals and Purpose</a></h2>

- Derive discoveries, information, patterns, trends, and actionable insights
    + Automated, ad-hoc, and self-service
- Transform data into value
- Build data products providing actionable information while abstracting away technical details
- Improving products for user benefit and experience
- Driving business decisions and solutions
    + Improving the decisions your business makes
        * Decision science uses data to analyze business metrics — such as growth, engagement, profitability drivers, and user feedback — to inform strategy and key business decisions.
        * Decision science uses data analysis and visualization to inform business and product decisions.
    + Inform strategic decisions
    + Inform product changes and drive company KPIs
    + Shift from HiPPO decision making to data-driven decision making
- Automated decision making, predictions, recommendations, and deeper insights
- Competitive advantage, differentiation, future-proofing, and opportunity costs
- Complement business intelligence functions
- Predict and advise
- Shifting between deductive (hypothesis-based) and inductive (pattern- based) reasoning (ref. Booz, Allen, Hamilton)
    + Deductive
        * Formulate hypotheses about relationships and underlying models
        * Carry out experiments with the data to test hypotheses and models
    + Inductive
        * Exploratory data analysis to discover or refine hypotheses
        * Discover new relationships, insights and analytic paths from the data
- Business top-level goals
    + Increase ROI and ROA
    + Increase revenue
    + Increase profit
    + Decrease costs
    + Predict and/or reduce risk
    + Increase operational efficiency
    + Reduce waste
    + Increase customer acquisition, retention, and growth
    + Maximize customer retention and minimize churn/loss (particularly to competitors)
    + Improve customer service
    + Enhance business development
    + Improve business governance
    + Improve business processes
    + Drive data and evidence-driven decisions
    + Improve business agility
    + Decision management and inform decisions
    + Drive new business models
    + Discover new products and services
    + Business reporting and analysis
- Customer goals
    + Great user experience and design
        * Self-evident, or at least self-explanatory
        * Easy to use
        * Easy to understand
        * No extra fluff or unnecessary information
    + Products that are sticky, i.e., maximum stickyness
    + Highly personalized experiences
- Goals for artificial intelligence in the future
    + It just works
    + Great user experience
    + Helps humanity, not hurts
- Generating and delivering actionable insights via
    + Story telling
    + Writing
    + Speaking
    + Reports
    + Dashboards
    + Visualizations
    + Presentation
    + Asynchronous messaging, notifications, insights, and alerts
- Creating intelligent agents able to predict and take action as opposed to only predictive models

<h2><a name="types">Types</a></h2>

- AI (encompasses machine learning and other fields)
    + Weak/Narrow - Non-sentient artificial intelligence that is focused on one narrow task
    + Deep - Machine learning based on deep neural networks, i.e., those with more than one hidden layer, each which learns components of the bigger picture
        * Multiple hidden layers allow deep neural networks to learn simple features of the data in a hierarchy, which are combined by the network
    + Shallow - One so-called hidden layer
        * Can be very wide to achieve similar results to deep, but not as efficient and requires more neurons
    + Soft - Not really used
    + Applied - Building ‘smart’ systems or enhancing or extending software applications, where each AI is built for a single purpose
    + AGI, Strong, Full, Hard, Complete
        * Hard/Complete (AI-hard/AI-complete) - The most difficult problems, where the difficulty of these computational problems is equivalent to that of solving the central artificial intelligence problem—making computers as intelligent as people, or strong AI
        * Full - Synonymous with strong and AGI
        * Strong - Strong AI's goal is to develop artificial intelligence to the point where the machine's intellectual capability is functionally equal to a human's.
        * AGI - A machine with the ability to apply intelligence to any problem, rather than just one specific problem
            - The intelligence of a machine that could successfully perform any intellectual task that a human being can
- Machine learning (ML, and subset of AI)
    + Primary
        * Supervised
            - Regression
            - Classification
        * Unsupervised
            - Clustering
            - Anomaly detection
        * Semi-supervised
        * Reinforcement learning
    + Other
        * Transfer learning
        * Recommender systems
            - Content-based and collaborative-based filtering
        * Ensemble methods
            - Bagging (random forests) and boosting

<h2><a name="terms">AI Related Terms</a></h2>

- Sentient - Able to perceive or feel things
- Consciousness - The state of being awake and aware of one's surroundings (opposite is dead or inanimate object)
    + The awareness or perception of something by a person
    + The fact of awareness by the mind of itself and the world
- Mind - The element, part, substance, or process that reasons, thinks, feels, wills, perceives, judges, etc.
- Think - To employ one’s mind rationally and objectively in evaluating or dealing with a given situation
- Artificial (aka machine or synthetic) consciousness - The aim of the theory of artificial consciousness is to "Define that which would have to be synthesized were consciousness to be found in an engineered artifact"
- Sensing
- Learning
- Remembering
- Reasoning
- Transcendent 
- Conscious 
- Self-aware
- Latent Intelligence

<h2><a name="process">AI and ML Process (High-level)</a></h2>

- Ask the right question
- Obtain the data
- Parse and process the data
- EDA - statistical analysis and data visualization
- Choose models/algorithms and performance metric
- Iterate and improve performance
- Deliver/communicate/present results

<h2><a name="theory">Theory</a></h2>

- No Free Lunch (NFL) theorem
    + There is no model that is a priori guaranteed to work better
    + Must evaluate multiple and make reasonable assumptions
- Universal approximation theorem

<h2><a name="tradeoffs">AI and ML Concepts, Tradeoffs, Considerations, and Constraints (High-level)</a></h2>

<h3><a name="model_selection">Model Selection</a></h3>

- Generalization vs representation
+ Parametric vs non-parametric
    * Parametric examples
        - Simple and multiple linear regresion
    * Non-parametric examples
        - Decision trees
        - KNN - k nearest neighbors
        - SVMs
+ Regressor vs classifier
+ Generative vs discriminative models<sup>1</sup>
    * Generative
        - Gaussian mixture model
        - Hidden Markov model
        - Probabilistic context-free grammar
        - Naive Bayes
        - Averaged one-dependence estimators
        - Latent Dirichlet allocation
        - Restricted Boltzmann machine
        - Generative adversarial networks
    * Discriminative
        - Linear regression
        - Logistic regression
        - Support Vector Machines
        - Maximum Entropy Markov Model
        - Boosting (meta-algorithm)
        - Conditional Random Fields
        - Neural Networks
        - Random forests
- Model performance vs interpretability and explainability
    + Black box vs non-black box algorithms
+ Model complexity vs simplicity (i.e., parsimony)
    * Degree of non-linearity
    * Number, type, and combination of functions and parameters
    * Number and type of inputs/features
    * Start simple and increase complexity, not the other way around
        - Start with single hidden layer for DL for example
+ Model assumptions (e.g., linearity)
+ Handling of redundant or irrelevant features
+ Ability to perform with small data sets
+ Complexity vs parsimony
+ Ability to automatically learn feature interactions
+ Classification via definitive vs probabilistic assigment
+ Output types
    * Continuous numeric
    * Probability: 0 to 1
    * Classification
        - Binary
        - Multiclass/ multinomial
            + One vs all or one vs rest (probability per class and ranked)
            + One vs one (binary classifier for every class)
        - Multilabel (assign to multiple classes)
            + Multiple binary labels
        - Multioutput
            + Multiple labels, each label can be multiclass (more than two classes)
        - Hierarchical
+ Kernel selection (e.g., SVM)
    * Linear
    * Polynomial
    * Gaussian
    * RBF
+ Feature engineering tractability
    * Use NN and DL if intractable
+ Neural networks and deep learning-specific
    * Artificial neuron type
        - Linear threshold unit (LTU)
    * Inputs
        - Sound/audio
        - Time series
        - Text
        - Image
        - Video
        - Structured data
        - Unlabeled and/or unstructured data
        - Labeled data
            + Can be difficult, time consuming, and/or costly to obtain
            + Consider a service, e.g., Mechanical Turk, CrowdFlower, ...
    * Architecture
        - Architecture type <sup>4</sup>
            + Unsupervised Pretrained Networks (UPNs)
                * Autoencoders
                * Deep Belief Networks (DBNs)
                * Generative Adversarial Networks (GANs)
            + Convolutional Neural Networks (CNNs)
                * Image modeling
                * Architecture types
                    - VGG
                    - AlexNet
                    - ResNet
                    - Inception network
                    - LeNet
            + Recurrent Neural Networks
                * Sequence modeling
                    - E.g., Long Short-Term Memory (LSTM)
            + Recursive Neural Networks
        - Number of hidden layers
        - Layer type
            + Fully connected (aka dense)
            + Restricted Boltzman Machine (RBM)
            + Autoencoders
        - Number of neurons per layer
        - Network topology and interconnections and interconnection types between neurons
        - Network depth vs width
            + Parameter efficiency (increased depth reduces neurons required)
        - Outputs and output type
            + Output neurons (same number as training data outputs)
                * Single neuron: real-valued numeric (regression)
                * Single neuron: binary classification (classification)
                * Multiple neurons: class scores/probabilities (classification)
                * Multiple neurons: binary assignment per class (classification)
            + Output controlled by output layer activation functions
        - Activation functions and forward propogation
            + Output values, e.g., 0 to 1, -1 to 1, ...
            + Nonzero is 'activated'
            + Examples
                * Linear
                * Rectified linear units (ReLU), Leaky ReLU, Randomized leaky ReLU (RReLU), Parametric leaky ReLU (PReLU)
                * Sigmoid and hard sigmoid
                * Tanh, hard Tanh, rational Tanh
                * Softmax and hierarchical softmax
                * Softplus
                * Softsign
                * Exponential linear unit (ELU)
                    - Slower, but very high performing
                * Cube
            + Hidden activations scaling (similar to inputs)
            + Output activations scaling (similar to outputs)
    * Algorithms
        - First and second order
            + First-order partial derivatives (Jacobians) vs Second-order partial derivatives (the Hessians)
    * Translation invariance
- Choice of loss function
    + Regression examples
        * L1 and L2 loss
        * Mean squared error (MSE)
        * Mean squared log error (MSLE)
        * Root mean squared error (RMSE)
        * Mean absolute percentage error (MAPE)
        * Mean absolute error (MAE)
    + Classification examples
        * Cross entropy (RMSE, Multiclass, ...)
        * Hinge loss
        * Logistic loss
        * Negative log likelihood
        * Exponential log likelihood
    + Reconstruction
        * Entropy loss
    + Categories
        * Squared Loss
        * Absolute loss

<h3><a name="model_performance">Model Performance and Potential Issues</a></h3>

- Overfitting (variance)
    + Noise - errors and outliers
    + Not enough data
    + Overly complex model
    + Diff between test and training error, when test error greater
    + Identifiers
        * Low training error, high dev/test error
    + Solutions
        * Data augmentation (training data only) and/or more data (more data = less regularization)
        * Reduce noise and/or increase the signal-to-noise ratio
        * Reduce model complexity (see Model Complexity and Reduction section)
        * Batch normalization
        * Regularization
        * Ensemble methods
        * Complexity reduction
            - Trees
                + Pruning
                + Max depth
    + NOTE: ensure not due to different distributions between training and dev/test set data <sup>6</sup>
- Underfitting (bias)
    + Opposite of overfitting
    + Identifiers
        * High training error
        * Training error higher than dev/test error
    + Solutions
        * Increase model complexity and reduce regularization (if applicable)
        * Feature engineering
        * Select more powerful and complex model (e.g., neural networks)
- Well performing model
    + Low training and dev/test errors, and both similar
- Performance metric selection
    + Single number metric (preferred)
    + Precision vs recall vs F1 score
    + ROC and AUC
    + Satisficing vs Optimizing metrics
- Performance metric tradeoffs
    + Cost of false positives vs false negatives
- Error types
    + Error types 1 and 2
    + Out-of-sample error (generalization error)
        * Sum of: bias error + variance error + irreducible error
    + In-sample vs out-of-sample errors
- Inability to achieve desired performance level
    + Ill-posed problem
    + Intractability
        * Bad data and/or bad algorithm
        * Identifiers
            - High training error and test error, and unable to improve
- Collinearity, multicollinearity, correlation, ...
- Confounding variables
- Missing features
- Global vs local minima
    + Local minima almost impossible in deep learning due to number inputs and lack of simultaneously non-zero gradients
- Linear separability
- Bayes optimal, Bayes error (minimum possible error, human error a proxy), and avoidable bias (diff between bayes and training errors)
- Vanishing, exploding, and unstable gradients
- Slow learning, training, prediction, model results, ...
- Internal Covariate Shift problem
- Statistical, algorithmic, and cognitive biases
    + Sampling bias 
    + Differing training/testing distributions (inputs and activations)
    + Data
    + Cognitive
    + Sample or selection bias
    + Confirmation bias
    + Algorithmic bias <sup>3</sup>
- Neural networks and deep learning-specific
    + Activation function saturation
    + Dying or comatose ReLUs
        * Only output 0, neuron goes dead, once dead -> stays dead
- Data-related
    + Balanced vs imbalanced data
        * Equal proportion of target values
    + Data availability, amount, and feature space span
        * Small data sets
            - Selecting models that excel with small data sets
            - Sampling noise
            - Transfer learning very helpful for DL with small data sets
        * Moderate to large data sets
            - Sampling bias
        * Sparse data
        * More data allows for simpler models/algorithms/architectures/hyperparameters and less feature engineering (and vice versa)
        * Resources
            - http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/35179.pdf
            - https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/acl2001.pdf
    + Curse of dimensionality and representative data
        * Exponential increase in data needed to prevent loss of predictive power
        * Data must fill as much as the feature space as possible, and be as balanced as possible
    + Data (features) not IID (independent and identically distributed)
    + Data mismatch <sup>6</sup>
        * Data quality and distribution between training and dev/test
        * Example for images: image data for training significantly higher quality and resolution than typical unseen data passed to classifier, hence different data distributions
    + Data quality
        * Signal to noise ratio
        * Data veracity
        * Errors
        * Outliers
        * Missing data
        * NAs
        * Irrelevant features
- Computing requirements
    + CPU/GPU processing power and speed
    + System memory (RAM)
    + Disk storage
    + I/O (disk, network, ...)
- Data leakage

<h3><a name="model_training">Model Training, Learning, and Execution</a></h3>

- Learning type
    + Supervised
    + Unsupervised
    + Semi-supervised
        * Much more unlabled data
        * examples
            * Apply a label to clusters
            * Fine tuning neural networks trained with unsupervised methods
    + Reinforcement
        * Agent
        * Environment
        * Actions
        * Policy
        * Rewards
        * Credit assignment problem
    + Transfer
        * Reuse similar network lower layers
        * Requires much less training data
        * Speeds up training
        * Frozen layers and stop gradient
        * Model zoo for models
- Machine learning algorithm families <sup>5</sup>
    + Information-based learning
    + Similarity-based learning
    + Probability-based learning
    + Error-based learning
- Offline (batch) vs online learning
    + Offline
        * Train and deploy
    + Online (incremental)
        * Limited computing resources
        * Usually done offline
        * Requires close monitoring of input data for data quality and anomaly/outlier detection, and also to model performance over time
    + Out-of-core (type of online learning)
        * Train on datasets too big to fit into system memory
        * Incremental loading of data into memory and training
- Instance-based vs model-based learning
    + Comparison vs pattern/relationship detection
- Iterative learning
    + Gradient descent
        * Batch (full dataset)
        * Mini-batch (small datasets)
        * Stochastic gradient descent (single instance)
    + KNN
- Gradient descent
    + Random initialization
    + Convergence vs divergence
    + Global vs local minimum
    + Convex vs non-convex functions
    + Weight space vs error/loss space
    + Early stopping
- Gradient descent vs normal equations
    + Depends on computational complexity and speed/time
- Batch normalization (BN)
    + Speeds training by helping gradient descent and parameter scaling and ranges
    + Reduces and normalizes dimensional scaling for activations (similar to input normalization)
    + Reduces overfitting without model information loss (can use less dropout)
    + May allow increased learning rate
- Gradient clipping (for exploding gradients)
- Unsupervised Pretraining
    + RBM or autoencoders
- Model pretraining, transfer learning, and frozen layers
    + Pretraining for weight initialization
    + Model zoos
    + Lower layer reuse
- Max margin learning
- Initilization strategy
    + He initialization
    + Glorot initialization
    + Xavier initialization
    + Random initializations
    + ESN-based Initialization
- Sparse data
    + Dual averaging, aka Follow The Regularized Leader (FTRL)
- Parameters vs hyperparameters
- Training, dev (aka validation), and test dataset sizes (proportions) and distributions
    + Distributions should be the same for all datasets
    + Entire datasets usually not necessary if huge, can use sample
    + Example splits for smaller datasets
        * 80/20
        * 70/30
        * 60/20/20
    + Example splits for very big datasets (tune to dev and test size needed for good performance evaluation)
        * 98/1/1
- Execution models
    + Sequencing
        * Serial execution
        * Parallel execution
    + Scheduling
        * Streaming execution
        * Batch execution
- Automated machine learning (AML/AutoML)
    + Model/algorithm selection
    + Hyperparameter tuning
    + Diagnostics
    + Automation
- Distributed training

<h3><a name="model_tuning">Model Validation, Tuning, Complexity Reduction, and Optimization</a></h3>

- Resampling
    + Cross-validation
    + Bootstrap
- Hyperparameters and model adjustments
    + Hyperparameter values and scaling (choosing appropriate scale for search and tuning)
    + NN and DL <sup>4, 6</sup>
        * Architectural
            - Number artificial neurons (hidden and output layers)
            - Number and type of hidden layers
        * Momentum (exponentially weighted average applied to gradients and has increased learning rate effect)
            - Beta (typically 0.90)
        * RMSProp (exponentially weighted average applied to element-wise squared gradients with gradient adjustment scaled, prevents explosion and helps convergence)
            - Beta (typically 0.999, and different from momentum beta hyperparameter)
            - Epsilon (typically 10^-8)
        * Adam
            - Hyperparameters for both momentum and RMSProp
        * Learning rate
            - Alpha
        * Learning rate decay/annealing
        * Regularization
            - Dropout percentage (network and/or per layer)
                + 0.50 a good starting point, but use trial and error
            - Lamda: L1 and L2
        * Activation functions
        * Weight initialization strategy
        * Batch normalization (typically applied to z, not a)
            - Beta (learnable model parameters)
            - Gamma (learnable model parameters)
        * Loss functions
        * Batch/Mini-batch size
            - Batch: Full dataset
            - Mini-batch: Subsets of full dataset
                + Single instance as extreme case
        * Epochs and iterations
        * Normalization scheme for input data (vectorization)
        * Frozen layers (pre-training and transfer learning) vs fine tuning layers
        * Convolutional NN
            - Filter size (height, width, and depth, where depth matches depth of input)
                + Note: can use 1x1 convolution
            - Number of filters
                + Number of filters is the same as depth as activations
            - Stride
            - Layers
                + Input
                + Convolutions
                + Pooling (max and average, often part of same layer as convolution)
                    * Hyperparameters: Filter size and stride (not learnable)
                    * No learnable parameters associated with pooling layers
                + Fully connected (dense)
                + Bottleneck
                    * Shrink before making larger
                    * Reduces computational cost
                + Output
    + Trees
        * Gini vs entropy for impurity
- Hyperparameter tuning and optimization methods
    + Grid search
    + Randomized search for large search space
    + Coordinate Descent
    + Learning rate reduction (learning schedules)
        * Predetermined piecewise constant learning rate
        * Performance scheduling
        * Exponential scheduling
        * Power scheduling
- Pandas vs caviar <sup>6</sup>
- Ensemble methods
    + Hard voting (majority vote)
    + Soft voting
    + Bagging and boosting
- Kernel selection (e.g., SVM)
- Learning curves
- Bias correction
- Speed up DNN training <sup>5</sup>
    + Apply a good initialization strategy for the connection weights
    + Use a good activation function
    + Use Batch Normalization
    + Use parts of a pretrained network
    + Use a faster optimizer (updater) than the regular Gradient Descent optimizer
        * Momentum optimization
        * Nesterov Accelerated Gradient
        * AdaGrad
        * RMSProp
        * Adam (momentum and RMSProp)
        * AdaADelta
        * SGD
        * Conjugate Gradient
        * Hessian Free
        * LBFGS
        * Line Gradient Descent
- Subset selection
    + Best subset selection
    + Stepwise selection (forward and backward)
- Shrinkage and regularization
    + Ridge regression
    + The Lasso
    + Elastic Net
    + Neural networks
        * Early stopping
        * L1 and L2 regularization
            - L1 good for sparse models
        * Dropout
        * Drop connect
        * Max-norm regularization
        * Data augmentation
            - Learnable and not noise
            - Apply when dataset small
            - Examples
                + Images
                    * Shift/translate, rotate, resize, shearing, local warping
                    * Brightness, contrast, saturation, and hue
                    * Color shifting
                        - Add distortions to color channels (e.g., RGB individually)
                        - Can help simulate changing lighting conditions (sun, ...)
                        - PCA color augmentation
                    * Flip across axis (mirroring)
                    * Cropping (random, closely cropped, ...)
- Dimension reduction
    + PCA
    + Partial least squares
- Tree methods
    + Pruning
    + Max depth
- Feature selection, engineering, and extraction
    + Collinearity, multicollinearity, correlation, ...
    + Confounding variables
    + Missing features
    + Feature extraction via dimensionality reduction
- Pseudo labeling
- For image DL
    + Multi-crop (e.g., 10-crop)

<h2><a name="data">Data, Data Sources, and Data Preparation</a></h2>

- Analytics base table (ABT) and data quality report <sup>5</sup>
- Data types
    + Structured data
    + Unstructured data
    + Semi-structured data
    + Streaming data
    + Batch data
- Data acquisition, ingestion, and processing
    + Disparate data sources
- Data preparation
    + Feature scaling
        * Standardization
        * Min-max (normalization)
        * Normally normalize inputs and outputs to be mean 0 and stdev 1
    + Feature encoding
        * One hot encoding for categorical variables
    + Missing values handling
    + Imputation

<h2><a name="requirements">Computing and Infrastructure Requirements and Performance</a></h2>

- CPU processing speed and computing time
    + Data processing, model training, and prediction speed
- CPU vs GPU
- System memory (RAM)
- Disk storage
- I/O: disk, network, etc.
- Cost
- Training and prediction speed
- Computational complexity

<h2><a name="real-world">Real-World AI and Machine Learning</a></h2>

- AI and machine learning process
    + CRISP-DM, etc.
- AI and machine learnng in production
    + Deploying to production and maintenance
    + Scalability
    + Online vs offline learning
    + Distributed learning
    + Model types and implementation lanaguages
    + Data access (if required), particularly for disparate data sources
    + Temporal drift, i.e., data changing over time
        * Avoid stale models
    + Model monitoring and accuracy tracking over time
- Working with large datasets <sup>1</sup><sup>,</sup><sup>2</sup>
    + Partition data vs partition execution
    + Software-specific memory allocation
    + Full dataset vs subset (down sampling, e.g., 1/10 to 1/100)
    + Local development vs remote compute resources (e.g., AWS) with lots of RAM
    + Data format transformation
    + Streaming machine learning
    + Progressively loading
    + Direct database connection for machine learning
    + Big data platformes (e.g., Spark and MLLib)
    + Parallel and distributed computing and associated communications cost and complexity
    + Parameter server
- Model sharing and formats
    + [ONNX](https://onnx.ai/) - Open format to represent deep learning models

<h2><a name="stats">Related Statistics</a></h2>

- Prediction vs inference
    + Prediction
        * Model an output variable Y (aka response or dependent variable) as a function of a set of input variables X (aka inputs, features, predictors, independent variables)
            - Estimate f such that: Y = f(X) + err
        * Consider reducible vs irreducible errors
    + Inference
        * Understand how Y varies with X and any underlying relationships between Y and X, particularly wrt. individual predictors and each of their impact on the response
        * Understand the degree to which Y varies with each predictor, e.g., linear, non-linear, ...

<h2><a name="future">AI Current and Future</a></h2>

- Hype cycle
    + AI winter and trough of disillutionment
- Expectation management
- Actual AI vs simpler machine learning algorithms
- Machine learning by itself is not AI, intelligent systems are built from machine learning models and much more
- AI and machine learning automations and future advancements
    + Automated learning (AutoML)
        * Auto-sklearn, TPOT, ...
- AI limitations
    + Unsupervised learning to some extent
    + Solving multiple problems at once
- Current and future active areas of innovation and progress
    + Deep learning
        * Faster training NN architectures
        * Improved NN performance via new hyperparameters, techniques, and architectures
    + Reinforcement learning
        * Deep Q-learning
    + Computing (power, distributed, parallel, GPUs, cheap CPUs for parallel, ...)
    + Evolution algorithms <sup>7</sup>
        * Genetic algorithms
        * Novelty search
    + Neuroevolution
    + Bayesian deep learning
    + Hardware
        * Nvidia
        * Intel
        * Google
        * Tesla
- Digital twins

<h2><a name="costs">AI and Machine Learning Costs</a></h2>

- Financial
    + Employees/talent
    + Tools and software
    + Cloud computing
    + Data and data labeling 
    + Research
    + Computing equipment
- Non-financial
    + Coordination
    + Communication
    + Opportunity

## References

1. [7 Ways to Handle Large Data Files for Machine Learning](https://machinelearningmastery.com/large-data-files-machine-learning/)
2. [Beyond Distributed ML Algorithms: Techniques for Learning from Large Data Sets](https://iwringer.wordpress.com/2015/10/06/techniques-for-learning-from-large-amounts-of-data/)
3. [Joy Buolamwini - TED Talk](https://www.ted.com/talks/joy_buolamwini_how_i_m_fighting_bias_in_algorithms)
4. [Deep Learning by Josh Patterson and Adam Gibson - O'Reilly](https://www.amazon.com/Deep-Learning-Practitioners-Josh-Patterson-ebook/dp/B074D5YF1D/ref=mt_kindle?_encoding=UTF8&me=)
5. [Fundamentals of Machine Learning for Predictive Data Analytics](https://www.amazon.com/Fundamentals-Machine-Learning-Predictive-Analytics-ebook/dp/B013FHC8CM/ref=sr_1_1)
6. [deeplearning.ai Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
7. [AI and Deep Learning in 2017 – A Year in Review](http://www.wildml.com/2017/12/ai-and-deep-learning-in-2017-a-year-in-review/)
