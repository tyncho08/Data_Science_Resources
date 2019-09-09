### Table of Contents
- <a href="#categories">Algorithm Categories</a>
- <a href="#fields">Algorithms and Applications by Field/Industry</a>
- <a href="#data-type">Algorithms and Applications by Data Type</a>
- <a href="#tasks">Algorithms and Applications by Task</a>
- <a href="#regression">Regression (Supervised) - Univariate, Multivariate, ...</a>
- <a href="#classification">Classification (Supervised) - Unary (one-class), Binary, and Multi-class</a>
- <a href="#regularization">Regularization and Overfitting Prevention</a>
- <a href="#clustering">Clustering (Unsupervised)</a>
- <a href="#ensemble">Ensemble Methods (Supervised, Unsupervised)</a>
- <a href="#recommendation">Recommender Systems and Recommendations</a>
- <a href="#neural-nets">Neural Networks and Deep Learning</a>
- <a href="#recognition">Recognition and Computer Vision</a>
- <a href="#anomaly">Anomaly Detection (Supervised, Unsupervised, Semi-supervised)</a>
- <a href="#text-analytics">Text Processing and Analytics</a>
- <a href="#nlp">Natural Language Processing (NLP), Natural Language Generation (NLG), and Natural Language Understanding (NLU)</a>
- <a href="#reinforcement">Reinforcement Learning</a>
- <a href="#model-selection">Model selection, validation, and resampling methods</a>
- <a href="#model-tuning">Model tuning: bias variance tradeoff and model complexity</a>
- <a href="#features">Feature extraction, feature selection, and feature engineering</a>
- <a href="#dimensionality-reduction">Dimensionality Reduction</a>
- <a href="#reality">Virtual and Augmented Reality</a>
- <a href="#information-retrieval">Information Retrieval</a>
- <a href="#logical-reasoning">Logical Reasoning</a>
- <a href="#optimization">Optimization and Search</a>
- <a href="#risk">Quantitative Finance and Risk Management</a>
- <a href="#ranking">Ranking</a>
- <a href="#time-series">Time-series</a>
- <a href="#survival">Survival</a>
- <a href="#forecasting">Forecasting (Wikipedia)</a>
- <a href="#simulation">Simulation</a>
- <a href="#segmentation">Segmentation</a>
- <a href="#experimentation">Experimentation and Experimental Design</a>
- <a href="#embedded">Embedded</a>
- <a href="#hypothesis">Hypothesis Testing</a>
- <a href="#hybrid">Hybrid Solutions and Applications</a>
- <a href="#other">Other Algorithms</a>
- <a href="#polls">Polls and Popularity</a>

<h2><a name="categories">Algorithm Categories</a></h2>

- Decision tree learning
- Association rule learning
- Artificial neural networks
- Inductive logic programming
- Support vector machines
- Clustering
- Bayesian networks
- Reinforcement learning
- Representation learning
- Similarity and metric learning
- Sparse dictionary learning
- Genetic algorithms

<h2><a name="fields">Algorithms and Applications by Field/Industry</a></h2>

- Telecommunications
    + 5G
    + IoT
    + Autonomous vehicles
    + Smart cities
    + Engagement insights whose info is sold back to advertisers
- Marketing
    + Segmentation
    + Ranking/scoring
    + Market basket analysis > location and promotions of items
    + Cohort analysis and segmentation > targeted marketing
    + Customer churn prediction > churn prevention
    + Customer lifetime value forecasting > future business value and predicting growth
    + Targeted and personalized advertising
    + Companies
        * [Appier](https://www.appier.com/)
        * [Voyager Labs](http://voyagerlabs.co/)
- Sales
    + Revenue forecasting and growth
- Security intelligence (security, fraud, and risk analysis)
- Customer relationship management
- AR and VR
- Gaming
    + Examples
        * AlphaGo (Monte-Carlo Tree Search)
        * DeepBlue
        * Watson Jeapordy
- Health care
- Insurance
- Retail
    + Recommendation engines
    + Virtual reality fitting systems
    + Shopping experience
- Autonomous transportation
    + Companies
        * Google
        * Apple
        * Uber
        * Lyft
        * Tesla
        * Waymo

<h2><a name="data-type">Algorithms and Applications by Data Type</a></h2>

Ref <sup>2</sup>
- Sound/Audio
    + Voice detection/recognition
    + Voice search
    + Speaker identification
    + Sentiment analysis
    + Flaw detection (engine noise)
    + Fraud detection (latent audio artifacts)
    + Speech-to-Text
- Time Series/sequence
    + Log analysis/Risk detection
    + Enterprise resource planning
    + Predictive analysis using sensor data
    + Business and economic analysis
    + Recommendation engine
    + Examples and algorithms
        * Web log
            - RNN
        * Time series in general (has time stamp)
            - RNN
        * Sensors and measures over time
            - RNN
        * Arbitrarily long sequence that may take full input data
            - RNN
            - Markova model with large hidden state space
        * Fixed length sequence
            - CNN
            - Multilayer perceptron
- Text
    + Sentiment analysis
    + Augmented search, theme detection
    + Threat detection
    + Fraud detection
    + Named-entity recognition
- Image
    + Facial recognition and expression recognition
    + People identification
    + Image search
    + Machine vision
    + Photo clustering
    + Image recognition/classification
        * Is it a certain class or multiple classes (e.g., cat, car, ...)
    + Object recognition and detection
        * Detection is the location of the recognized object in the image (i.e., localization)
            - Output is bounding box (b_x, b_y, b_h, B_w), is object recognized in image, and class label(s)
            - Loss function calculation depends on whether the object is detected in the image
            - Sliding window detection (window size and stride)
                + Pass window as input to CNN
    + Landmark detection
        * X,Y point pairs representing individual landmarks in the image
        * Useful for emotion detection, filters, pose detection, ...
    + Algorithms
        * CNN
- Video
    + Motion detection
    + Real-time threat detection
    + Gesture recognition
- Unlabeled and/or unstructured data
    + Clustering
    + Anamoly detection (detecting anamolies)
    + Search (detecting similarities)
        * Compare docs, images, sounds, etc., and return similar items
- Labeled data
    + Predictive analytics
        * Regression and classification
            - Hardware failure
            - Health degredation, failure, and disease
            - Customer churn
            - Employee churn
- Columnar/tabular
    + Classic multilayer perceptrons + feature engineering

<h2><a name="tasks">Algorithms and Applications by Task</a></h2>

- Prediction
    + Regression/classification
        * RNN
- Recommendation
- Generative
    + Novel output
        * RNN
- Reconstruction
    + Example: MINST
- Recognition and computer vision
    + Changing images in time (video)
        * LSTM (temporal aspect) with convolutions layers (capture structure/features)
- NLP, NLG, NLU
    + Machine translation
        * CNN
    + Sentiment analysis
        * CNN
    + Sentence classification
        * CNN
- Personal assistant
    + Voice to text then NLP to understand what the user wants to accomplish, then generating text, voice, action
- Anamoly detection
- Reinforcement learning
- Reality capture and reality computing

<h2><a name="regression">Regression (Supervised) - Univariate, Multivariate, ...</a></h2>

- Simple and multiple linear regression
- Tree-based methods (e.g., decision tree or forest)
- Generalized linear models (GLM)
    + Poisson regression, aka log-linear model
- Generalized additive model (GAM)
- Regression with shrinkage (e.g., regularization)
- Stepwise regression
- Ordinary least squares
- Artificial Neural networks (ANN) and deep learning
- Ordinal regression
- Polynomial regression
- Nearest neighbor methods (e.g., k-NN or k-Nearest Neighbors)
- Gradient tree boosting
- Logistic regression
- Nonlinear regression

**Example Applications**

- Stock market predictions and algorithmic trading
    + Companies
        * [Kavout](https://www.kavout.com/)
        * [Sentient](http://www.sentient.ai/)
        * [Genotick](http://genotick.com/)
        * [Numerai](https://numer.ai/)
        * [QPLUM](https://www.qplum.co/)

<h2><a name="classification">Classification (Supervised) - Unary (one-class), Binary, and Multi-class</a></h2>

- Linear
    + Linear discriminant analysis (LDA), aka Fisher's linear discriminant
    + Logistic regression and multinomial logistic regression
    + Bayesian classifiers (as opposed to frequentist)
        * Naive Bayes
    + Perceptron methods
- Decision trees and random forests
- Naive bayes
- Hidden markov model
- Support vector machines (SVM)
    + Least squares support vector machines
- Artificial Neural networks (ANN) and deep learning
- Kernel estimation
    + Nearest neighbor methods (e.g., k-NN or k-Nearest Neighbors)
- One vs Rest and One vs One (binary transformation)
- Gradient tree boosting

**Example Applications**

- Many diseases or issues, including stroke, cancer, ...
    + Cancer detection using cell-free genomes
    + Cardiovascular events prediction (e.g., heart attack, stroke)
    + Companies
        * [Google DeepMind](https://deepmind.com/)
        * IBM's [Watson](https://www.ibm.com/watson/) (Watson for Oncology)
        * Others
            * [Freenome](https://www.freenome.com/)
            * [CureMetrix](http://curemetrix.com/)
- Spam for email
- Smart email categorization (Gmail)
    + Primary, social, and promotion inboxes, as well as labeling emails as important
- Credit decisions
    + Companies
        * [Underwrite.ai](http://www.underwrite.ai/)

<h2><a name="regularization">Regularization and Overfitting Prevention</a></h2>

- Least absolute shrinkage and selection operator (LASSO)
- Ridge regression
- Akaike information criterion (AIC)
- Bayesian information criterion (BIC)

<h2><a name="clustering">Clustering (Unsupervised)</a></h2>

- Hierarchical clustering, aka connectivity-basedclustering and Hierarchical Cluster Analysis (HCA)
    + Single-linkage clustering
    + Complete linkage clustering
    + Unweighted Pair Group Method with Arithmetic Mean (UPGMA), aka average linkage clustering
- Centroid-based clustering
    + k-means
    + k-medoids
    + k-medians
    + K-means++
    + Fuzzy c-means
- Distribution-based clustering
    + Gaussian mixture models via expectation-maximization algorithm
- Density-based clustering
    + Density-based spatial clustering of applications with noise (DBSCAN)
    + Ordering points to identify the clustering structure (OPTICS)
    + Mean-shift
- Canapoy
- Association rule learning
    + Apriori
    + Eclat
- Topic modeling (text data)
- Fractal
- Guassian mixture models

**Example Applications**

<h2><a name="ensemble">Ensemble Methods (Supervised, Unsupervised)</a></h2>

- Bootstrap aggregating (bagging)
    + Random Forests and ExtraTrees
- Boosting
    + AdaBoost
    + Gradient boosting
    + Boost by majority
    + BrownBoost
    + xgboost
    + MadaBoost
    + LogitBoost
    + LPBoost
    + TotalBoost
- Pasting
- Bayesian model averaging (BMA)
- Weak learner theory
- Stacking (stacked generalization) and Blending
- Bayes optimal classifier
- Bayesian parameter averaging (BPA)
- Bayesian model combination (BMC)
- Bucket of models

<h2><a name="recommendation">Recommender Systems and Recommendations</a></h2>

- Collaborative filtering
- Content-based filtering
- Graph-based methods

**Example Applications**

- Netflix
    + Increase engagement, retention, and revenues
    + Examples
        * "Because you watched ..."
        * "Top picks for ..."
        * Recommendations by category
            - Trending Now
            - Neflix originals
            - TV Documentaries
- Amazon
    + Increase average order size and therefore sales (studies show between 5.9 to 30%)
    + Examples
        * "Customers who bought this item also bought"
        * "Customers who viewed this item also viewed"
        * "What other items do customers buy after viewing this item?"
        * "Recommendations for you in ..." (e.g., "Recommended for You in Kindle Books")
        * "New for you"
        * "Inspired by your shopping trends"
        * "Inspired by your Wish List"
- Robo-advisors and portfolio rebalancing
    + [Weathfront](https://www.wealthfront.com/)
    + [Betterment](https://www.betterment.com/)
- Spotify
    + [Daily Mix](https://support.spotify.com/us/using_spotify/search_play/daily-mix/)
- Personalized news feeds, including Facebook

<h2><a name="neural-nets" href="http://www.asimovinstitute.org/neural-network-zoo/">Neural Networks and Deep Learning</a></h2>

- Feed forward neural networks (FF or FFNN) and perceptrons (P)
- Radial basis function (RBF)
- Hopfield network (HN)
- Markov chains (MC or discrete time Markov Chain, DTMC)
- Boltzmann machines (BM)
- Restricted Boltzmann machines (RBM)
- Autoencoders (AE)
- Sparse autoencoders (SAE)
- Variational autoencoders (VAE)
- Denoising autoencoders (DAE) 
- Deep belief networks (DBN)
- Convolutional neural networks (CNN or deep convolutional neural networks, DCNN)
- Deconvolutional networks (DN)
- Deep convolutional inverse graphics networks (DCIGN)
- Generative adversarial networks (GAN)Recurrent neural networks (RNN)Long / short term memory (LSTM)
    + CycleGAN
    + DiscoGAN
    + StarGAN
- Gated recurrent units (GRU)
- Neural Turing machines (NTM)
- Bidirectional recurrent neural networks, bidirectional long / short term memory networks and bidirectional gated recurrent units (BiRNN/BRNN, BiLSTM and BiGRU respectively)
- Deep residual networks (DRN)
- Echo state networks (ESN)
- Extreme learning machines (ELM)
- Liquid state machines (LSM)
- Support vector machines (SVM)
- Kohonen networks (KN, also self organising (feature) map, SOM, SOFM)

**Example Applications**<sup>1</sup>

- Feed forward neural network and Multilayer perceptron
    + Regression and classifications
- Restricted Boltzmann machine
    + Dimensionality reduction
    + Feature extraction/learning
    + Classification
    + Recommender systems
    + Topic modeling
    + Pretraining for weight initialization
- Autoencoders
    + Dimensionality reduction
    + Anomaly detection
    + Generative modeling
- Convolutional neural network
    + Image recognition
    + Video recognition
    + Automatic speech recognition (ASR)
    + Recommender systems
    + Natural language processing
- Recurrent neural network
    + Language modeling
    + Machine translation
    + Handwriting recognition
    + Speech recognition
    + Multilingual Language Processing
    + Natural language processing
- Self-organizing map
    + Dimensionality reduction
- Generative models
- Combinations
    + Image captioning (LSTM + CNN)

<h2><a name="recognition">Recognition and Computer Vision</a></h2>

- Image
- Speech
- Video
- Text and optical character
- Pattern
- Audio
- Facial    
- Handwriting

**Example Applications**

- Recognition
    + [Shazam](https://www.shazam.com/)
    + Wine
        * Companies
            - [Delectable](https://delectable.com/)
            - [Vivino](https://www.vivino.com/)
    + Facebook photo recognition (highlights faces and suggests friends to tag)
    + Speech/Voice to text (faster to talk than to type acc'g to Stanford)
        * Companies
            - [Google Cloud Speech API](https://cloud.google.com/speech/)
    + Text to speech
        * Companies
            - [Amazon Polly](https://aws.amazon.com/polly/)
    + Video
        * Companies
            - [Clarifai](https://www.clarifai.com/)
            - [Google Cloud Video Intelligence](https://cloud.google.com/video-intelligence/)
    + OCR
        * Mobile app check deposits and uploading receipts
        * Post office address recognition
    + Object recognition
        * Companies
            - [Pinterest](https://medium.com/@Pinterest_Engineering) (then used to recommend other pins)
    + Image
        * Companies
            - [Clarifai](https://www.clarifai.com/)
            - [Captricity](http://captricity.com/)
            - [Google Cloud Vision API](https://cloud.google.com/vision/)
            - [Amazon Rekognition](https://aws.amazon.com/rekognition/)
- Computer vision
    + Manufacturing
        * Inspections
        * Quality control
        * Assembly line
    + Visual surveillance
        * Companies
            - [BRS Labs AISight](https://www.genetec.com/solutions/resources/brs-labs-aisight-technology-and-omnicast-integration)
    + Navigation, including autonomous vehicles
        * Land, water, and space
    + Medical image processing and diagnosis
    + Military
        * Detection of enemy solidiers and vehicles
        * Missle guidance
    + Drones
        * Inspection (pipelines), surveillance, exploration (buildings), and protection
        * Companies
            - [Digital Signal](http://www.digitalsignalcorp.com/)
            - [Shield.ai](http://shield.ai/)
    + Item recognition
        * Companies
            - [Amazon Go](https://www.amazon.com/b?node=16008589011)

<h2><a name="anomaly">Anomaly Detection (Supervised, Unsupervised, Semi-supervised)</a></h2>

**Algorithms**
- Density-based techniques - K-nearest neighbor, Local outlier factor
- Subspace and correlation-based outlier detection for high-dimensional data
- One class support vector machines
- Replicator neural networks
- Cluster analysis-based outlier detection
- Deviations from association rules and frequent itemsets
- Fuzzy logic based outlier detection
- Ensemble techniques, using feature bagging, score normalization and different sources of diversity
- PCA (Principle component analysis)

**Example Applications**

- Per Wikipedia
    + Intrusion detection
    + Fraud detection
    + Fault detection
    + System health monitoring
    + Event detection in sensor networks
- Manufacturing
- Data security
    + Companies
        * [Cylance](https://www.cylance.com/en_us/home.html)
        * [Darktrace](https://www.darktrace.com/)
- Personal security (security screenings at airports, stadiums, concerts, and other venues)
- Law enforcement
- Application performance
- Credit card fraud detection

<h2><a name="text-analytics" href="https://en.wikipedia.org/wiki/Natural_language_processing">Text Processing, Analytics, and Mining</a></h2>

- [Text processing](https://en.wikipedia.org/wiki/Text_processing)
- [Lexical Analysis](https://en.wikipedia.org/wiki/Lexical_analysis)
- [Text Mining](https://en.wikipedia.org/wiki/Text_mining)
    + Information retrieval
    + Text categorization
    + Text clustering
    + Concept/entity extraction
    + Production of granular taxonomies
    + Sentiment analysis
    + Document summarization
    + Entity relation modeling
    + Named entity recognition
    + Recognition of Pattern Identified Entities
    + Coreference
    + Syntactic parsing
    + Part-of-speech tagging
    + Quantitative text analysis

<h2><a name="nlp" href="https://en.wikipedia.org/wiki/Natural_language_processing">Natural Language Processing (NLP), Natural Language Generation (NLG), and Natural Language Understanding (NLU)</a></h2>

- Syntax
    + Lemmatization
    + Morphological segmentation
    + Part-of-speech tagging
    + Parsing
    + Sentence breaking (also known as sentence boundary disambiguation)
    + Stemming
    + Word segmentation
    + Terminology extraction
- Semantics
    + Lexical semantics
    + Machine translation
    + Named entity recognition (NER)
    + Natural language generation
    + Natural language understanding
    + Optical character recognition (OCR)
    + Question answering
    + Recognizing Textual entailment
    + Relationship extraction
    + Sentiment analysis
    + Topic segmentation and recognition
    + Word sense disambiguation
- Discourse
    + Automatic summarization
    + Coreference resolution
    + Discourse analysis
- Speech
    + Speech recognition
    + Speech segmentation
    + Text-to-speech

**Example Applications**

- Smart personal assistants
    + Companies
        * [Alexa](https://developer.amazon.com/alexa)
        * [Google Assistant](https://assistant.google.com/)
        * [Siri](https://www.apple.com/ios/siri/)
    + Uses
        * Internet searches and answer questions
        * Set reminders
        * Integrate with your calendar
            - Make appointments
        * Receive sports, news, and finance updates
        * Create to-do lists
        * Order items online
        * Use services (e.g., order an Uber)
        * Play music
        * Play games
        * Smart home integration
- NLG - computer generated reports and news
    + Summarizing documents
    + Story telling
    + Sports recaps
    + Companies
        * [Narrative Science](https://www.narrativescience.com/)
- NLP and language translation
    + Voicemail transcripts
    + eDiscovery
    + Companies
        * [Google Natural Language API](https://cloud.google.com/natural-language/)
        * [Google Cloud Translation API](https://cloud.google.com/translate/)
        * [Textio](https://textio.com/) for writing optimal job descriptions
- NLU and Chatbots
    + Shopping
    + Errands
    + Day to day tasks
    + Companies
        * [x.ai](https://x.ai/) (personal assistant)
        * [MindMeld](https://www.mindmeld.com/)
        * [Google Inbox Smart Reply](https://blog.google/products/gmail/save-time-with-smart-reply-in-gmail/)
        * [Amazon Lex](https://aws.amazon.com/lex/), includes Automatic speech recognition (ASR)
- Smart instant messaging
    + Companies
        * [Google Allo](https://allo.google.com/) smart messaging app (https://allo.google.com/)

<h2><a name="reinforcement">Reinforcement Learning</a></h2>

- Q-learning
- Markov decision process (MDP)
- Finite MDPs
- Monte Carlo methods
- Criterion of optimality
- Brute force
- Value function estimation
- Direct policy search
- Temporal difference methods
- Generalized policy iteration
- Stochastic optimization
- Gradient ascent
- Simulation-based optimization
- Learning Automata[edit]
- Example
    + Multi-armed bandit problem

**Example Applications**

<h2><a name="model-selection">Model selection, validation, and resampling methods</a></h2>

- Cross-validation
- Hyperparameter optimization
- Bootstrap
- Mallow’s Cp
- Akaike information criterion (AIC)
- Bayesian information criterion (BIC)
- Minimum description length (MDL)

<h2><a name="model-tuning">Model tuning: bias variance tradeoff and model complexity</a></h2>

- Validation curve
- Learning curve
- Residual sum of squares
- Goodness-of-fit metrics
- Grid search

<h2><a name="features">Feature extraction, feature selection, and feature engineering</a></h2>

- Wrapper methods
- Sensitivity analysis
- PCA
- Random forests
    + Mean decrease impurity
    + Mean decrease accuracy
- Text-based
    + Stemming
    + Tokenizing
    + Synonym substitutions
- Least absolute shrinkage and selection operator (LASSO)
- Subset selection

<h2><a name="dimensionality-reduction">Dimensionality Reduction</a></h2>

- Principle component analysis (PCA)
- Kernel PCA
- Locally-Linear Embedding (LLE)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Factor analysis
- K-means clustering
- Canopy clustering
- Feature hashing
- Wrapper methods
- Sensitivity analysis
- Self organizing maps
- Text data
    + Term frequency (TF)
    + Inverse document frequency (IDF)
- Latent Dirichlet Allocation (LDA)

<h2><a name="reality">Virtual and Augmented Reality</a></h2>

- Coming soon...

<h2><a name="information-retrieval">Information Retrieval</a></h2>

- Discounted cumulative gain (DCG)
- Discounted cumulative gain (nDCG)
- Term frequency–inverse document frequency (TF-IDF)

<h2><a name="logical-reasoning">Logical Reasoning</a></h2>

- Expert systems
- Logical reasoning

<h2><a name="optimization">Optimization and Search</a></h2>

- Stochastic search
- Stochastic optimization (SO) methods
- Genetic algorithms
- Simulated annealing
- Gradient search
- Linear programming
- Integrer programming
- Non-linear programming
- Active learning
- Ensemble learning
- Minimum
- Maximum
- Optimal value or optimal combination
- Metaheuristic methods
- Randomized search methods
- Tree search
- Monte Carlo tree search (MCTS)
- Evolutionary computation

<h2><a name="risk">Mathematical/quantitative Finance and Risk Management</a></h2>

- [Risk management](https://en.wikipedia.org/wiki/Risk_management)
- [Mathematical/quantitative Finance](https://en.wikipedia.org/wiki/Mathematical_finance)
- Linear Regression
- Monte Carlo methods
- Empirical risk minimization

**Example Applications**

<h2><a name="ranking">Ranking</a></h2>

- [Ranking](https://en.wikipedia.org/wiki/Ranking)

**Example Applications**

<h2><a name="time-series">Time-series</a></h2>

- [Time series](https://en.wikipedia.org/wiki/Time_series)
- Rolling means
- Autocorrelation
- Frequency vs time domains and transfers (e.g., spectral analysis)
- Trend and residual component decomposition
- ARIMA modeling for forecasting and detecting trends

**Example Applications**

<h2><a name="survival">Survival</a></h2>

- [Survival analysis](https://en.wikipedia.org/wiki/Survival_analysis)

**Example Applications**

<h2><a name="forecasting">Forecasting (Wikipedia)</a></h2>

- Last period demand
- Simple and weighted N-Period moving averages
- Simple exponential smoothing
- Poisson process model based forecasting and multiplicative seasonal indexes
- Average approach
- Naïve approach
- Drift method
- Seasonal naïve approach
- Time series methods
    + Moving average
    + Weighted moving average
    + Kalman filtering
    + Exponential smoothing
    + Autoregressive moving average (ARMA)
    + Autoregressive integrated moving average (ARIMA)
    + Extrapolation
    + Linear prediction
    + Trend estimation
    + Growth curve (statistics)
- Causal / econometric forecasting methods
    + Regression analysis
        * Parametric (linear or non-linear)
        * Non-parametric techniques
    + Autoregressive moving average with exogenous inputs (ARMAX)
- Judgmental methods
    + Composite forecasts
    + Cooke's method
    + Delphi method
    + Forecast by analogy
    + Scenario building
    + Statistical surveys
    + Technology forecasting
- Artificial intelligence methods
    + Artificial neural networks
    + Group method of data handling
    + Support vector machines
- Other
    + Simulation
    + Prediction market
    + Probabilistic forecasting and Ensemble forecasting
- Considerations
    + Seasonality and cyclic behaviour

**Example Applications**

<h2><a name="simulation">Simulation</a></h2>

- Discrete event simulation
- Markov models
- Agent-based simulations
- Monte carlo simulations
- Systems dynamics
- Activity-based simulation
- ODES and PDES
- Fuzzy logic

**Example Applications**

<h2><a name="segmentation">Segmentation</a></h2>

- Behavioral
- Demographic
- Geographic

**Example Applications**

<h2><a name="experimentation">Experimentation and Experimental Design</a></h2>

- Design of Experiments (DOE)
- A/B testing

**Example Applications**

<h2><a name="embedded">Embedded</a></h2>

- Deep learning

**Example Applications**

- Robotic cognition

<h2><a name="hypothesis">Hypothesis Testing</a></h2>

- T-test - Compare two groups
- ANOVA - Compare multiple groups

<h2><a name="hybrid">Hybrid Solutions and Applications</a></h2>

**Example Applications**
- Google search
- Autonymous vehicles (Business insider)
    + Reduce accidents and related injuries and death
    + Improved traffic (via ridesharing and smart traffic lights) and fuel efficiency
    + Reduced carbon emissions
    + Faster commutes and travel time
    + Get your time back in the vehicle to do what you want
    + Efficient ride-sharing
    + Companies
        * [Zoox](http://zoox.com/)
        * [Nauto](http://www.nauto.com/)
        * [nuTonomy](http://nutonomy.com/)
- Home monitoring, control, and security
    + Companies
        * [Flare](https://buddyguard.io/)
- Voice-controled robotics
- Photo-realistic pictures generation from text or sketches
    + [NYU article](http://cds.nyu.edu/astronomers-explore-uses-ai-generated-images-using-ai-generating-imagess/)
- Music generation
    + Companies
        * [Jukedeck](https://www.jukedeck.com/)
- Movie and script generation
- Automatically generated software code
    + Companies
        * [DeepCoder](https://openreview.net/pdf?id=ByldLrqlx) (Microsoft and Cambridge)
- Authentication without passwords (using mobile phone that knows it's you)
    + Companies
        * [TypingDNA](https://typingdna.com/)
- Customer support
    + Companies
        * [DigitalGenius](https://www.digitalgenius.com/)
- Optimized directions and routes
- Plagiarism Checkers
- Robo-readers and graders
- Virtual reality
- Gaming
- [Zillow’s](https://www.zillow.com/zestimate/) “zestimate” feature, which estimates the price of homes
- Medical/Health
    + Companies
        * [BenevolentAI](http://benevolent.ai/)
- Sales
    + Companies
        * [InsideSales.com](https://www.insidesales.com/)
- Crime
    + Who, Type, and location
    + Based on previous crime and social media activity
    + Companies
        * [BRS Labs AISight](https://www.genetec.com/solutions/resources/brs-labs-aisight-technology-and-omnicast-integration)
- Suicide risk
    + Based on a lot of different risk factors
    + Companies
        * [Facebook](https://research.fb.com/category/facebook-ai-research-fair/)
        * [Instagram](https://www.instagram.com/)
        * [Cogito](https://www.cogitocorp.com/)
- Agriculture - predicting crop yields
    + Companies
        * [Descartes Lab](http://www.descarteslabs.com/)
        * [Stanford's Sustainability and Artificial Intelligence Lab](http://sustain.stanford.edu/)
- Uber's ETA

<h2><a name="other">Other Algorithms</a></h2>

- Massive-scale graph
- Geospatial temporal predictive analytics
- Hyperfast analytics
- Embedded deep learning
- Cognitive machine learning and IoT
- Natural language processing, generation, and understanding
- Structured database generation
- Game theory
- Control theory
- Operations research
- Information theory
- Simulation-based optimization
- Multi-agent systems
- Swarm intelligence
- Genetic algorithms

<h2><a name="polls">Polls and Popularity</a></h2>

- [Top Algorithms and Methods Used by Data Scientists](http://www.kdnuggets.com/2016/09/poll-algorithms-used-data-scientists.html)

## References

1. [Wikipedia](https://en.wikipedia.org/wiki/Main_Page)
2. [DL4J Deep Learning Use Cases](https://deeplearning4j.org/use_cases)
3. [Wikipedia Outline of Machine Learning](https://en.wikipedia.org/wiki/Outline_of_machine_learning)
4. [Wikipedia Machine Learning Portal](https://en.wikipedia.org/wiki/Portal:Machine_learning)
5. [Wikipedia Outline of Artificial Intelligence](https://en.wikipedia.org/wiki/Outline_of_artificial_intelligence)
6. [Wikipedia Artificial Intelligence Portal](https://en.wikipedia.org/wiki/Portal:Artificial_intelligence)
7. [Wikipedia List of Emerging Technologies](https://en.wikipedia.org/wiki/List_of_emerging_technologies)
8. [Wikipedia Outline of Statistics](https://en.wikipedia.org/wiki/Outline_of_statistics)
9. [Wikipedia Statistics Portal](https://en.wikipedia.org/wiki/Portal:Statistics)
10. [Wikipedia Outline of Mathemetics](https://en.wikipedia.org/wiki/Outline_of_mathematics)
11. [Wikipedia Mathemetics Portal](https://en.wikipedia.org/wiki/Portal:Mathematics)

<!-- ## References
- [The Field Guide to Data Science (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf) -->

