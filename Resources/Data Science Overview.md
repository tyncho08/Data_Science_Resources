## Applications
- Descriptive, predictive, and prescriptive analytics
- Machine learning
- Data/text mining and data exploration
- Real-time stream processing and computing (spot opportunities and risks)
- Statistical analysis
- Multivariate statistics
- Automated decision making, predictions, recommendations, and insights
- Deployed batch and/or real-time solution (e.g., analytics models and algorithms), potentially including monitoring, performance measurement, analytics, dashboards, and reporting

## Analytics Types
- Descriptive analytics - What happened and why?
    + Visualization
    + EDA
- Predictive analytics - What is the probability of something happening?
    + Machine learning
    + Artificial intelligence
    + Forecasting
- Prescriptive analytics - What specific recommendations will drive business decisions and help achieve business goals (i.e., what to do if 'X' happens)
- Entity analytics
- Streaming media analytics
- Text analytics
    + Semantic analysis
- Graph analytics (e.g., social network analysis)

## Analytics Toolbox
- Mathematical
- Computational
- Visual
- Analytic
- Statistical
- Experimental
- Problem definition
- Model building
- Validation

## Data Science Process (non-linear, iterative, and cyclical)
- Domain discovery, goal identification, and question development
    + Understanding the goal of the project and define objectives
    + Drives mapping the problem space to the solution space
- Determine the type of problem and type of solution required
    + Regression, classification, unsupervised, recommender, reinforcement, text analytics, ...
- Data instrumentation, acquisition, collection, extraction, merging/joining, ETL, storage, and pipeline architecture/development
    + Data sources: raw data, real time measurement, events, IoT, and so on
    + Data warehouse or data lake
        * Data lake eliminates need for ETL according to Booz, Allen, and Hamilton
    + Note that more data is almost always better
- Data munging/wrangling
    - Data parsing, cleaning, and tidying
    - Data processing, transformation, and aggregation
        + Includes [feature scaling](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html), normalization, and/or standardization
            * Primary
                - Standard score aka z-score (+/-): (x-Mean)/StdDev
                - Min-Max aka unity-based normalization ([0, 1]): (x-Min)/Range
            * Other
                - General feature scaling (adjust to range): (Min+Range) * FeatureScaledValue
                    + FeatureScaledValue between 0 and 1
                - Relative to mean: x/Mean
                - Relative to max: x/Max
        + Categorical feature transformation and dummy variables
            * One-hot encoding or category indexing
                - Category indexing
                    + Assign a numeric value to each category
                    + Implies ordering of categories, e.g., 0, 1, 2, ...
                    + Best for ordinal variables
                - One-hot encoding
                    + Converts categories into binary vectors with at most one nonzero value, e.g., [0, 0], [0, 1], [1, 0]
                    + Best for nominal (non-ordinal) variables
            * Note: For large number of categorical features, a good technique is to encode only the most important features (e.g., 95% of the importance) and then assign the rest to an 'others' class
        + Deduplication
        + Data conversions
            * Character dates to dates
            * Numerics to strings and vice versa
        + Format conversion
        + Frequency space
            * Fast fourier transform (FFT)
            * Discrete wavelet transform
        + Euclidian space
            * Coordinate transform
        + Transposition if needed
        + Sorting as needed
    - Data filtering
        + Outlier detection and removal
        + Exponential smoothing
        + Gaussian filter
        + Median filter
        + Noise handling and reduction
    - Find and handle missing values
        + Imputation
            * Generate values for other observations in dataset
                - Random sampling
                - Markov chain monte carlo
            * Without other observations
                - Mean
                - Statistical distributions
                - Regression models
        + Data deletion
- Exploratory data analysis (EDA), statistical analysis, descriptive analytics, and visualization
    + General plots | characteristics, trends, outliers, ...
        * Histogram
        * Scatter plot and scatter matrix (aka pair plots)
        * Box plot
        * Line graph
    + Generate frequency table
    + Generate contingency table (i.e., cross tabulation or crosstab)
    + Target variable distribution
    + Distribution and outlier detection via box and scatter plots for numeric features
    + Correlation analysis and pairwise distribution plots
    + Plots with point color as label for classification tasks
    + Aggregations and analysis
        * Group for counts, sums, and averages
    + Statistical analysis
        * Sensitivity analysis
        * Correlation and causation
        * Variance and standard deviation
        * Mean, median, and mode
        * Max and min
        * Quartiles and inter-quartile
        * Skew and kurtosis
        * Quartiles
        * Distribution
        * Count
        * Range
        * Plots
        * Validation via p-values and confidence intervals
        * ANOVA
    + Summary statistics
- Feature selection and feature engineering
    + Purpose
        * Simpler and faster training
        * Less computationally expensive
        * Find feature interactions to use as new features
        * Improve model performance
        * Optimize bias/variance tradeoff and minimize overfitting
- Performance metric selection, including:
    + Mean squared error (MSE)
    + Root mean squared error (RMSE)
    + Mean absolute error (MAE)
    + R squared (aka explained variance)
    + [Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
        * Outcomes
            - True positive
            - True negative
            - False positive (Type 1 error, false alarm)
            - False negative - (Type II error, miss)
        * Metrics
            - Recall, sensitivity, hit rate, or true positive rate (TPR)
                + TP/P = TP/(TP+FN)
                + Fraction of relevant instances that are retrieved
                + How complete the results are
            - Specificity or true negative rate (TNR)
                + TN/N = TN/(FP+TN)
            - Precision or positive predictive value (PPV)
                + TP/(TP+FP)
                + Fraction of retrieved instances that are relevant
                + How useful the results are
            - Negative predictive value (NPV)
                + TN/(TN+FN)
            - Fall-out or false positive rate (FPR)
                + FP/N = FP/(FP+TN) = 1 - TNR
            - False discovery rate (FDR)
                + FP/(FP+TP) = 1 - PPV
            - Miss rate or false negative rate (FNR)
                + FN/P = FN/(FN+TP) = 1 - TPR
            - Accuracy
                + (TP+TN)/(P+N)
            - F1 score, or F-score
                + (2TP)/(2TP+FP+FN)
            - Matthews correlation coefficient (MCC)
            - Informedness or Bookmaker Informedness (BM)
            - Markedness (MK)
    + Receiver operator characteristic (ROC)
        * Area under the ROC curve (AUC)
            - Represents the likelihood of giving the positive example a score higher than a negative example
    + Loss functions
    + Per Kaggle
        * LB: Score you get is calculated on a subset of testing set
        * CV: Score you get by local cross validation is commonly referred to as a CV score.
        * Mean Average Precision (MAP)
- Data splitting
    + Training, validation, test
    + Cross-validation as alternative to traditional splitting
- Model selection, training, evaluation, validation, resampling methods, tuning, and complexity reduction
    + Iterative process and involves revisiting previous stages, including model selection
    + Ensemble methods exploration and implementation as needed to achieve performance goals
        * Benefits
            - Bias and variance reduction, thus reduced overfitting risk
            - Increased performance
    + Error analysis and tradeoffs
        * Type 1
        * Type 2
        * ...
    + Kaggle notes
        * Tree-based (top for Kaggle)
            - Gradient Boosted Trees
            - Random Forest
            - Extra Randomized Trees
        * Lower performance
            - SVM
            - Linear Regression
            - Logistic Regression
            - Neural Networks
    + Model validation, resampling methods, and selection
        * E.g., cross-validation
    + Model complexity reduction via subset selection, shrinkage methods, regularization (e.g., ridge regression and lasso), and dimensionality reduction
    + Bias variance tradeoff
    + Hyperparameter optimization, grid search, ...
    + Model tuning goals
        * Performance
        * Accuracy
        * Robustness
        * Speed
- Deliverables, deployment, and results communication

## Tips
- Random forest usually reach optimum when max_features is set to the square root of the total number of features.
- [How to Rank 10% in Your First Kaggle Competition](https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/)
    + Ensemble methods
        * Base models should be as unrelated as possibly. This is why we tend to include non-tree-based models in the ensemble even though they don’t perform as well. The math says that the greater the diversity, and less bias in the final ensemble.
        * Performance of base models shouldn’t differ to much.
    + Cross-validation
        * Usually 5-fold CV is good enough
        * We shouldn’t use too many folds if our training data is limited. Otherwise we would have too few samples in each fold to guarantee statistical significance.
        * More folds, the CV score would become more reliable, but the training takes longer to finish as well.
    + Automated pipeline
        * Modularized feature transformations. We only need to write a few lines of codes (or better, rules / DSLs) and the new feature is added to the training set.
        * Automated grid search. We only need to set up models and parameter grid, the search will be run and the best parameters will be recorded.
        * Automated ensemble selection. Use K best models for training the ensemble as soon as we put another base model into the pool.
- [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
    + Start with directly observed and reported features as opposed to learned features.
        * A learned feature is a feature generated either by an external system (such as an unsupervised clustering system) or by the learner itself (e.g. via a factored model or deep learning).
        * The primary issue with factored models and deep models is that they are non­convex. Thus, there is no guarantee that an optimal solution can be approximated or found, and the local minima found on each iteration can be different. This variation makes it hard to judge whether the impact of a change to your system is meaningful or random.
    + Use very specific features when you can.
        * You can use regularization to eliminate the features that apply to too few examples.
    + Combine and modify existing features to create new features in human­understandable ways.
        * Transformations, with the two most standard approaches are “discretizations” and “crosses”.
            - Discretization consists of taking a continuous feature and creating many discrete features from it. Don’t overthink the boundaries of these histograms: basic quantiles will give you most of the impact.
            - Crosses combine two or more feature columns.
                + Note that it takes massive amounts of data to learn models with crosses of three, four, or more base feature columns.
            - Crosses that produce very large feature columns may overfit.
    + The number of feature weights you can learn in a linear model is roughly proportional to the amount of data you have.
        * There are fascinating statistical learning theory results concerning the appropriate level of complexity for a model, but this rule is basically all you need to know.
    + Clean up features you are no longer using. Unused features create technical debt.
    + When choosing models, utilitarian performance trumps predictive power.
    + Look for patterns in the measured errors, and create new features.
    + Keep ensembles simple
        * To keep things simple, each model should either be an ensemble only taking the input of other models, or a base model taking many features, but not both
- [Practical advice for analysis of large, complex data sets](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)
    + Technical
        * Look at your distributions
            - Visualizations
                + Histograms
                + QDfs
                + Q-Q plots
            - Look for
                + Multi-modal behavior or a significant class of outliers that you need to decide how to summarize
        * Consider the outliers
            - It’s fine to exclude them from your data or to lump them together into an “Unusual” category, but you should make sure you know why data ended up in that category
        * Report noise/confidence
            - We must be aware that randomness exists and will fool us
            - Every estimator that you produce should have a notion of your confidence in this estimate attached to it
                + Formal
                    * Confidence intervals or credible intervals for estimators
                    * p-vales or Bayes factors for conclusions
        * Look at examples
            - You should be doing stratified sampling to look at a good sample across the distribution of values so you are not too focussed on the most common cases
        * Slice your data
            - Separate your data into subgroups and look at the values of your metrics in those subgroups separately
            - Be aware of mix shifts. A mix shift is when the amount of data in a slice is different across the groups you are comparing. Simpson’s paradox and other confusions can result. Generally, if the relative amount of data in a slice is the same across your two groups, you can safely make a comparison.
        * Consider practical significance
        * Check for consistency over time
            - One particular slicing you should almost always employ is to slice by units of time (we often use days, but other units may be useful also). This is because many disturbances to underlying data happen as our systems evolve over time.
            - Just because a particular day or set of days is an outlier does not mean you should discard it. Use the data as a hook to find a causal reason for that day being different before you discard it.
            - The other benefit of looking at day over day data is it gives you a sense of the variation in the data that would eventually lead to confidence intervals or claims of statistical significance.
    + Process
        * Separate Validation, Description, and Evaluation
        * Confirm experimentation/data collection setup
        * Check vital signs
        * Standard first, custom second
        * Measure twice, or more
        * Check for reproducibility
        * Check for consistency with past measurements
        * Make hypotheses and look for evidence
        * Exploratory analysis benefits from end to end iteration
    + Social
        * Data analysis starts with questions, not data or a technique
        * Acknowledge and count your filtering
        * Ratios should have clear numerator and denominators
        * Educate your consumers
        * Be both skeptic and champion
        * Share with peers first, external consumers second
        * Expect and accept ignorance and mistakes

## Process Models
- [Ask, Get, Explore, Model, Communicate and visualize results](http://www.datascientists.net/what-is-data-science)
- [Sample, Explore, Modify, Model, and Assess - SEMMA](https://en.wikipedia.org/wiki/SEMMA)
- [Acquire, Prepare, Analyze, Act (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf)
    + Analyze stage (iterate evaluation)
        * Setup
        * Try
        * Do
- [The Data Science Maturity Model: Collect, Describe, Discover, Predict, Advise (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf)
- [Balancing the Five Analytic Dimensions (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf)
    + Speed
    + Analytic complexity
    + Accuracy and precision
    + Data size
    + Data complexity
- [Implementation Constraints (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf)
    + Computational frequency
    + Solution timeliness
    + Implementation speed
    + Computational resource limitations
    + Data storage limitations
- [Fractal Analytic Model (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf)
    + Goal
        * Describe
        * Discover
        * Predict
        * Advise
    + Data
    + Computation
        * Aggregation
        * Enrichment
        * Clustering
        * Classification
    + Action
        * Productization
        * Data monetization
        * Insights and relationships
- [Cross Industry Standard Process for Data Mining - CRISP-DM](https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining)
    + [CRISP Visual Guide](https://exde.files.wordpress.com/2009/03/crisp_visualguide.pdf)
- [Knowledge Discovery in Databases - KDD](https://en.wikipedia.org/wiki/Data_mining#Process)
    + [Overview of the KDD Process](http://www2.cs.uregina.ca/~dbd/cs831/notes/kdd/1_kdd.html)
    + [From Data Mining to Knowledge Discovery in Databases](http://www.kdnuggets.com/gpspubs/aimag-kdd-overview-1996-Fayyad.pdf)
- [Team Data Science Process (TDSP)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-process-overview) and [Interactive graphic](https://azure.microsoft.com/en-us/documentation/learning-paths/data-science-process/)
- [Acquire, Prepare, Analyze, Report, Act](https://www.coursera.org/learn/big-data-introduction/lecture/Fonq2/steps-in-the-data-science-process)
- [The REASON Method: Relating, Explaining, Selecting, Outlining, and Navigating](http://www.datasciencecentral.com/m/blogpost?id=6448529%3ABlogPost%3A369943)

<!-- ## References
- [The Field Guide to Data Science (Booz, Allen, Hamilton)](https://www.boozallen.com/content/dam/boozallen/documents/2015/12/2015-FIeld-Guide-To-Data-Science.pdf) -->
