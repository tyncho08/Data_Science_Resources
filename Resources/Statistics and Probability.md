## Statistics
- Descriptive statistics
    + Summary statistics
        * Location or central tendency
            - Arithmetic mean
            - Median
            - Mode
            - Interquartile mean
        * Spread or statistical dispersion (measures of variability)
            - Standard deviation
            - Variance
            - Minimum and maximum
            - Range
            - Interquartile range
            - Order and rank statistics
            - Absolute deviation
            - Distance standard deviation
            - Coefficient of variation
            - Gini coefficient
            - Percentiles
        * Shape
            - L-moments and L-statistic
            - Skewness or kurtosis
            - Distance skewness
        * Dependence
            - Pearson product-moment correlation coefficient
            - Spearman's rank correlation coefficient
    + Univariate analysis - distribution of single variable
        * Central tendancy
        * Dispersion
        * Visualization
    + Bivariate and multivariate analysis - describe the relationship between pairs of variables
        * Cross-tabulations and contingency tables
        * Graphical representation via scatterplots
        * Quantitative measures of dependence
            - Correlation
                + Pearson's r
                + Spearman's rho
                + Covariance
        * Descriptions of conditional distributions
        * Visualization
- Exploratory Data Analysis (EDA)
    + Visualization/graphical
        * Box plot
        * Histogram
        * Multi-vari chart
        * Run chart
        * Pareto chart
        * Scatter plot
        * Stem-and-leaf plot
        * Parallel coordinates
        * Odds ratio
        * Multidimensional scaling
        * Targeted projection pursuit
        * Principal component analysis
        * Multilinear PCA
    + Quantitative
        * Median polish
        * Trimean
        * Ordination
- Analysis of Variance (ANOVA)
- Structured data analysis
    + Algebraic data analysis
    + Bayesian analysis
    + Cluster analysis
    + Combinatorial data analysis
    + Formal concept analysis
    + Functional data analysis
    + Geometric data analysis
    + Regression analysis
    + Shape analysis
    + Topological data analysis
    + Tree structured data analysis
- Statistical inference
- Inductive Statistics
- Distributions
- Visualization and plots
- Experimental Design
- Concepts
    + Statistical power

## Probability
Coming soon...

## [Data Analysis](https://en.wikipedia.org/wiki/Data_analysis)
- Quantitative messages
    + Time-series (line chart)
    + Ranking (bar chart)
    + Part-to-whole (bar chart)
    + Deviation (bar chart)
    + Frequency distribution (histogram)
    + Correlation (scatter plot)
    + Nominal comparison (bar chart)
    + Geographic or geospatial (cartogram)
- Analytical activities
    + Retrieve Value
    + Filter
    + Compute Derived Value
    + Find Extremum
    + Sort
    + Determine Range
    + Characterize Distribution
    + Find Anomalies
    + Cluster
    + Correlate
- Initial transformations - After assessing the quality of the data and of the measurements, one might decide to impute missing data, or to perform initial transformations of one or more variables
    + Square root transformation (if the distribution differs moderately from normal)
    + Log-transformation (if the distribution differs substantially from normal)
    + Inverse transformation (if the distribution differs severely from normal)
    + Make categorical (ordinal / dichotomous) (if the distribution differs severely from normal, and no transformations help)
- Characteristics of data sample
    + Basic statistics of important variables
    + Scatter plots
    + Correlations and associations
    + Cross-tabulations
- Important decisions
    + In the case of non-normals: should one transform variables; make variables categorical (ordinal/dichotomous); adapt the analysis method?
    + In the case of missing data: should one neglect or impute the missing data; which imputation technique should be used?
    + In the case of outliers: should one use robust analysis techniques?
    + In case items do not fit the scale: should one adapt the measurement instrument by omitting items, or rather ensure comparability with other (uses of the) measurement instrument(s)?
    + In the case of (too) small subgroups: should one drop the hypothesis about inter-group differences, or use small sample techniques, like exact tests or bootstrapping?
    + In case the randomization procedure seems to be defective: can and should one calculate propensity scores and include them as covariates in the main analyses?
- Analysis
    + Types
        * Univariate statistics (single variable)
        * Bivariate associations (correlations)
        * Graphical techniques (scatter plots)
    + Techniques
        * Nominal and ordinal variables
            - Frequency counts (numbers and percentages)
            - Associations
                + circumambulations (crosstabulations)
                + hierarchical loglinear analysis (restricted to a maximum of 8 variables)
                + loglinear analysis (to identify relevant/important variables and possible confounders)
            - Exact tests or bootstrapping (in case subgroups are small)
            - Computation of new variables
        * Continuous variables
            - Distribution
                + Statistics (M, SD, variance, skewness, kurtosis)
                + Stem-and-leaf displays
                + Box plots

## [68–95–99.7 rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)
- Percentage of values that lie within a band around the mean in a normal distribution with a width of two, four and six standard deviations, respectively
- 68.27%, 95.45% and 99.73% of the values lie within one, two and three standard deviations of the mean, respectively
- Three-sigma rule of thumb - "nearly all" values are taken to lie within three standard deviations of the mean, i.e. that it is empirically useful to treat 99.7% probability as "near certainty"
- Three-sigma rule - even for non-normally distributed variables, at least 98% of cases should fall within properly-calculated three-sigma intervals
