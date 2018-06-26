# FeatureImportance
This project aims to explore some commonly used methods for feature importance measusrement, in both classical machine learning and neural network fields. 

The term feature 'importance', or 'attribution', or 'sensitivity', could be quite vague statistically. It is either not mathmetically well-defined, or narrowed to a very specific approach hence not comparable in a larger scope. Generally speaking, there are three categories of importance measurement:
* [Univariate/Marginal Importance](#univariatemarginal-importance);
* [Predictive Importance](#predictive-importance);
* [Causal Importance](#causal-importance); 

A roadmap for this project is shown below. The circle represents the state of data processing, from raw data to a selected range of models, all the way to trained model instances. And the feature importance measurements associated with these states are in the linked squares. Of all the measurements, univariate importance methods are in red, predictive importance methods are in blue, and casual importance methods are in green. 
<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/feature_importance.png" width="80%">
</p>

The introduction, implmenation, and the discussion for each measurement are listed below. All of them are tested on the [toy dataset](#a-toy-dataset) and visualized in bar plots. Still, with these rankings and plots, making comparison over different measurements can be hard: given two equally informative features A and B, one method ranks A before B and another methods rank otherwise. We cannot hereby claim which method is better. In practice, we do not have the ground truth about which feature is more 'informative' in the first place, which makes the comparison between different measurements harder, if not completely impossible.

However, the bottom line for a good importance measurement is clear: **informative features should have higher importance than noises**, which is the primary property that we are looking for when evaluating these different measurements.

## A Toy Dataset
Before getting started, we need a toy dataset simple enough for the evaluation of importance measurements. Here I used [make_classification](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) to generate a scaled and shifted dataset, and then normalize it. Both normalized and un-normalized data are used for most importance measurements because I would like to observe if normalization (or scaling/shifting in general) can influence the feature importance. 

Here is the code for dataset generation:
```python
original_x, y = make_classification(n_samples=5000,
                                    n_features=20,
                                    n_informative=8,
                                    n_redundant=2,
                                    n_repeated=2,
                                    n_classes=2,
                                    flip_y=0.1,
                                    shift=None,
                                    scale=None,
                                    random_state=0,
                                    shuffle=False)
scaler = StandardScaler()
normalized_x = scaler.fit_transform(original_x)
```

The plot of all the mean and variances of both original and noralized data are shown below:
<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/data_summary.png" width="100%">
</p>

## Univariate/Marginal Importance
Univariate importance measures a statisicial relation of a single feature and the target. This realtion could be anythong from R-score to mutual information, as long as it is well-defined between two vectors. 

To implment this, I used [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) with [ANOVA F-value](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif) and [mutual information](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif). The importance measurements using both metrics on normalzied and un-normalzied data are shown below:
<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/univariate_score.png" width="100%">
</p>

The advantages are obvious: 
* mathmetically well-defined (which ensures that repeated features have the same importance scores);
* could be invariant to shifting/scaling with the right scoring metric;
* feaures with higher importance scores are usually useful in real learning for they are related to the target; 

However, note that because this measurement only considers one feature at a time, the importance scores could be **very misleading and inpractical**. Consider a situation like this: there are two features x1 and x2 for target y, where x1 is random noise N(0, 1) and x2 is exactly  (y-x1). The learning result should be easy: y = x1 + x2, which means that both features contribute equally. But in univariate importance measurement, x1 will have really low importance scores because noises are not related to the target in any way. So in conclusion, univariate/marginal importance cannot recoginize complicated relations in the input space, hence fails giving credit to more subtle and informative features. 

## Predictive Importance 
Predictive importance measures the increased generalization error when one or more features are altered (omitted, zeroed, blurred with noise, etc.). This measurement is quite easy to understand: if changing a feature makes the prediction worse, then it must be important.

However, in practice, we have different ways to implement this measurement. Most notably, after altering a feature, we need to evaluate the performance of the new set of feature by predicting our target. Can we still use the trained model from our original run? Or do we need to train a new model on new features?

In my opinion, the re-training is necessary. Imagine a scenario like this: there are two features x1 and x2 for target y, and any one of these two features are capable of predicting the target individually. So we can easily find a model that completely relies on x1 and ignores x2


## Causal Importance
