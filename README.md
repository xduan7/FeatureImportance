# Feature Importance
This project aims to explore some commonly used methods for feature importance measurements, in both classical machine learning and neural network fields. 

The term feature 'importance', or 'attribution', or 'relevance', could be quite vague statistically. It is either not mathematically  well-defined, or narrowed to a very specific approach hence not comparable in a larger scope. Generally speaking, there are three categories of importance measurement:
* [Univariate/Marginal Importance](#univariatemarginal-importance);
* [Predictive Importance](#predictive-importance);
* [Causal Importance](#causal-importance); 

A roadmap for this project is shown below. The circle represents the state of data processing, from raw data to a selected range of models, all the way to trained model instances. And the feature importance measurements associated with these states are in the linked squares. Of all the measurements, the univariate importance methods are in red, predictive importance methods are in blue, and casual importance methods are in green. 
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

**In this dataset, feature #0-7 are informative, feature #8 and #9 are linear combination of the informative features, feature #10 and #11 is identical to feature #0 and #4 separately, and the rest are simply noises.** 

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

In my opinion, the **re-training is necessary**. Imagine a scenario like this: there are two features x1 and x2 for target y, and any one of these two features are capable of predicting the target individually. So we can easily find a model that completely relies on x1 and ignores x2, which we name model1. A plot of such scenario (red cross and blue dots represent different classes) is shown below. Suppose that now we would like to measure the importance of x1 and x2 using model1 without re-training. When x1 is omitted, model1 cannot make any prediction; when x2 is omitted, model1 can make perfect prediction. Then the predictive importance of x1 is 1 but x2 gets 0. We know that this result is unfair because both features are equally informative, and can be avoided using re-training. 
<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/2_clf.png" width="50%">
</p>

In the implmentation, I tried three different strategies shown as the following:
* [omit one feature at a time with re-training](https://github.com/xduan7/FeatureImportance/blob/master/img/feature_elim_score.png)
* [zero out one feature at a time without re-training](https://github.com/xduan7/FeatureImportance/blob/master/img/feature_rplc(trained)_score.png)
* [zero out one feature at a time with re-training](https://github.com/xduan7/FeatureImportance/blob/master/img/feature_rplc(untrained)_score.png)

Here shows the accuracy reults of the last strategy (zero out one feature at a time with re-training):
<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/feature_rplc(untrained)_score.png" width="100%">
</p>

Despite some cases where the classifer fails (e.g. RBF SVC on un-normalized data), all three strategies ranks features in a relatively reasonable ordering, where most informative features are ranked before noises. Also we can see that re-training does not seem to play a big role in this experiment, which implies that all features are used properly during training. 

However, there are two things worth attention:
* most classifiers failed to rank all informative features before all noises;
* the accruacy dropped only a little when one feature was altered;
In fact, these two drawbacks are reflecting the same thing: **informative features have extremely low predictive importance scores if they are correlated/dependent to each other**. When we have features that are correlated to each other (in our case, we have repeated/ redundant features), altering one feature does not reduce the information over the input space, which means that the model performance will not be effectly significantly. This disadvantage lies in the root of predictive importance. And in practice, we have no idea if our data contains correlated features. 

One seemly possible way to avoid this is to identify independent components using [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) or [ICA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html). We can use PCA/ICA to get a certain number of components, and then use predictive importance measurement to assign scores to each components, lastly use inverse transformation of PCA/ICA to get hte importance of each feature. However, note that predictive importance meausres the change of error for altering each features, and the change of error (in our case, the classification accuracy), whcih mathmetical non-sense to perform inverse transformation upon. Check out these gibberish results if you are not convinced ([PCA(n=4)](https://github.com/xduan7/FeatureImportance/blob/master/img/feature(4)_rplc(untrained)_score.png), [PCA(n=8)](https://github.com/xduan7/FeatureImportance/blob/master/img/feature(8)_rplc(untrained)_score.png), [PCA(n=12)](https://github.com/xduan7/FeatureImportance/blob/master/img/feature(12)_rplc(untrained)_score.png), [PCA(n=16)](https://github.com/xduan7/FeatureImportance/blob/master/img/feature(16)_rplc(untrained)_score.png)). Things would make more sense if we are using the change of neural network logits instead of accuracy to measure this importance. 


## Causal Importance
Causal importance is vaguely defined as 'how much does the output changes when an input feature changes'. It looks somehow like the feature altering strategy in predictive importance but differs in practice. In reality, there are different approaches other than actually perturb each features: we can either use the weights associated with such feature, or we can use the gradients backpropagated from the output, which are easily accessible in nerual networks. Another difference between causal importance and predictive importance is that, the latter need to measure change of error over the whole validation dataset, while causal importance can usually be performed upon a sinlge datapoint. 

### Weight-based Importance
The most common weight-based importance measurement is probably by using tree/forest models, where each node is assocaited with a bunch of features and we can easily calculate how 'important' a feature is by computing the '(normalized) total reduction of the criterion brought by that feature'. 

Feature importance are already implmented in most tree-based models in scikit-learn. After training, the importance scores can be accessed at a model attribute named feature_importances_. However, despite how easy and accessible this measurement is, a good model is crucial for better importance measurement. 

To perform this experiment, I used [decision tree](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [random forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) on both normalized and un-normalized data. For each model, I trained them with the following three different settings:
* default hyper-parameters, random seed = 0;
* best hyper-paramesters, random seed = 0;
* best hyper-paramesters, random seed = 1;
Note that best hyper-parameters are collected using grid search, which does not necessary mean optimal in any way, but much better than default models to see the difference. The results of this experiment are shown below. 
<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/tree_score.png" width="100%">
</p>
From the results we have the following observations/conclusions:
* in almost all cases, tree-based models rank informative/redundant/repeated features before all noises;
* randomness plays a role in feature importance while using tree-based models;
* data scaling/shifting does not effect the performance of tree-based models as well as their importance measurements;
* a better model with higher accuracy is more likely to rank all the features correcly;

These are all desirable properties, and these are definitely the best feature importance measurement so far in this project. However, some of the informative features seem to have similar importance as noises, which is reasonable considering that we have a redundant feature space. To eliminiate the redundancy, I tried [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and [ICA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html) to reduce the feature space, and then inverse transform the feature importance back. Since the importance score of each feature is actually the normalized weight in the classifier, it makes sense to apply linear transformation backward and still preserve the convept of importance. 

The results for PCA/ICA with tree-based importance are shown below. There are two interesting properties:
* identical features have the same importance scores (duh);
* PCA with 4 components seems to work fantastic, resulting a noticable gap between the importance scores of informative/redundant/repeated features and the noises;

<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/indep_tree_score.png" width="100%">
</p>


Both properties are rather desirable, especially the second one. The first one is actually trivial because of PCA/ICA; it is the second propety that drives me insane, for I have no evidence about why shrinking the number of component to 4 (not 8 or hgiher) will make such big difference. 

My explaination is that, by reducing the demensionality greatly, we already filtered out some noise (with a little bit of useful information) in the original feature space during PCA. The filtered-out components will not have any importance score because they are not the input of our classifier. That's why with higher number of components, the importance score of noises are getting higher, and that desirable 'gap' disappear. When number of component is 12, we can actually see that the ranking is not entirely correct, and that is when PCA/ICA filters out enough useful information to mess up the measurement. 

So if my speculation holds true, then PCA is not actually that useful. PCA (or ICA in the similar way) can only contribute to the feature importance if they filter out a huge chunk of noise and only a little useful information. As far as I know, this condition does not always hold. Also in practice, we have no idea how many components we shall keep in order to make this miracle happen again. 


### Gradient-based Importance
Gradient-based importacne measurements become more and more relavent for two reasons:
* gradients are naturally more interpretable than weights 
* nerual networks become important and gradients are easily accessible in nn;

The second point is obvious. But the first one is subtle. In the article '[How to measure important features](ftp://ftp.sas.com/pub/neural/importance.html)', the author wrote 'a partial derivative has the same interpretation as a weight in a linear model, except for one crucial difference: a weight in a linear model applies to the entire input space, but a partial derivative applies only to a small neighborhood of the input point at which it is computed'. Weights are global while gradients are local; to explain why a model makes such prediction given an input, gradients are simply far better as they can tell us what feature is 'sensitive' at this very situation. 

However, when we are looking at the entire feature space, not just a single sample, things get trickier. What we have is all the gradients of the features w.r.t. the output over all data points, how to use all the gradinets is the problem. 
* if we sum up all the gradients of all data points directly, postive and negative gradients might canel each other out;
* if we sum up all the absolute or squared gradients of all data points, the local gradients might fail on relecting the overall feature space;

Let's imagine a scenario similar to the example in the article '[How to measure important features](ftp://ftp.sas.com/pub/neural/importance.html)': y = tanh(x1 + 1) - tanh(x1 - 1) + tanh(x2 + 2) - tanh(x2 - 2). Two features x1 and x2 and sampled independently and uniformly in the range \[-3, 3\]. If we simply sum up all the gradients, both features will get 0 for the importance scores; and if we use the sum of the absolute/sqaured value of the gradients, x1 and x2 will have the same importance scores, despite the fact that feature x2 contribute much more to the target y. 

There are multiple ways to counter this, but most of them ultilize a specific or unspecific reference: instead of summing up all the gradient directly, we multiply the gradient with the feature value with a refernce offset, and then sum them up. This multiplication can be viewed as injecting glabal information into local partial derivatives.

In the table 1 of the paper [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104), multiple gradient-based importance are compared against [Occlusion-1](https://arxiv.org/pdf/1311.2901.pdf), which is a pertubation-based method, which approximates gradients. Using the same software [DeepExplain](https://github.com/marcoancona/DeepExplain), and sum up all the (absolute value of) feature importance over the validation set, all the features are ranked as shown below. Note that all the importance scores are calculate using the same trained model. 

<p align="center">
  <img src="https://github.com/xduan7/FeatureImportance/blob/master/img/deep_explain.png" width="100%">
</p>

All methods ranked features correctly (informative/repeated/redundant features > noises) and there is a noticable gap between the noises and the useful features. The ordering differs somehow: with the same model, some might find feature #4 more important than #5 while some find otherwise. This difference is due to the different strategies of gradient backpropagation and integration shown in the table 1 of  [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104).

At this point, PCA can be applied to make things look better [PCA(n=4)](https://github.com/xduan7/FeatureImportance/blob/master/img/pca(4)_deep_explain.png). Since all these methods are gradeint-based, the linear transformation like PCA or ICA makes sense. However, just as I speculated in the last part of [Weight-based Importance](#weight\-based-importance), PCA only filters out the noises to make things look better, it does not necessary increase the performance of feature importance measurement. If more components are included, the rankings are actually incorrect like this ([PCA(n=16)](https://github.com/xduan7/FeatureImportance/blob/master/img/pca(16)_deep_explain.png)).


## Final Thoughts
* consensus in feature importance seems impossible;
* is there an actual correct ranking of feature importance?
