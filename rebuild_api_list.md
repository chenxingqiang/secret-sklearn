# [Xgboost scikit-learn likely API reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sflearn)

# sflearn API Reference

This is the class and function reference of scikit-learn. Please refer to the [full user guide](https://scikit-learn.org/stable/user_guide.html#user-guide) for further details, as the class and function raw specifications may not be enough to give full guidelines on their uses. For reference on concepts repeated across the API, see [Glossary of Common Terms and API Elements](https://scikit-learn.org/stable/glossary.html#glossary).

## [`sflearn.base`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.base): Base classes and utility functions

Base classes for all estimators.

### Base classes

| [`base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sflearn.base.BaseEstimator.html#sflearn.base.BaseEstimator) | Base class for all estimators in scikit-learn.               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`base.BiclusterMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.BiclusterMixin.html#sflearn.base.BiclusterMixin) | Mixin class for all bicluster estimators in scikit-learn.    |
| [`base.ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.ClassifierMixin.html#sflearn.base.ClassifierMixin) | Mixin class for all classifiers in scikit-learn.             |
| [`base.ClusterMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.ClusterMixin.html#sflearn.base.ClusterMixin) | Mixin class for all cluster estimators in scikit-learn.      |
| [`base.DensityMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.DensityMixin.html#sflearn.base.DensityMixin) | Mixin class for all density estimators in scikit-learn.      |
| [`base.RegressorMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.RegressorMixin.html#sflearn.base.RegressorMixin) | Mixin class for all regression estimators in scikit-learn.   |
| [`base.TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.TransformerMixin.html#sflearn.base.TransformerMixin) | Mixin class for all transformers in scikit-learn.            |
| [`base.OneToOneFeatureMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.OneToOneFeatureMixin.html#sflearn.base.OneToOneFeatureMixin) | Provides `get_feature_names_out` for simple transformers.    |
| [`base.ClassNamePrefixFeaturesOutMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.base.ClassNamePrefixFeaturesOutMixin.html#sflearn.base.ClassNamePrefixFeaturesOutMixin) | Mixin class for transformers that generate their own names by prefixing. |
| [`feature_selection.SelectorMixin`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectorMixin.html#sflearn.feature_selection.SelectorMixin) | Transformer mixin that performs feature selection given a support mask |

### Functions

| [`base.clone`](https://scikit-learn.org/stable/modules/generated/sflearn.base.clone.html#sflearn.base.clone)(estimator, *[, safe]) | Construct a new unfitted estimator with the same parameters. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`base.is_classifier`](https://scikit-learn.org/stable/modules/generated/sflearn.base.is_classifier.html#sflearn.base.is_classifier)(estimator) | Return True if the given estimator is (probably) a classifier. |
| [`base.is_regressor`](https://scikit-learn.org/stable/modules/generated/sflearn.base.is_regressor.html#sflearn.base.is_regressor)(estimator) | Return True if the given estimator is (probably) a regressor. |
| [`config_context`](https://scikit-learn.org/stable/modules/generated/sflearn.config_context.html#sflearn.config_context)(*[, assume_finite, ...]) | Context manager for global scikit-learn configuration.       |
| [`get_config`](https://scikit-learn.org/stable/modules/generated/sflearn.get_config.html#sflearn.get_config)() | Retrieve current values for configuration set by [`set_config`](https://scikit-learn.org/stable/modules/generated/sflearn.set_config.html#sflearn.set_config). |
| [`set_config`](https://scikit-learn.org/stable/modules/generated/sflearn.set_config.html#sflearn.set_config)([assume_finite, working_memory, ...]) | Set global scikit-learn configuration                        |
| [`show_versions`](https://scikit-learn.org/stable/modules/generated/sflearn.show_versions.html#sflearn.show_versions)() | Print useful debugging information"                          |

## [`sflearn.calibration`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.calibration): Probability Calibration

Calibration of predicted probabilities.

**User guide:** See the [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html#calibration) section for further details.

| [`calibration.CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sflearn.calibration.CalibratedClassifierCV.html#sflearn.calibration.CalibratedClassifierCV)([...]) | Probability calibration with isotonic regression or logistic regression. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

| [`calibration.calibration_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.calibration.calibration_curve.html#sflearn.calibration.calibration_curve)(y_true, y_prob, *) | Compute true and predicted probabilities for a calibration curve. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

## [`sflearn.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.cluster): Clustering

The [`sflearn.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.cluster) module gathers popular unsupervised clustering algorithms.

**User guide:** See the [Clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering) and [Biclustering](https://scikit-learn.org/stable/modules/biclustering.html#biclustering) sections for further details.

### Classes

| [`cluster.AffinityPropagation`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.AffinityPropagation.html#sflearn.cluster.AffinityPropagation)(*[, damping, ...]) | Perform Affinity Propagation Clustering of data.             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.AgglomerativeClustering.html#sflearn.cluster.AgglomerativeClustering)([...]) | Agglomerative Clustering.                                    |
| [`cluster.Birch`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.Birch.html#sflearn.cluster.Birch)(*[, threshold, ...]) | Implements the BIRCH clustering algorithm.                   |
| [`cluster.DBSCAN`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.DBSCAN.html#sflearn.cluster.DBSCAN)([eps, min_samples, metric, ...]) | Perform DBSCAN clustering from vector array or distance matrix. |
| [`cluster.FeatureAgglomeration`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.FeatureAgglomeration.html#sflearn.cluster.FeatureAgglomeration)([n_clusters, ...]) | Agglomerate features.                                        |
| [`cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.KMeans.html#sflearn.cluster.KMeans)([n_clusters, init, n_init, ...]) | K-Means clustering.                                          |
| [`cluster.BisectingKMeans`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.BisectingKMeans.html#sflearn.cluster.BisectingKMeans)([n_clusters, init, ...]) | Bisecting K-Means clustering.                                |
| [`cluster.MiniBatchKMeans`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.MiniBatchKMeans.html#sflearn.cluster.MiniBatchKMeans)([n_clusters, init, ...]) | Mini-Batch K-Means clustering.                               |
| [`cluster.MeanShift`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.MeanShift.html#sflearn.cluster.MeanShift)(*[, bandwidth, seeds, ...]) | Mean shift clustering using a flat kernel.                   |
| [`cluster.OPTICS`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.OPTICS.html#sflearn.cluster.OPTICS)(*[, min_samples, max_eps, ...]) | Estimate clustering structure from vector array.             |
| [`cluster.SpectralClustering`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.SpectralClustering.html#sflearn.cluster.SpectralClustering)([n_clusters, ...]) | Apply clustering to a projection of the normalized Laplacian. |
| [`cluster.SpectralBiclustering`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.SpectralBiclustering.html#sflearn.cluster.SpectralBiclustering)([n_clusters, ...]) | Spectral biclustering (Kluger, 2003).                        |
| [`cluster.SpectralCoclustering`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.SpectralCoclustering.html#sflearn.cluster.SpectralCoclustering)([n_clusters, ...]) | Spectral Co-Clustering algorithm (Dhillon, 2001).            |

### Functions

| [`cluster.affinity_propagation`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.affinity_propagation.html#sflearn.cluster.affinity_propagation)(S, *[, ...]) | Perform Affinity Propagation Clustering of data.             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`cluster.cluster_optics_dbscan`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.cluster_optics_dbscan.html#sflearn.cluster.cluster_optics_dbscan)(*, ...) | Perform DBSCAN extraction for an arbitrary epsilon.          |
| [`cluster.cluster_optics_xi`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.cluster_optics_xi.html#sflearn.cluster.cluster_optics_xi)(*, reachability, ...) | Automatically extract clusters according to the Xi-steep method. |
| [`cluster.compute_optics_graph`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.compute_optics_graph.html#sflearn.cluster.compute_optics_graph)(X, *, ...) | Compute the OPTICS reachability graph.                       |
| [`cluster.dbscan`](https://scikit-learn.org/stable/modules/generated/dbscan-function.html#sflearn.cluster.dbscan)(X[, eps, min_samples, ...]) | Perform DBSCAN clustering from vector array or distance matrix. |
| [`cluster.estimate_bandwidth`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.estimate_bandwidth.html#sflearn.cluster.estimate_bandwidth)(X, *[, quantile, ...]) | Estimate the bandwidth to use with the mean-shift algorithm. |
| [`cluster.k_means`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.k_means.html#sflearn.cluster.k_means)(X, n_clusters, *[, ...]) | Perform K-means clustering algorithm.                        |
| [`cluster.kmeans_plusplus`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.kmeans_plusplus.html#sflearn.cluster.kmeans_plusplus)(X, n_clusters, *[, ...]) | Init n_clusters seeds according to k-means++.                |
| [`cluster.mean_shift`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.mean_shift.html#sflearn.cluster.mean_shift)(X, *[, bandwidth, seeds, ...]) | Perform mean shift clustering of data using a flat kernel.   |
| [`cluster.spectral_clustering`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.spectral_clustering.html#sflearn.cluster.spectral_clustering)(affinity, *[, ...]) | Apply clustering to a projection of the normalized Laplacian. |
| [`cluster.ward_tree`](https://scikit-learn.org/stable/modules/generated/sflearn.cluster.ward_tree.html#sflearn.cluster.ward_tree)(X, *[, connectivity, ...]) | Ward clustering based on a Feature matrix.                   |

## [`sflearn.compose`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.compose): Composite Estimators

Meta-estimators for building composite models with transformers

In addition to its current contents, this module will eventually be home to refurbished versions of Pipeline and FeatureUnion.

**User guide:** See the [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) section for further details.

| [`compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.compose.ColumnTransformer.html#sflearn.compose.ColumnTransformer)(transformers, *[, ...]) | Applies transformers to columns of an array or pandas DataFrame. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`compose.TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.compose.TransformedTargetRegressor.html#sflearn.compose.TransformedTargetRegressor)([...]) | Meta-estimator to regress on a transformed target.           |

| [`compose.make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sflearn.compose.make_column_transformer.html#sflearn.compose.make_column_transformer)(*transformers) | Construct a ColumnTransformer from the given transformers.   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sflearn.compose.make_column_selector.html#sflearn.compose.make_column_selector)([pattern, ...]) | Create a callable to select columns to be used with `ColumnTransformer`. |

## [`sflearn.covariance`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.covariance): Covariance Estimators

The [`sflearn.covariance`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.covariance) module includes methods and algorithms to robustly estimate the covariance of features given a set of points. The precision matrix defined as the inverse of the covariance is also estimated. Covariance estimation is closely related to the theory of Gaussian Graphical Models.

**User guide:** See the [Covariance estimation](https://scikit-learn.org/stable/modules/covariance.html#covariance) section for further details.

| [`covariance.EmpiricalCovariance`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.EmpiricalCovariance.html#sflearn.covariance.EmpiricalCovariance)(*[, ...]) | Maximum likelihood covariance estimator.                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`covariance.EllipticEnvelope`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.EllipticEnvelope.html#sflearn.covariance.EllipticEnvelope)(*[, ...]) | An object for detecting outliers in a Gaussian distributed dataset. |
| [`covariance.GraphicalLasso`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.GraphicalLasso.html#sflearn.covariance.GraphicalLasso)([alpha, mode, ...]) | Sparse inverse covariance estimation with an l1-penalized estimator. |
| [`covariance.GraphicalLassoCV`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.GraphicalLassoCV.html#sflearn.covariance.GraphicalLassoCV)(*[, alphas, ...]) | Sparse inverse covariance w/ cross-validated choice of the l1 penalty. |
| [`covariance.LedoitWolf`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.LedoitWolf.html#sflearn.covariance.LedoitWolf)(*[, store_precision, ...]) | LedoitWolf Estimator.                                        |
| [`covariance.MinCovDet`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.MinCovDet.html#sflearn.covariance.MinCovDet)(*[, store_precision, ...]) | Minimum Covariance Determinant (MCD): robust estimator of covariance. |
| [`covariance.OAS`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.OAS.html#sflearn.covariance.OAS)(*[, store_precision, ...]) | Oracle Approximating Shrinkage Estimator.                    |
| [`covariance.ShrunkCovariance`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.ShrunkCovariance.html#sflearn.covariance.ShrunkCovariance)(*[, ...]) | Covariance estimator with shrinkage.                         |

| [`covariance.empirical_covariance`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.empirical_covariance.html#sflearn.covariance.empirical_covariance)(X, *[, ...]) | Compute the Maximum likelihood covariance estimator.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`covariance.graphical_lasso`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.graphical_lasso.html#sflearn.covariance.graphical_lasso)(emp_cov, alpha, *) | L1-penalized covariance estimator.                           |
| [`covariance.ledoit_wolf`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.ledoit_wolf.html#sflearn.covariance.ledoit_wolf)(X, *[, ...]) | Estimate the shrunk Ledoit-Wolf covariance matrix.           |
| [`covariance.oas`](https://scikit-learn.org/stable/modules/generated/oas-function.html#sflearn.covariance.oas)(X, *[, assume_centered]) | Estimate covariance with the Oracle Approximating Shrinkage algorithm. |
| [`covariance.shrunk_covariance`](https://scikit-learn.org/stable/modules/generated/sflearn.covariance.shrunk_covariance.html#sflearn.covariance.shrunk_covariance)(emp_cov[, ...]) | Calculate a covariance matrix shrunk on the diagonal.        |

## [`sflearn.cross_decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.cross_decomposition): Cross decomposition

**User guide:** See the [Cross decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition) section for further details.

| [`cross_decomposition.CCA`](https://scikit-learn.org/stable/modules/generated/sflearn.cross_decomposition.CCA.html#sflearn.cross_decomposition.CCA)([n_components, ...]) | Canonical Correlation Analysis, also known as "Mode B" PLS. |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| [`cross_decomposition.PLSCanonical`](https://scikit-learn.org/stable/modules/generated/sflearn.cross_decomposition.PLSCanonical.html#sflearn.cross_decomposition.PLSCanonical)([...]) | Partial Least Squares transformer and regressor.            |
| [`cross_decomposition.PLSRegression`](https://scikit-learn.org/stable/modules/generated/sflearn.cross_decomposition.PLSRegression.html#sflearn.cross_decomposition.PLSRegression)([...]) | PLS regression.                                             |
| [`cross_decomposition.PLSSVD`](https://scikit-learn.org/stable/modules/generated/sflearn.cross_decomposition.PLSSVD.html#sflearn.cross_decomposition.PLSSVD)([n_components, ...]) | Partial Least Square SVD.                                   |

## [`sflearn.datasets`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.datasets): Datasets

The [`sflearn.datasets`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.datasets) module includes utilities to load datasets, including methods to load and fetch popular reference datasets. It also features some artificial data generators.

**User guide:** See the [Dataset loading utilities](https://scikit-learn.org/stable/datasets.html#datasets) section for further details.

### Loaders

| [`datasets.clear_data_home`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.clear_data_home.html#sflearn.datasets.clear_data_home)([data_home]) | Delete all the content of the data home cache.               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`datasets.dump_svmlight_file`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.dump_svmlight_file.html#sflearn.datasets.dump_svmlight_file)(X, y, f, *[, ...]) | Dump the dataset in svmlight / libsvm file format.           |
| [`datasets.fetch_20newsgroups`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_20newsgroups.html#sflearn.datasets.fetch_20newsgroups)(*[, data_home, ...]) | Load the filenames and data from the 20 newsgroups dataset (classification). |
| [`datasets.fetch_20newsgroups_vectorized`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_20newsgroups_vectorized.html#sflearn.datasets.fetch_20newsgroups_vectorized)(*[, ...]) | Load and vectorize the 20 newsgroups dataset (classification). |
| [`datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_california_housing.html#sflearn.datasets.fetch_california_housing)(*[, ...]) | Load the California housing dataset (regression).            |
| [`datasets.fetch_covtype`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_covtype.html#sflearn.datasets.fetch_covtype)(*[, data_home, ...]) | Load the covertype dataset (classification).                 |
| [`datasets.fetch_kddcup99`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_kddcup99.html#sflearn.datasets.fetch_kddcup99)(*[, subset, ...]) | Load the kddcup99 dataset (classification).                  |
| [`datasets.fetch_lfw_pairs`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_lfw_pairs.html#sflearn.datasets.fetch_lfw_pairs)(*[, subset, ...]) | Load the Labeled Faces in the Wild (LFW) pairs dataset (classification). |
| [`datasets.fetch_lfw_people`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_lfw_people.html#sflearn.datasets.fetch_lfw_people)(*[, data_home, ...]) | Load the Labeled Faces in the Wild (LFW) people dataset (classification). |
| [`datasets.fetch_olivetti_faces`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_olivetti_faces.html#sflearn.datasets.fetch_olivetti_faces)(*[, ...]) | Load the Olivetti faces data-set from AT&T (classification). |
| [`datasets.fetch_openml`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_openml.html#sflearn.datasets.fetch_openml)([name, version, ...]) | Fetch dataset from openml by name or dataset id.             |
| [`datasets.fetch_rcv1`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_rcv1.html#sflearn.datasets.fetch_rcv1)(*[, data_home, subset, ...]) | Load the RCV1 multilabel dataset (classification).           |
| [`datasets.fetch_species_distributions`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.fetch_species_distributions.html#sflearn.datasets.fetch_species_distributions)(*[, ...]) | Loader for species distribution dataset from Phillips et.    |
| [`datasets.get_data_home`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.get_data_home.html#sflearn.datasets.get_data_home)([data_home]) | Return the path of the scikit-learn data directory.          |
| [`datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_breast_cancer.html#sflearn.datasets.load_breast_cancer)(*[, return_X_y, ...]) | Load and return the breast cancer wisconsin dataset (classification). |
| [`datasets.load_diabetes`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_diabetes.html#sflearn.datasets.load_diabetes)(*[, return_X_y, ...]) | Load and return the diabetes dataset (regression).           |
| [`datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_digits.html#sflearn.datasets.load_digits)(*[, n_class, ...]) | Load and return the digits dataset (classification).         |
| [`datasets.load_files`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_files.html#sflearn.datasets.load_files)(container_path, *[, ...]) | Load text files with categories as subfolder names.          |
| [`datasets.load_iris`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_iris.html#sflearn.datasets.load_iris)(*[, return_X_y, as_frame]) | Load and return the iris dataset (classification).           |
| [`datasets.load_linnerud`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_linnerud.html#sflearn.datasets.load_linnerud)(*[, return_X_y, as_frame]) | Load and return the physical exercise Linnerud dataset.      |
| [`datasets.load_sample_image`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_sample_image.html#sflearn.datasets.load_sample_image)(image_name) | Load the numpy array of a single sample image.               |
| [`datasets.load_sample_images`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_sample_images.html#sflearn.datasets.load_sample_images)() | Load sample images for image manipulation.                   |
| [`datasets.load_svmlight_file`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_svmlight_file.html#sflearn.datasets.load_svmlight_file)(f, *[, ...]) | Load datasets in the svmlight / libsvm format into sparse CSR matrix. |
| [`datasets.load_svmlight_files`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_svmlight_files.html#sflearn.datasets.load_svmlight_files)(files, *[, ...]) | Load dataset from multiple files in SVMlight format.         |
| [`datasets.load_wine`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.load_wine.html#sflearn.datasets.load_wine)(*[, return_X_y, as_frame]) | Load and return the wine dataset (classification).           |

### Samples generator

| [`datasets.make_biclusters`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_biclusters.html#sflearn.datasets.make_biclusters)(shape, n_clusters, *) | Generate a constant block diagonal structure array for biclustering. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`datasets.make_blobs`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_blobs.html#sflearn.datasets.make_blobs)([n_samples, n_features, ...]) | Generate isotropic Gaussian blobs for clustering.            |
| [`datasets.make_checkerboard`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_checkerboard.html#sflearn.datasets.make_checkerboard)(shape, n_clusters, *) | Generate an array with block checkerboard structure for biclustering. |
| [`datasets.make_circles`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_circles.html#sflearn.datasets.make_circles)([n_samples, shuffle, ...]) | Make a large circle containing a smaller circle in 2d.       |
| [`datasets.make_classification`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_classification.html#sflearn.datasets.make_classification)([n_samples, ...]) | Generate a random n-class classification problem.            |
| [`datasets.make_friedman1`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_friedman1.html#sflearn.datasets.make_friedman1)([n_samples, ...]) | Generate the "Friedman #1" regression problem.               |
| [`datasets.make_friedman2`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_friedman2.html#sflearn.datasets.make_friedman2)([n_samples, noise, ...]) | Generate the "Friedman #2" regression problem.               |
| [`datasets.make_friedman3`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_friedman3.html#sflearn.datasets.make_friedman3)([n_samples, noise, ...]) | Generate the "Friedman #3" regression problem.               |
| [`datasets.make_gaussian_quantiles`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_gaussian_quantiles.html#sflearn.datasets.make_gaussian_quantiles)(*[, mean, ...]) | Generate isotropic Gaussian and label samples by quantile.   |
| [`datasets.make_hastie_10_2`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_hastie_10_2.html#sflearn.datasets.make_hastie_10_2)([n_samples, ...]) | Generate data for binary classification used in Hastie et al. 2009, Example 10.2. |
| [`datasets.make_low_rank_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_low_rank_matrix.html#sflearn.datasets.make_low_rank_matrix)([n_samples, ...]) | Generate a mostly low rank matrix with bell-shaped singular values. |
| [`datasets.make_moons`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_moons.html#sflearn.datasets.make_moons)([n_samples, shuffle, ...]) | Make two interleaving half circles.                          |
| [`datasets.make_multilabel_classification`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_multilabel_classification.html#sflearn.datasets.make_multilabel_classification)([...]) | Generate a random multilabel classification problem.         |
| [`datasets.make_regression`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_regression.html#sflearn.datasets.make_regression)([n_samples, ...]) | Generate a random regression problem.                        |
| [`datasets.make_s_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_s_curve.html#sflearn.datasets.make_s_curve)([n_samples, noise, ...]) | Generate an S curve dataset.                                 |
| [`datasets.make_sparse_coded_signal`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_sparse_coded_signal.html#sflearn.datasets.make_sparse_coded_signal)(n_samples, ...) | Generate a signal as a sparse combination of dictionary elements. |
| [`datasets.make_sparse_spd_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_sparse_spd_matrix.html#sflearn.datasets.make_sparse_spd_matrix)([dim, ...]) | Generate a sparse symmetric definite positive matrix.        |
| [`datasets.make_sparse_uncorrelated`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_sparse_uncorrelated.html#sflearn.datasets.make_sparse_uncorrelated)([...]) | Generate a random regression problem with sparse uncorrelated design. |
| [`datasets.make_spd_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_spd_matrix.html#sflearn.datasets.make_spd_matrix)(n_dim, *[, ...]) | Generate a random symmetric, positive-definite matrix.       |
| [`datasets.make_swiss_roll`](https://scikit-learn.org/stable/modules/generated/sflearn.datasets.make_swiss_roll.html#sflearn.datasets.make_swiss_roll)([n_samples, noise, ...]) | Generate a swiss roll dataset.                               |

## [`sflearn.decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.decomposition): Matrix Decomposition

The [`sflearn.decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.decomposition) module includes matrix decomposition algorithms, including among others PCA, NMF or ICA. Most of the algorithms of this module can be regarded as dimensionality reduction techniques.

**User guide:** See the [Decomposing signals in components (matrix factorization problems)](https://scikit-learn.org/stable/modules/decomposition.html#decompositions) section for further details.

| [`decomposition.DictionaryLearning`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.DictionaryLearning.html#sflearn.decomposition.DictionaryLearning)([...]) | Dictionary learning.                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`decomposition.FactorAnalysis`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.FactorAnalysis.html#sflearn.decomposition.FactorAnalysis)([n_components, ...]) | Factor Analysis (FA).                                        |
| [`decomposition.FastICA`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.FastICA.html#sflearn.decomposition.FastICA)([n_components, ...]) | FastICA: a fast algorithm for Independent Component Analysis. |
| [`decomposition.IncrementalPCA`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.IncrementalPCA.html#sflearn.decomposition.IncrementalPCA)([n_components, ...]) | Incremental principal components analysis (IPCA).            |
| [`decomposition.KernelPCA`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.KernelPCA.html#sflearn.decomposition.KernelPCA)([n_components, ...]) | Kernel Principal component analysis (KPCA) [[R396fc7d924b8-1\]](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.KernelPCA.html#r396fc7d924b8-1). |
| [`decomposition.LatentDirichletAllocation`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.LatentDirichletAllocation.html#sflearn.decomposition.LatentDirichletAllocation)([...]) | Latent Dirichlet Allocation with online variational Bayes algorithm. |
| [`decomposition.MiniBatchDictionaryLearning`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.MiniBatchDictionaryLearning.html#sflearn.decomposition.MiniBatchDictionaryLearning)([...]) | Mini-batch dictionary learning.                              |
| [`decomposition.MiniBatchSparsePCA`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.MiniBatchSparsePCA.html#sflearn.decomposition.MiniBatchSparsePCA)([...]) | Mini-batch Sparse Principal Components Analysis.             |
| [`decomposition.NMF`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.NMF.html#sflearn.decomposition.NMF)([n_components, init, ...]) | Non-Negative Matrix Factorization (NMF).                     |
| [`decomposition.MiniBatchNMF`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.MiniBatchNMF.html#sflearn.decomposition.MiniBatchNMF)([n_components, ...]) | Mini-Batch Non-Negative Matrix Factorization (NMF).          |
| [`decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.PCA.html#sflearn.decomposition.PCA)([n_components, copy, ...]) | Principal component analysis (PCA).                          |
| [`decomposition.SparsePCA`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.SparsePCA.html#sflearn.decomposition.SparsePCA)([n_components, ...]) | Sparse Principal Components Analysis (SparsePCA).            |
| [`decomposition.SparseCoder`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.SparseCoder.html#sflearn.decomposition.SparseCoder)(dictionary, *[, ...]) | Sparse coding.                                               |
| [`decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.TruncatedSVD.html#sflearn.decomposition.TruncatedSVD)([n_components, ...]) | Dimensionality reduction using truncated SVD (aka LSA).      |

| [`decomposition.dict_learning`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.dict_learning.html#sflearn.decomposition.dict_learning)(X, n_components, ...) | Solve a dictionary learning matrix factorization problem.    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`decomposition.dict_learning_online`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.dict_learning_online.html#sflearn.decomposition.dict_learning_online)(X[, ...]) | Solve a dictionary learning matrix factorization problem online. |
| [`decomposition.fastica`](https://scikit-learn.org/stable/modules/generated/fastica-function.html#sflearn.decomposition.fastica)(X[, n_components, ...]) | Perform Fast Independent Component Analysis.                 |
| [`decomposition.non_negative_factorization`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.non_negative_factorization.html#sflearn.decomposition.non_negative_factorization)(X) | Compute Non-negative Matrix Factorization (NMF).             |
| [`decomposition.sparse_encode`](https://scikit-learn.org/stable/modules/generated/sflearn.decomposition.sparse_encode.html#sflearn.decomposition.sparse_encode)(X, dictionary, *) | Sparse coding.                                               |

## [`sflearn.discriminant_analysis`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.discriminant_analysis): Discriminant Analysis

Linear Discriminant Analysis and Quadratic Discriminant Analysis

**User guide:** See the [Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda) section for further details.

| [`discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sflearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sflearn.discriminant_analysis.LinearDiscriminantAnalysis)([...]) | Linear Discriminant Analysis.    |
| ------------------------------------------------------------ | -------------------------------- |
| [`discriminant_analysis.QuadraticDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sflearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sflearn.discriminant_analysis.QuadraticDiscriminantAnalysis)(*) | Quadratic Discriminant Analysis. |

## [`sflearn.dummy`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.dummy): Dummy estimators

**User guide:** See the [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) section for further details.

| [`dummy.DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.dummy.DummyClassifier.html#sflearn.dummy.DummyClassifier)(*[, strategy, ...]) | DummyClassifier makes predictions that ignore the input features. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`dummy.DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.dummy.DummyRegressor.html#sflearn.dummy.DummyRegressor)(*[, strategy, ...]) | Regressor that makes predictions using simple rules.         |

## [`sflearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.ensemble): Ensemble Methods

The [`sflearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.ensemble) module includes ensemble-based methods for classification, regression and anomaly detection.

**User guide:** See the [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#ensemble) section for further details.

| [`ensemble.AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.AdaBoostClassifier.html#sflearn.ensemble.AdaBoostClassifier)([estimator, ...]) | An AdaBoost classifier.                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`ensemble.AdaBoostRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.AdaBoostRegressor.html#sflearn.ensemble.AdaBoostRegressor)([estimator, ...]) | An AdaBoost regressor.                                       |
| [`ensemble.BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.BaggingClassifier.html#sflearn.ensemble.BaggingClassifier)([estimator, ...]) | A Bagging classifier.                                        |
| [`ensemble.BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.BaggingRegressor.html#sflearn.ensemble.BaggingRegressor)([estimator, ...]) | A Bagging regressor.                                         |
| [`ensemble.ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.ExtraTreesClassifier.html#sflearn.ensemble.ExtraTreesClassifier)([...]) | An extra-trees classifier.                                   |
| [`ensemble.ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.ExtraTreesRegressor.html#sflearn.ensemble.ExtraTreesRegressor)([n_estimators, ...]) | An extra-trees regressor.                                    |
| [`ensemble.GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.GradientBoostingClassifier.html#sflearn.ensemble.GradientBoostingClassifier)(*[, ...]) | Gradient Boosting for classification.                        |
| [`ensemble.GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.GradientBoostingRegressor.html#sflearn.ensemble.GradientBoostingRegressor)(*[, ...]) | Gradient Boosting for regression.                            |
| [`ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.IsolationForest.html#sflearn.ensemble.IsolationForest)(*[, n_estimators, ...]) | Isolation Forest Algorithm.                                  |
| [`ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.RandomForestClassifier.html#sflearn.ensemble.RandomForestClassifier)([...]) | A random forest classifier.                                  |
| [`ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.RandomForestRegressor.html#sflearn.ensemble.RandomForestRegressor)([...]) | A random forest regressor.                                   |
| [`ensemble.RandomTreesEmbedding`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.RandomTreesEmbedding.html#sflearn.ensemble.RandomTreesEmbedding)([...]) | An ensemble of totally random trees.                         |
| [`ensemble.StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.StackingClassifier.html#sflearn.ensemble.StackingClassifier)(estimators[, ...]) | Stack of estimators with a final classifier.                 |
| [`ensemble.StackingRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.StackingRegressor.html#sflearn.ensemble.StackingRegressor)(estimators[, ...]) | Stack of estimators with a final regressor.                  |
| [`ensemble.VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.VotingClassifier.html#sflearn.ensemble.VotingClassifier)(estimators, *[, ...]) | Soft Voting/Majority Rule classifier for unfitted estimators. |
| [`ensemble.VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.VotingRegressor.html#sflearn.ensemble.VotingRegressor)(estimators, *[, ...]) | Prediction voting regressor for unfitted estimators.         |
| [`ensemble.HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.HistGradientBoostingRegressor.html#sflearn.ensemble.HistGradientBoostingRegressor)([...]) | Histogram-based Gradient Boosting Regression Tree.           |
| [`ensemble.HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.ensemble.HistGradientBoostingClassifier.html#sflearn.ensemble.HistGradientBoostingClassifier)([...]) | Histogram-based Gradient Boosting Classification Tree.       |

## [`sflearn.exceptions`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.exceptions): Exceptions and warnings

The [`sflearn.exceptions`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.exceptions) module includes all custom warnings and error classes used across scikit-learn.

| [`exceptions.ConvergenceWarning`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.ConvergenceWarning.html#sflearn.exceptions.ConvergenceWarning) | Custom warning to capture convergence problems               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`exceptions.DataConversionWarning`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.DataConversionWarning.html#sflearn.exceptions.DataConversionWarning) | Warning used to notify implicit data conversions happening in the code. |
| [`exceptions.DataDimensionalityWarning`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.DataDimensionalityWarning.html#sflearn.exceptions.DataDimensionalityWarning) | Custom warning to notify potential issues with data dimensionality. |
| [`exceptions.EfficiencyWarning`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.EfficiencyWarning.html#sflearn.exceptions.EfficiencyWarning) | Warning used to notify the user of inefficient computation.  |
| [`exceptions.FitFailedWarning`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.FitFailedWarning.html#sflearn.exceptions.FitFailedWarning) | Warning class used if there is an error while fitting the estimator. |
| [`exceptions.NotFittedError`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.NotFittedError.html#sflearn.exceptions.NotFittedError) | Exception class to raise if estimator is used before fitting. |
| [`exceptions.UndefinedMetricWarning`](https://scikit-learn.org/stable/modules/generated/sflearn.exceptions.UndefinedMetricWarning.html#sflearn.exceptions.UndefinedMetricWarning) | Warning used when the metric is invalid                      |

## [`sflearn.experimental`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.experimental): Experimental

The [`sflearn.experimental`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.experimental) module provides importable modules that enable the use of experimental features or estimators.

The features and estimators that are experimental aren’t subject to deprecation cycles. Use them at your own risks!

| [`experimental.enable_hist_gradient_boosting`](https://scikit-learn.org/stable/modules/generated/sflearn.experimental.enable_hist_gradient_boosting.html#module-sflearn.experimental.enable_hist_gradient_boosting) | This is now a no-op and can be safely removed from your code. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`experimental.enable_iterative_imputer`](https://scikit-learn.org/stable/modules/generated/sflearn.experimental.enable_iterative_imputer.html#module-sflearn.experimental.enable_iterative_imputer) | Enables IterativeImputer                                     |
| [`experimental.enable_halving_search_cv`](https://scikit-learn.org/stable/modules/generated/sflearn.experimental.enable_halving_search_cv.html#module-sflearn.experimental.enable_halving_search_cv) | Enables Successive Halving search-estimators                 |

## [`sflearn.feature_extraction`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.feature_extraction): Feature Extraction

The [`sflearn.feature_extraction`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.feature_extraction) module deals with feature extraction from raw data. It currently includes methods to extract features from text and images.

**User guide:** See the [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction) section for further details.

| [`feature_extraction.DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.DictVectorizer.html#sflearn.feature_extraction.DictVectorizer)(*[, ...]) | Transforms lists of feature-value mappings to vectors. |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [`feature_extraction.FeatureHasher`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.FeatureHasher.html#sflearn.feature_extraction.FeatureHasher)([...]) | Implements feature hashing, aka the hashing trick.     |

### From images

The [`sflearn.feature_extraction.image`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.feature_extraction.image) submodule gathers utilities to extract features from images.

| [`feature_extraction.image.extract_patches_2d`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.image.extract_patches_2d.html#sflearn.feature_extraction.image.extract_patches_2d)(...) | Reshape a 2D image into a collection of patches.  |
| ------------------------------------------------------------ | ------------------------------------------------- |
| [`feature_extraction.image.grid_to_graph`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.image.grid_to_graph.html#sflearn.feature_extraction.image.grid_to_graph)(n_x, n_y) | Graph of the pixel-to-pixel connections.          |
| [`feature_extraction.image.img_to_graph`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.image.img_to_graph.html#sflearn.feature_extraction.image.img_to_graph)(img, *) | Graph of the pixel-to-pixel gradient connections. |
| [`feature_extraction.image.reconstruct_from_patches_2d`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.image.reconstruct_from_patches_2d.html#sflearn.feature_extraction.image.reconstruct_from_patches_2d)(...) | Reconstruct the image from all of its patches.    |
| [`feature_extraction.image.PatchExtractor`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.image.PatchExtractor.html#sflearn.feature_extraction.image.PatchExtractor)(*[, ...]) | Extracts patches from a collection of images.     |

### From text

The [`sflearn.feature_extraction.text`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.feature_extraction.text) submodule gathers utilities to build feature vectors from text documents.

| [`feature_extraction.text.CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.text.CountVectorizer.html#sflearn.feature_extraction.text.CountVectorizer)(*[, ...]) | Convert a collection of text documents to a matrix of token counts. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`feature_extraction.text.HashingVectorizer`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.text.HashingVectorizer.html#sflearn.feature_extraction.text.HashingVectorizer)(*) | Convert a collection of text documents to a matrix of token occurrences. |
| [`feature_extraction.text.TfidfTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.text.TfidfTransformer.html#sflearn.feature_extraction.text.TfidfTransformer)(*) | Transform a count matrix to a normalized tf or tf-idf representation. |
| [`feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_extraction.text.TfidfVectorizer.html#sflearn.feature_extraction.text.TfidfVectorizer)(*[, ...]) | Convert a collection of raw documents to a matrix of TF-IDF features. |

## [`sflearn.feature_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.feature_selection): Feature Selection

The [`sflearn.feature_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.feature_selection) module implements feature selection algorithms. It currently includes univariate filter selection methods and the recursive feature elimination algorithm.

**User guide:** See the [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection) section for further details.

| [`feature_selection.GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.GenericUnivariateSelect.html#sflearn.feature_selection.GenericUnivariateSelect)([...]) | Univariate feature selector with configurable strategy.      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`feature_selection.SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectPercentile.html#sflearn.feature_selection.SelectPercentile)([...]) | Select features according to a percentile of the highest scores. |
| [`feature_selection.SelectKBest`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectKBest.html#sflearn.feature_selection.SelectKBest)([score_func, k]) | Select features according to the k highest scores.           |
| [`feature_selection.SelectFpr`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectFpr.html#sflearn.feature_selection.SelectFpr)([score_func, alpha]) | Filter: Select the pvalues below alpha based on a FPR test.  |
| [`feature_selection.SelectFdr`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectFdr.html#sflearn.feature_selection.SelectFdr)([score_func, alpha]) | Filter: Select the p-values for an estimated false discovery rate. |
| [`feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectFromModel.html#sflearn.feature_selection.SelectFromModel)(estimator, *) | Meta-transformer for selecting features based on importance weights. |
| [`feature_selection.SelectFwe`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SelectFwe.html#sflearn.feature_selection.SelectFwe)([score_func, alpha]) | Filter: Select the p-values corresponding to Family-wise error rate. |
| [`feature_selection.SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.SequentialFeatureSelector.html#sflearn.feature_selection.SequentialFeatureSelector)(...) | Transformer that performs Sequential Feature Selection.      |
| [`feature_selection.RFE`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.RFE.html#sflearn.feature_selection.RFE)(estimator, *[, ...]) | Feature ranking with recursive feature elimination.          |
| [`feature_selection.RFECV`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.RFECV.html#sflearn.feature_selection.RFECV)(estimator, *[, ...]) | Recursive feature elimination with cross-validation to select features. |
| [`feature_selection.VarianceThreshold`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.VarianceThreshold.html#sflearn.feature_selection.VarianceThreshold)([threshold]) | Feature selector that removes all low-variance features.     |

| [`feature_selection.chi2`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.chi2.html#sflearn.feature_selection.chi2)(X, y) | Compute chi-squared stats between each non-negative feature and class. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`feature_selection.f_classif`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.f_classif.html#sflearn.feature_selection.f_classif)(X, y) | Compute the ANOVA F-value for the provided sample.           |
| [`feature_selection.f_regression`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.f_regression.html#sflearn.feature_selection.f_regression)(X, y, *[, ...]) | Univariate linear regression tests returning F-statistic and p-values. |
| [`feature_selection.r_regression`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.r_regression.html#sflearn.feature_selection.r_regression)(X, y, *[, ...]) | Compute Pearson's r for each features and the target.        |
| [`feature_selection.mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.mutual_info_classif.html#sflearn.feature_selection.mutual_info_classif)(X, y, *) | Estimate mutual information for a discrete target variable.  |
| [`feature_selection.mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sflearn.feature_selection.mutual_info_regression.html#sflearn.feature_selection.mutual_info_regression)(X, y, *) | Estimate mutual information for a continuous target variable. |

## [`sflearn.gaussian_process`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.gaussian_process): Gaussian Processes

The [`sflearn.gaussian_process`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.gaussian_process) module implements Gaussian Process based regression and classification.

**User guide:** See the [Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process) section for further details.

| [`gaussian_process.GaussianProcessClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.GaussianProcessClassifier.html#sflearn.gaussian_process.GaussianProcessClassifier)([...]) | Gaussian process classification (GPC) based on Laplace approximation. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`gaussian_process.GaussianProcessRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.GaussianProcessRegressor.html#sflearn.gaussian_process.GaussianProcessRegressor)([...]) | Gaussian process regression (GPR).                           |

Kernels:

| [`gaussian_process.kernels.CompoundKernel`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.CompoundKernel.html#sflearn.gaussian_process.kernels.CompoundKernel)(kernels) | Kernel which is composed of a set of other kernels.          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`gaussian_process.kernels.ConstantKernel`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.ConstantKernel.html#sflearn.gaussian_process.kernels.ConstantKernel)([...]) | Constant kernel.                                             |
| [`gaussian_process.kernels.DotProduct`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.DotProduct.html#sflearn.gaussian_process.kernels.DotProduct)([...]) | Dot-Product kernel.                                          |
| [`gaussian_process.kernels.ExpSineSquared`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.ExpSineSquared.html#sflearn.gaussian_process.kernels.ExpSineSquared)([...]) | Exp-Sine-Squared kernel (aka periodic kernel).               |
| [`gaussian_process.kernels.Exponentiation`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.Exponentiation.html#sflearn.gaussian_process.kernels.Exponentiation)(...) | The Exponentiation kernel takes one base kernel and a scalar parameter � and combines them via |
| [`gaussian_process.kernels.Hyperparameter`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.Hyperparameter.html#sflearn.gaussian_process.kernels.Hyperparameter)(...) | A kernel hyperparameter's specification in form of a namedtuple. |
| [`gaussian_process.kernels.Kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.Kernel.html#sflearn.gaussian_process.kernels.Kernel)() | Base class for all kernels.                                  |
| [`gaussian_process.kernels.Matern`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.Matern.html#sflearn.gaussian_process.kernels.Matern)([...]) | Matern kernel.                                               |
| [`gaussian_process.kernels.PairwiseKernel`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.PairwiseKernel.html#sflearn.gaussian_process.kernels.PairwiseKernel)([...]) | Wrapper for kernels in sflearn.metrics.pairwise.             |
| [`gaussian_process.kernels.Product`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.Product.html#sflearn.gaussian_process.kernels.Product)(k1, k2) | The `Product` kernel takes two kernels �1 and �2 and combines them via |
| [`gaussian_process.kernels.RBF`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.RBF.html#sflearn.gaussian_process.kernels.RBF)([length_scale, ...]) | Radial basis function kernel (aka squared-exponential kernel). |
| [`gaussian_process.kernels.RationalQuadratic`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.RationalQuadratic.html#sflearn.gaussian_process.kernels.RationalQuadratic)([...]) | Rational Quadratic kernel.                                   |
| [`gaussian_process.kernels.Sum`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.Sum.html#sflearn.gaussian_process.kernels.Sum)(k1, k2) | The `Sum` kernel takes two kernels �1 and �2 and combines them via |
| [`gaussian_process.kernels.WhiteKernel`](https://scikit-learn.org/stable/modules/generated/sflearn.gaussian_process.kernels.WhiteKernel.html#sflearn.gaussian_process.kernels.WhiteKernel)([...]) | White kernel.                                                |

## [`sflearn.impute`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.impute): Impute

Transformers for missing value imputation

**User guide:** See the [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html#impute) section for further details.

| [`impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sflearn.impute.SimpleImputer.html#sflearn.impute.SimpleImputer)(*[, missing_values, ...]) | Univariate imputer for completing missing values with simple strategies. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`impute.IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sflearn.impute.IterativeImputer.html#sflearn.impute.IterativeImputer)([estimator, ...]) | Multivariate imputer that estimates each feature from all the others. |
| [`impute.MissingIndicator`](https://scikit-learn.org/stable/modules/generated/sflearn.impute.MissingIndicator.html#sflearn.impute.MissingIndicator)(*[, missing_values, ...]) | Binary indicators for missing values.                        |
| [`impute.KNNImputer`](https://scikit-learn.org/stable/modules/generated/sflearn.impute.KNNImputer.html#sflearn.impute.KNNImputer)(*[, missing_values, ...]) | Imputation for completing missing values using k-Nearest Neighbors. |

## [`sflearn.inspection`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.inspection): Inspection

The [`sflearn.inspection`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.inspection) module includes tools for model inspection.

| [`inspection.partial_dependence`](https://scikit-learn.org/stable/modules/generated/sflearn.inspection.partial_dependence.html#sflearn.inspection.partial_dependence)(estimator, X, ...) | Partial dependence of `features`.                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`inspection.permutation_importance`](https://scikit-learn.org/stable/modules/generated/sflearn.inspection.permutation_importance.html#sflearn.inspection.permutation_importance)(estimator, ...) | Permutation importance for feature evaluation [[Rd9e56ef97513-BRE\]](https://scikit-learn.org/stable/modules/generated/sflearn.inspection.permutation_importance.html#rd9e56ef97513-bre). |

### Plotting

| [`inspection.DecisionBoundaryDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.inspection.DecisionBoundaryDisplay.html#sflearn.inspection.DecisionBoundaryDisplay)(*, xx0, ...) | Decisions boundary visualization. |
| ------------------------------------------------------------ | --------------------------------- |
| [`inspection.PartialDependenceDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.inspection.PartialDependenceDisplay.html#sflearn.inspection.PartialDependenceDisplay)(...[, ...]) | Partial Dependence Plot (PDP).    |

## [`sflearn.isotonic`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.isotonic): Isotonic regression

**User guide:** See the [Isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html#isotonic) section for further details.

| [`isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sflearn.isotonic.IsotonicRegression.html#sflearn.isotonic.IsotonicRegression)(*[, y_min, ...]) | Isotonic regression model. |
| ------------------------------------------------------------ | -------------------------- |
|                                                              |                            |

| [`isotonic.check_increasing`](https://scikit-learn.org/stable/modules/generated/sflearn.isotonic.check_increasing.html#sflearn.isotonic.check_increasing)(x, y) | Determine whether y is monotonically correlated with x. |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| [`isotonic.isotonic_regression`](https://scikit-learn.org/stable/modules/generated/sflearn.isotonic.isotonic_regression.html#sflearn.isotonic.isotonic_regression)(y, *[, ...]) | Solve the isotonic regression model.                    |

## [`sflearn.kernel_approximation`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.kernel_approximation): Kernel Approximation

The [`sflearn.kernel_approximation`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.kernel_approximation) module implements several approximate kernel feature maps based on Fourier transforms and Count Sketches.

**User guide:** See the [Kernel Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation) section for further details.

| [`kernel_approximation.AdditiveChi2Sampler`](https://scikit-learn.org/stable/modules/generated/sflearn.kernel_approximation.AdditiveChi2Sampler.html#sflearn.kernel_approximation.AdditiveChi2Sampler)(*) | Approximate feature map for additive chi2 kernel.            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`kernel_approximation.Nystroem`](https://scikit-learn.org/stable/modules/generated/sflearn.kernel_approximation.Nystroem.html#sflearn.kernel_approximation.Nystroem)([kernel, ...]) | Approximate a kernel map using a subset of the training data. |
| [`kernel_approximation.PolynomialCountSketch`](https://scikit-learn.org/stable/modules/generated/sflearn.kernel_approximation.PolynomialCountSketch.html#sflearn.kernel_approximation.PolynomialCountSketch)(*) | Polynomial kernel approximation via Tensor Sketch.           |
| [`kernel_approximation.RBFSampler`](https://scikit-learn.org/stable/modules/generated/sflearn.kernel_approximation.RBFSampler.html#sflearn.kernel_approximation.RBFSampler)(*[, gamma, ...]) | Approximate a RBF kernel feature map using random Fourier features. |
| [`kernel_approximation.SkewedChi2Sampler`](https://scikit-learn.org/stable/modules/generated/sflearn.kernel_approximation.SkewedChi2Sampler.html#sflearn.kernel_approximation.SkewedChi2Sampler)(*[, ...]) | Approximate feature map for "skewed chi-squared" kernel.     |

## [`sflearn.kernel_ridge`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.kernel_ridge): Kernel Ridge Regression

Module [`sflearn.kernel_ridge`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.kernel_ridge) implements kernel ridge regression.

**User guide:** See the [Kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge) section for further details.

| [`kernel_ridge.KernelRidge`](https://scikit-learn.org/stable/modules/generated/sflearn.kernel_ridge.KernelRidge.html#sflearn.kernel_ridge.KernelRidge)([alpha, kernel, ...]) | Kernel ridge regression. |
| ------------------------------------------------------------ | ------------------------ |
|                                                              |                          |

## [`sflearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.linear_model): Linear Models

The [`sflearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.linear_model) module implements a variety of linear models.

**User guide:** See the [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#linear-model) section for further details.

The following subsections are only rough guidelines: the same estimator can fall into multiple categories, depending on its parameters.

### Linear classifiers

| [`linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LogisticRegression.html#sflearn.linear_model.LogisticRegression)([penalty, ...]) | Logistic Regression (aka logit, MaxEnt) classifier.          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LogisticRegressionCV.html#sflearn.linear_model.LogisticRegressionCV)(*[, Cs, ...]) | Logistic Regression CV (aka logit, MaxEnt) classifier.       |
| [`linear_model.PassiveAggressiveClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.PassiveAggressiveClassifier.html#sflearn.linear_model.PassiveAggressiveClassifier)(*) | Passive Aggressive Classifier.                               |
| [`linear_model.Perceptron`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.Perceptron.html#sflearn.linear_model.Perceptron)(*[, penalty, alpha, ...]) | Linear perceptron classifier.                                |
| [`linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.RidgeClassifier.html#sflearn.linear_model.RidgeClassifier)([alpha, ...]) | Classifier using Ridge regression.                           |
| [`linear_model.RidgeClassifierCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.RidgeClassifierCV.html#sflearn.linear_model.RidgeClassifierCV)([alphas, ...]) | Ridge classifier with built-in cross-validation.             |
| [`linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.SGDClassifier.html#sflearn.linear_model.SGDClassifier)([loss, penalty, ...]) | Linear classifiers (SVM, logistic regression, etc.) with SGD training. |
| [`linear_model.SGDOneClassSVM`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.SGDOneClassSVM.html#sflearn.linear_model.SGDOneClassSVM)([nu, ...]) | Solves linear One-Class SVM using Stochastic Gradient Descent. |

### Classical linear regressors

| [`linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LinearRegression.html#sflearn.linear_model.LinearRegression)(*[, ...]) | Ordinary least squares Linear Regression.                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.Ridge.html#sflearn.linear_model.Ridge)([alpha, fit_intercept, ...]) | Linear least squares with l2 regularization.                 |
| [`linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.RidgeCV.html#sflearn.linear_model.RidgeCV)([alphas, ...]) | Ridge regression with built-in cross-validation.             |
| [`linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.SGDRegressor.html#sflearn.linear_model.SGDRegressor)([loss, penalty, ...]) | Linear model fitted by minimizing a regularized empirical loss with SGD. |

### Regressors with variable selection

The following estimators have built-in variable selection fitting procedures, but any estimator using a L1 or elastic-net penalty also performs variable selection: typically [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.SGDRegressor.html#sflearn.linear_model.SGDRegressor) or [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.SGDClassifier.html#sflearn.linear_model.SGDClassifier) with an appropriate penalty.

| [`linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.ElasticNet.html#sflearn.linear_model.ElasticNet)([alpha, l1_ratio, ...]) | Linear regression with combined L1 and L2 priors as regularizer. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.ElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.ElasticNetCV.html#sflearn.linear_model.ElasticNetCV)(*[, l1_ratio, ...]) | Elastic Net model with iterative fitting along a regularization path. |
| [`linear_model.Lars`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.Lars.html#sflearn.linear_model.Lars)(*[, fit_intercept, ...]) | Least Angle Regression model a.k.a.                          |
| [`linear_model.LarsCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LarsCV.html#sflearn.linear_model.LarsCV)(*[, fit_intercept, ...]) | Cross-validated Least Angle Regression model.                |
| [`linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.Lasso.html#sflearn.linear_model.Lasso)([alpha, fit_intercept, ...]) | Linear Model trained with L1 prior as regularizer (aka the Lasso). |
| [`linear_model.LassoCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LassoCV.html#sflearn.linear_model.LassoCV)(*[, eps, n_alphas, ...]) | Lasso linear model with iterative fitting along a regularization path. |
| [`linear_model.LassoLars`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LassoLars.html#sflearn.linear_model.LassoLars)([alpha, ...]) | Lasso model fit with Least Angle Regression a.k.a.           |
| [`linear_model.LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LassoLarsCV.html#sflearn.linear_model.LassoLarsCV)(*[, fit_intercept, ...]) | Cross-validated Lasso, using the LARS algorithm.             |
| [`linear_model.LassoLarsIC`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.LassoLarsIC.html#sflearn.linear_model.LassoLarsIC)([criterion, ...]) | Lasso model fit with Lars using BIC or AIC for model selection. |
| [`linear_model.OrthogonalMatchingPursuit`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.OrthogonalMatchingPursuit.html#sflearn.linear_model.OrthogonalMatchingPursuit)(*[, ...]) | Orthogonal Matching Pursuit model (OMP).                     |
| [`linear_model.OrthogonalMatchingPursuitCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.OrthogonalMatchingPursuitCV.html#sflearn.linear_model.OrthogonalMatchingPursuitCV)(*) | Cross-validated Orthogonal Matching Pursuit model (OMP).     |

### Bayesian regressors

| [`linear_model.ARDRegression`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.ARDRegression.html#sflearn.linear_model.ARDRegression)(*[, n_iter, tol, ...]) | Bayesian ARD regression.   |
| ------------------------------------------------------------ | -------------------------- |
| [`linear_model.BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.BayesianRidge.html#sflearn.linear_model.BayesianRidge)(*[, n_iter, tol, ...]) | Bayesian ridge regression. |

### Multi-task linear regressors with variable selection

These estimators fit multiple regression problems (or tasks) jointly, while inducing sparse coefficients. While the inferred coefficients may differ between the tasks, they are constrained to agree on the features that are selected (non-zero coefficients).

| [`linear_model.MultiTaskElasticNet`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.MultiTaskElasticNet.html#sflearn.linear_model.MultiTaskElasticNet)([alpha, ...]) | Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.MultiTaskElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.MultiTaskElasticNetCV.html#sflearn.linear_model.MultiTaskElasticNetCV)(*[, ...]) | Multi-task L1/L2 ElasticNet with built-in cross-validation.  |
| [`linear_model.MultiTaskLasso`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.MultiTaskLasso.html#sflearn.linear_model.MultiTaskLasso)([alpha, ...]) | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer. |
| [`linear_model.MultiTaskLassoCV`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.MultiTaskLassoCV.html#sflearn.linear_model.MultiTaskLassoCV)(*[, eps, ...]) | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer. |

### Outlier-robust regressors

Any estimator using the Huber loss would also be robust to outliers, e.g. [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.SGDRegressor.html#sflearn.linear_model.SGDRegressor) with `loss='huber'`.

| [`linear_model.HuberRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.HuberRegressor.html#sflearn.linear_model.HuberRegressor)(*[, epsilon, ...]) | L2-regularized linear regression model that is robust to outliers. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.QuantileRegressor.html#sflearn.linear_model.QuantileRegressor)(*[, ...]) | Linear regression model that predicts conditional quantiles. |
| [`linear_model.RANSACRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.RANSACRegressor.html#sflearn.linear_model.RANSACRegressor)([estimator, ...]) | RANSAC (RANdom SAmple Consensus) algorithm.                  |
| [`linear_model.TheilSenRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.TheilSenRegressor.html#sflearn.linear_model.TheilSenRegressor)(*[, ...]) | Theil-Sen Estimator: robust multivariate regression model.   |

### Generalized linear models (GLM) for regression

These models allow for response variables to have error distributions other than a normal distribution:

| [`linear_model.PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.PoissonRegressor.html#sflearn.linear_model.PoissonRegressor)(*[, alpha, ...]) | Generalized Linear Model with a Poisson distribution. |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [`linear_model.TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.TweedieRegressor.html#sflearn.linear_model.TweedieRegressor)(*[, power, ...]) | Generalized Linear Model with a Tweedie distribution. |
| [`linear_model.GammaRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.GammaRegressor.html#sflearn.linear_model.GammaRegressor)(*[, alpha, ...]) | Generalized Linear Model with a Gamma distribution.   |

### Miscellaneous

| [`linear_model.PassiveAggressiveRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.PassiveAggressiveRegressor.html#sflearn.linear_model.PassiveAggressiveRegressor)(*[, ...]) | Passive Aggressive Regressor.                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.enet_path`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.enet_path.html#sflearn.linear_model.enet_path)(X, y, *[, l1_ratio, ...]) | Compute elastic net path with coordinate descent.            |
| [`linear_model.lars_path`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.lars_path.html#sflearn.linear_model.lars_path)(X, y[, Xy, Gram, ...]) | Compute Least Angle Regression or Lasso path using the LARS algorithm [1]. |
| [`linear_model.lars_path_gram`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.lars_path_gram.html#sflearn.linear_model.lars_path_gram)(Xy, Gram, *, ...) | The lars_path in the sufficient stats mode [1].              |
| [`linear_model.lasso_path`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.lasso_path.html#sflearn.linear_model.lasso_path)(X, y, *[, eps, ...]) | Compute Lasso path with coordinate descent.                  |
| [`linear_model.orthogonal_mp`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.orthogonal_mp.html#sflearn.linear_model.orthogonal_mp)(X, y, *[, ...]) | Orthogonal Matching Pursuit (OMP).                           |
| [`linear_model.orthogonal_mp_gram`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.orthogonal_mp_gram.html#sflearn.linear_model.orthogonal_mp_gram)(Gram, Xy, *) | Gram Orthogonal Matching Pursuit (OMP).                      |
| [`linear_model.ridge_regression`](https://scikit-learn.org/stable/modules/generated/sflearn.linear_model.ridge_regression.html#sflearn.linear_model.ridge_regression)(X, y, alpha, *) | Solve the ridge equation by the method of normal equations.  |

## [`sflearn.manifold`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.manifold): Manifold Learning

The [`sflearn.manifold`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.manifold) module implements data embedding techniques.

**User guide:** See the [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html#manifold) section for further details.

| [`manifold.Isomap`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.Isomap.html#sflearn.manifold.Isomap)(*[, n_neighbors, radius, ...]) | Isomap Embedding.                                           |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| [`manifold.LocallyLinearEmbedding`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.LocallyLinearEmbedding.html#sflearn.manifold.LocallyLinearEmbedding)(*[, ...]) | Locally Linear Embedding.                                   |
| [`manifold.MDS`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.MDS.html#sflearn.manifold.MDS)([n_components, metric, n_init, ...]) | Multidimensional scaling.                                   |
| [`manifold.SpectralEmbedding`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.SpectralEmbedding.html#sflearn.manifold.SpectralEmbedding)([n_components, ...]) | Spectral embedding for non-linear dimensionality reduction. |
| [`manifold.TSNE`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.TSNE.html#sflearn.manifold.TSNE)([n_components, perplexity, ...]) | T-distributed Stochastic Neighbor Embedding.                |

| [`manifold.locally_linear_embedding`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.locally_linear_embedding.html#sflearn.manifold.locally_linear_embedding)(X, *, ...) | Perform a Locally Linear Embedding analysis on the data.     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`manifold.smacof`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.smacof.html#sflearn.manifold.smacof)(dissimilarities, *[, ...]) | Compute multidimensional scaling using the SMACOF algorithm. |
| [`manifold.spectral_embedding`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.spectral_embedding.html#sflearn.manifold.spectral_embedding)(adjacency, *[, ...]) | Project the sample on the first eigenvectors of the graph Laplacian. |
| [`manifold.trustworthiness`](https://scikit-learn.org/stable/modules/generated/sflearn.manifold.trustworthiness.html#sflearn.manifold.trustworthiness)(X, X_embedded, *[, ...]) | Indicate to what extent the local structure is retained.     |

## [`sflearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.metrics): Metrics

See the [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) section and the [Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics) section of the user guide for further details.

The [`sflearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.metrics) module includes score functions, performance metrics and pairwise metrics and distance computations.

### Model Selection Interface

See the [The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) section of the user guide for further details.

| [`metrics.check_scoring`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.check_scoring.html#sflearn.metrics.check_scoring)(estimator[, scoring, ...]) | Determine scorer from user options.                       |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`metrics.get_scorer`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.get_scorer.html#sflearn.metrics.get_scorer)(scoring) | Get a scorer from string.                                 |
| [`metrics.get_scorer_names`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.get_scorer_names.html#sflearn.metrics.get_scorer_names)() | Get the names of all available scorers.                   |
| [`metrics.make_scorer`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.make_scorer.html#sflearn.metrics.make_scorer)(score_func, *[, ...]) | Make a scorer from a performance metric or loss function. |

### Classification metrics

See the [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) section of the user guide for further details.

| [`metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.accuracy_score.html#sflearn.metrics.accuracy_score)(y_true, y_pred, *[, ...]) | Accuracy classification score.                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.auc`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.auc.html#sflearn.metrics.auc)(x, y) | Compute Area Under the Curve (AUC) using the trapezoidal rule. |
| [`metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.average_precision_score.html#sflearn.metrics.average_precision_score)(y_true, ...) | Compute average precision (AP) from prediction scores.       |
| [`metrics.balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.balanced_accuracy_score.html#sflearn.metrics.balanced_accuracy_score)(y_true, ...) | Compute the balanced accuracy.                               |
| [`metrics.brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.brier_score_loss.html#sflearn.metrics.brier_score_loss)(y_true, y_prob, *) | Compute the Brier score loss.                                |
| [`metrics.class_likelihood_ratios`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.class_likelihood_ratios.html#sflearn.metrics.class_likelihood_ratios)(y_true, ...) | Compute binary classification positive and negative likelihood ratios. |
| [`metrics.classification_report`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.classification_report.html#sflearn.metrics.classification_report)(y_true, y_pred, *) | Build a text report showing the main classification metrics. |
| [`metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.cohen_kappa_score.html#sflearn.metrics.cohen_kappa_score)(y1, y2, *[, ...]) | Compute Cohen's kappa: a statistic that measures inter-annotator agreement. |
| [`metrics.confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.confusion_matrix.html#sflearn.metrics.confusion_matrix)(y_true, y_pred, *) | Compute confusion matrix to evaluate the accuracy of a classification. |
| [`metrics.dcg_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.dcg_score.html#sflearn.metrics.dcg_score)(y_true, y_score, *[, k, ...]) | Compute Discounted Cumulative Gain.                          |
| [`metrics.det_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.det_curve.html#sflearn.metrics.det_curve)(y_true, y_score[, ...]) | Compute error rates for different probability thresholds.    |
| [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.f1_score.html#sflearn.metrics.f1_score)(y_true, y_pred, *[, ...]) | Compute the F1 score, also known as balanced F-score or F-measure. |
| [`metrics.fbeta_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.fbeta_score.html#sflearn.metrics.fbeta_score)(y_true, y_pred, *, beta) | Compute the F-beta score.                                    |
| [`metrics.hamming_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.hamming_loss.html#sflearn.metrics.hamming_loss)(y_true, y_pred, *[, ...]) | Compute the average Hamming loss.                            |
| [`metrics.hinge_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.hinge_loss.html#sflearn.metrics.hinge_loss)(y_true, pred_decision, *) | Average hinge loss (non-regularized).                        |
| [`metrics.jaccard_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.jaccard_score.html#sflearn.metrics.jaccard_score)(y_true, y_pred, *[, ...]) | Jaccard similarity coefficient score.                        |
| [`metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.log_loss.html#sflearn.metrics.log_loss)(y_true, y_pred, *[, eps, ...]) | Log loss, aka logistic loss or cross-entropy loss.           |
| [`metrics.matthews_corrcoef`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.matthews_corrcoef.html#sflearn.metrics.matthews_corrcoef)(y_true, y_pred, *) | Compute the Matthews correlation coefficient (MCC).          |
| [`metrics.multilabel_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.multilabel_confusion_matrix.html#sflearn.metrics.multilabel_confusion_matrix)(y_true, ...) | Compute a confusion matrix for each class or sample.         |
| [`metrics.ndcg_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.ndcg_score.html#sflearn.metrics.ndcg_score)(y_true, y_score, *[, k, ...]) | Compute Normalized Discounted Cumulative Gain.               |
| [`metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.precision_recall_curve.html#sflearn.metrics.precision_recall_curve)(y_true, ...) | Compute precision-recall pairs for different probability thresholds. |
| [`metrics.precision_recall_fscore_support`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.precision_recall_fscore_support.html#sflearn.metrics.precision_recall_fscore_support)(...) | Compute precision, recall, F-measure and support for each class. |
| [`metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.precision_score.html#sflearn.metrics.precision_score)(y_true, y_pred, *[, ...]) | Compute the precision.                                       |
| [`metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.recall_score.html#sflearn.metrics.recall_score)(y_true, y_pred, *[, ...]) | Compute the recall.                                          |
| [`metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.roc_auc_score.html#sflearn.metrics.roc_auc_score)(y_true, y_score, *[, ...]) | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)     from prediction scores. |
| [`metrics.roc_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.roc_curve.html#sflearn.metrics.roc_curve)(y_true, y_score, *[, ...]) | Compute Receiver operating characteristic (ROC).             |
| [`metrics.top_k_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.top_k_accuracy_score.html#sflearn.metrics.top_k_accuracy_score)(y_true, y_score, *) | Top-k Accuracy classification score.                         |
| [`metrics.zero_one_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.zero_one_loss.html#sflearn.metrics.zero_one_loss)(y_true, y_pred, *[, ...]) | Zero-one classification loss.                                |

### Regression metrics

See the [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) section of the user guide for further details.

| [`metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.explained_variance_score.html#sflearn.metrics.explained_variance_score)(y_true, ...) | Explained variance regression score function.                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.max_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.max_error.html#sflearn.metrics.max_error)(y_true, y_pred) | The max_error metric calculates the maximum residual error.  |
| [`metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_absolute_error.html#sflearn.metrics.mean_absolute_error)(y_true, y_pred, *) | Mean absolute error regression loss.                         |
| [`metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_squared_error.html#sflearn.metrics.mean_squared_error)(y_true, y_pred, *) | Mean squared error regression loss.                          |
| [`metrics.mean_squared_log_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_squared_log_error.html#sflearn.metrics.mean_squared_log_error)(y_true, y_pred, *) | Mean squared logarithmic error regression loss.              |
| [`metrics.median_absolute_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.median_absolute_error.html#sflearn.metrics.median_absolute_error)(y_true, y_pred, *) | Median absolute error regression loss.                       |
| [`metrics.mean_absolute_percentage_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_absolute_percentage_error.html#sflearn.metrics.mean_absolute_percentage_error)(...) | Mean absolute percentage error (MAPE) regression loss.       |
| [`metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.r2_score.html#sflearn.metrics.r2_score)(y_true, y_pred, *[, ...]) | �2 (coefficient of determination) regression score function. |
| [`metrics.mean_poisson_deviance`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_poisson_deviance.html#sflearn.metrics.mean_poisson_deviance)(y_true, y_pred, *) | Mean Poisson deviance regression loss.                       |
| [`metrics.mean_gamma_deviance`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_gamma_deviance.html#sflearn.metrics.mean_gamma_deviance)(y_true, y_pred, *) | Mean Gamma deviance regression loss.                         |
| [`metrics.mean_tweedie_deviance`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_tweedie_deviance.html#sflearn.metrics.mean_tweedie_deviance)(y_true, y_pred, *) | Mean Tweedie deviance regression loss.                       |
| [`metrics.d2_tweedie_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.d2_tweedie_score.html#sflearn.metrics.d2_tweedie_score)(y_true, y_pred, *) | D^2 regression score function, fraction of Tweedie deviance explained. |
| [`metrics.mean_pinball_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mean_pinball_loss.html#sflearn.metrics.mean_pinball_loss)(y_true, y_pred, *) | Pinball loss for quantile regression.                        |
| [`metrics.d2_pinball_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.d2_pinball_score.html#sflearn.metrics.d2_pinball_score)(y_true, y_pred, *) | �2 regression score function, fraction of pinball loss explained. |
| [`metrics.d2_absolute_error_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.d2_absolute_error_score.html#sflearn.metrics.d2_absolute_error_score)(y_true, ...) | �2 regression score function,     fraction of absolute error explained. |

### Multilabel ranking metrics

See the [Multilabel ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) section of the user guide for further details.

| [`metrics.coverage_error`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.coverage_error.html#sflearn.metrics.coverage_error)(y_true, y_score, *[, ...]) | Coverage error measure.                  |
| ------------------------------------------------------------ | ---------------------------------------- |
| [`metrics.label_ranking_average_precision_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.label_ranking_average_precision_score.html#sflearn.metrics.label_ranking_average_precision_score)(...) | Compute ranking-based average precision. |
| [`metrics.label_ranking_loss`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.label_ranking_loss.html#sflearn.metrics.label_ranking_loss)(y_true, y_score, *) | Compute Ranking loss measure.            |

### Clustering metrics

See the [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation) section of the user guide for further details.

The [`sflearn.metrics.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.metrics.cluster) submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:

- supervised, which uses a ground truth class values for each sample.
- unsupervised, which does not and measures the ‘quality’ of the model itself.

| [`metrics.adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.adjusted_mutual_info_score.html#sflearn.metrics.adjusted_mutual_info_score)(...[, ...]) | Adjusted Mutual Information between two clusterings.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.adjusted_rand_score.html#sflearn.metrics.adjusted_rand_score)(labels_true, ...) | Rand index adjusted for chance.                              |
| [`metrics.calinski_harabasz_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.calinski_harabasz_score.html#sflearn.metrics.calinski_harabasz_score)(X, labels) | Compute the Calinski and Harabasz score.                     |
| [`metrics.davies_bouldin_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.davies_bouldin_score.html#sflearn.metrics.davies_bouldin_score)(X, labels) | Compute the Davies-Bouldin score.                            |
| [`metrics.completeness_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.completeness_score.html#sflearn.metrics.completeness_score)(labels_true, ...) | Compute completeness metric of a cluster labeling given a ground truth. |
| [`metrics.cluster.contingency_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.cluster.contingency_matrix.html#sflearn.metrics.cluster.contingency_matrix)(...[, ...]) | Build a contingency matrix describing the relationship between labels. |
| [`metrics.cluster.pair_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.cluster.pair_confusion_matrix.html#sflearn.metrics.cluster.pair_confusion_matrix)(...) | Pair confusion matrix arising from two clusterings [[R9ca8fd06d29a-1\]](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.cluster.pair_confusion_matrix.html#r9ca8fd06d29a-1). |
| [`metrics.fowlkes_mallows_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.fowlkes_mallows_score.html#sflearn.metrics.fowlkes_mallows_score)(labels_true, ...) | Measure the similarity of two clusterings of a set of points. |
| [`metrics.homogeneity_completeness_v_measure`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.homogeneity_completeness_v_measure.html#sflearn.metrics.homogeneity_completeness_v_measure)(...) | Compute the homogeneity and completeness and V-Measure scores at once. |
| [`metrics.homogeneity_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.homogeneity_score.html#sflearn.metrics.homogeneity_score)(labels_true, ...) | Homogeneity metric of a cluster labeling given a ground truth. |
| [`metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.mutual_info_score.html#sflearn.metrics.mutual_info_score)(labels_true, ...) | Mutual Information between two clusterings.                  |
| [`metrics.normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.normalized_mutual_info_score.html#sflearn.metrics.normalized_mutual_info_score)(...[, ...]) | Normalized Mutual Information between two clusterings.       |
| [`metrics.rand_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.rand_score.html#sflearn.metrics.rand_score)(labels_true, labels_pred) | Rand index.                                                  |
| [`metrics.silhouette_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.silhouette_score.html#sflearn.metrics.silhouette_score)(X, labels, *[, ...]) | Compute the mean Silhouette Coefficient of all samples.      |
| [`metrics.silhouette_samples`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.silhouette_samples.html#sflearn.metrics.silhouette_samples)(X, labels, *[, ...]) | Compute the Silhouette Coefficient for each sample.          |
| [`metrics.v_measure_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.v_measure_score.html#sflearn.metrics.v_measure_score)(labels_true, ...[, beta]) | V-measure cluster labeling given a ground truth.             |

### Biclustering metrics

See the [Biclustering evaluation](https://scikit-learn.org/stable/modules/biclustering.html#biclustering-evaluation) section of the user guide for further details.

| [`metrics.consensus_score`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.consensus_score.html#sflearn.metrics.consensus_score)(a, b, *[, similarity]) | The similarity of two sets of biclusters. |
| ------------------------------------------------------------ | ----------------------------------------- |
|                                                              |                                           |

### Distance metrics

| [`metrics.DistanceMetric`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.DistanceMetric.html#sflearn.metrics.DistanceMetric) | DistanceMetric class |
| ------------------------------------------------------------ | -------------------- |
|                                                              |                      |

### Pairwise metrics

See the [Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics) section of the user guide for further details.

| [`metrics.pairwise.additive_chi2_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.additive_chi2_kernel.html#sflearn.metrics.pairwise.additive_chi2_kernel)(X[, Y]) | Compute the additive chi-squared kernel between observations in X and Y. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.pairwise.chi2_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.chi2_kernel.html#sflearn.metrics.pairwise.chi2_kernel)(X[, Y, gamma]) | Compute the exponential chi-squared kernel between X and Y.  |
| [`metrics.pairwise.cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.cosine_similarity.html#sflearn.metrics.pairwise.cosine_similarity)(X[, Y, ...]) | Compute cosine similarity between samples in X and Y.        |
| [`metrics.pairwise.cosine_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.cosine_distances.html#sflearn.metrics.pairwise.cosine_distances)(X[, Y]) | Compute cosine distance between samples in X and Y.          |
| [`metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.distance_metrics.html#sflearn.metrics.pairwise.distance_metrics)() | Valid metrics for pairwise_distances.                        |
| [`metrics.pairwise.euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.euclidean_distances.html#sflearn.metrics.pairwise.euclidean_distances)(X[, Y, ...]) | Compute the distance matrix between each pair from a vector array X and Y. |
| [`metrics.pairwise.haversine_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.haversine_distances.html#sflearn.metrics.pairwise.haversine_distances)(X[, Y]) | Compute the Haversine distance between samples in X and Y.   |
| [`metrics.pairwise.kernel_metrics`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.kernel_metrics.html#sflearn.metrics.pairwise.kernel_metrics)() | Valid metrics for pairwise_kernels.                          |
| [`metrics.pairwise.laplacian_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.laplacian_kernel.html#sflearn.metrics.pairwise.laplacian_kernel)(X[, Y, gamma]) | Compute the laplacian kernel between X and Y.                |
| [`metrics.pairwise.linear_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.linear_kernel.html#sflearn.metrics.pairwise.linear_kernel)(X[, Y, ...]) | Compute the linear kernel between X and Y.                   |
| [`metrics.pairwise.manhattan_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.manhattan_distances.html#sflearn.metrics.pairwise.manhattan_distances)(X[, Y, ...]) | Compute the L1 distances between the vectors in X and Y.     |
| [`metrics.pairwise.nan_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.nan_euclidean_distances.html#sflearn.metrics.pairwise.nan_euclidean_distances)(X) | Calculate the euclidean distances in the presence of missing values. |
| [`metrics.pairwise.pairwise_kernels`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.pairwise_kernels.html#sflearn.metrics.pairwise.pairwise_kernels)(X[, Y, ...]) | Compute the kernel between arrays X and optional array Y.    |
| [`metrics.pairwise.polynomial_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.polynomial_kernel.html#sflearn.metrics.pairwise.polynomial_kernel)(X[, Y, ...]) | Compute the polynomial kernel between X and Y.               |
| [`metrics.pairwise.rbf_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.rbf_kernel.html#sflearn.metrics.pairwise.rbf_kernel)(X[, Y, gamma]) | Compute the rbf (gaussian) kernel between X and Y.           |
| [`metrics.pairwise.sigmoid_kernel`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.sigmoid_kernel.html#sflearn.metrics.pairwise.sigmoid_kernel)(X[, Y, ...]) | Compute the sigmoid kernel between X and Y.                  |
| [`metrics.pairwise.paired_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.paired_euclidean_distances.html#sflearn.metrics.pairwise.paired_euclidean_distances)(X, Y) | Compute the paired euclidean distances between X and Y.      |
| [`metrics.pairwise.paired_manhattan_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.paired_manhattan_distances.html#sflearn.metrics.pairwise.paired_manhattan_distances)(X, Y) | Compute the paired L1 distances between X and Y.             |
| [`metrics.pairwise.paired_cosine_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.paired_cosine_distances.html#sflearn.metrics.pairwise.paired_cosine_distances)(X, Y) | Compute the paired cosine distances between X and Y.         |
| [`metrics.pairwise.paired_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise.paired_distances.html#sflearn.metrics.pairwise.paired_distances)(X, Y, *[, ...]) | Compute the paired distances between X and Y.                |
| [`metrics.pairwise_distances`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise_distances.html#sflearn.metrics.pairwise_distances)(X[, Y, metric, ...]) | Compute the distance matrix from a vector array X and optional Y. |
| [`metrics.pairwise_distances_argmin`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise_distances_argmin.html#sflearn.metrics.pairwise_distances_argmin)(X, Y, *[, ...]) | Compute minimum distances between one point and a set of points. |
| [`metrics.pairwise_distances_argmin_min`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise_distances_argmin_min.html#sflearn.metrics.pairwise_distances_argmin_min)(X, Y, *) | Compute minimum distances between one point and a set of points. |
| [`metrics.pairwise_distances_chunked`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.pairwise_distances_chunked.html#sflearn.metrics.pairwise_distances_chunked)(X[, Y, ...]) | Generate a distance matrix chunk by chunk with optional reduction. |

### Plotting

See the [Visualizations](https://scikit-learn.org/stable/visualizations.html#visualizations) section of the user guide for further details.

| [`metrics.ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.ConfusionMatrixDisplay.html#sflearn.metrics.ConfusionMatrixDisplay)(...[, ...]) | Confusion Matrix visualization.                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.DetCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.DetCurveDisplay.html#sflearn.metrics.DetCurveDisplay)(*, fpr, fnr[, ...]) | DET curve visualization.                                     |
| [`metrics.PrecisionRecallDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.PrecisionRecallDisplay.html#sflearn.metrics.PrecisionRecallDisplay)(precision, ...) | Precision Recall visualization.                              |
| [`metrics.PredictionErrorDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.PredictionErrorDisplay.html#sflearn.metrics.PredictionErrorDisplay)(*, y_true, y_pred) | Visualization of the prediction error of a regression model. |
| [`metrics.RocCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.metrics.RocCurveDisplay.html#sflearn.metrics.RocCurveDisplay)(*, fpr, tpr[, ...]) | ROC Curve visualization.                                     |
| [`calibration.CalibrationDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.calibration.CalibrationDisplay.html#sflearn.calibration.CalibrationDisplay)(prob_true, ...) | Calibration curve (also known as reliability diagram) visualization. |

## [`sflearn.mixture`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.mixture): Gaussian Mixture Models

The [`sflearn.mixture`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.mixture) module implements mixture modeling algorithms.

**User guide:** See the [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#mixture) section for further details.

| [`mixture.BayesianGaussianMixture`](https://scikit-learn.org/stable/modules/generated/sflearn.mixture.BayesianGaussianMixture.html#sflearn.mixture.BayesianGaussianMixture)(*[, ...]) | Variational Bayesian estimation of a Gaussian mixture. |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [`mixture.GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sflearn.mixture.GaussianMixture.html#sflearn.mixture.GaussianMixture)([n_components, ...]) | Gaussian Mixture.                                      |

## [`sflearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.model_selection): Model Selection

**User guide:** See the [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation), [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) and [Learning curve](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve) sections for further details.

### Splitter Classes

| [`model_selection.GroupKFold`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.GroupKFold.html#sflearn.model_selection.GroupKFold)([n_splits]) | K-fold iterator variant with non-overlapping groups.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.GroupShuffleSplit.html#sflearn.model_selection.GroupShuffleSplit)([...]) | Shuffle-Group(s)-Out cross-validation iterator               |
| [`model_selection.KFold`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.KFold.html#sflearn.model_selection.KFold)([n_splits, shuffle, ...]) | K-Folds cross-validator                                      |
| [`model_selection.LeaveOneGroupOut`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.LeaveOneGroupOut.html#sflearn.model_selection.LeaveOneGroupOut)() | Leave One Group Out cross-validator                          |
| [`model_selection.LeavePGroupsOut`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.LeavePGroupsOut.html#sflearn.model_selection.LeavePGroupsOut)(n_groups) | Leave P Group(s) Out cross-validator                         |
| [`model_selection.LeaveOneOut`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.LeaveOneOut.html#sflearn.model_selection.LeaveOneOut)() | Leave-One-Out cross-validator                                |
| [`model_selection.LeavePOut`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.LeavePOut.html#sflearn.model_selection.LeavePOut)(p) | Leave-P-Out cross-validator                                  |
| [`model_selection.PredefinedSplit`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.PredefinedSplit.html#sflearn.model_selection.PredefinedSplit)(test_fold) | Predefined split cross-validator                             |
| [`model_selection.RepeatedKFold`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.RepeatedKFold.html#sflearn.model_selection.RepeatedKFold)(*[, n_splits, ...]) | Repeated K-Fold cross validator.                             |
| [`model_selection.RepeatedStratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.RepeatedStratifiedKFold.html#sflearn.model_selection.RepeatedStratifiedKFold)(*[, ...]) | Repeated Stratified K-Fold cross validator.                  |
| [`model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.ShuffleSplit.html#sflearn.model_selection.ShuffleSplit)([n_splits, ...]) | Random permutation cross-validator                           |
| [`model_selection.StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.StratifiedKFold.html#sflearn.model_selection.StratifiedKFold)([n_splits, ...]) | Stratified K-Folds cross-validator.                          |
| [`model_selection.StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.StratifiedShuffleSplit.html#sflearn.model_selection.StratifiedShuffleSplit)([...]) | Stratified ShuffleSplit cross-validator                      |
| [`model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.StratifiedGroupKFold.html#sflearn.model_selection.StratifiedGroupKFold)([...]) | Stratified K-Folds iterator variant with non-overlapping groups. |
| [`model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.TimeSeriesSplit.html#sflearn.model_selection.TimeSeriesSplit)([n_splits, ...]) | Time Series cross-validator                                  |

### Splitter Functions

| [`model_selection.check_cv`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.check_cv.html#sflearn.model_selection.check_cv)([cv, y, classifier]) | Input checker utility for building a cross-validator.        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.train_test_split.html#sflearn.model_selection.train_test_split)(*arrays[, ...]) | Split arrays or matrices into random train and test subsets. |

### Hyper-parameter optimizers

| [`model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.GridSearchCV.html#sflearn.model_selection.GridSearchCV)(estimator, ...) | Exhaustive search over specified parameter values for an estimator. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.HalvingGridSearchCV`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.HalvingGridSearchCV.html#sflearn.model_selection.HalvingGridSearchCV)(...[, ...]) | Search over specified parameter values with successive halving. |
| [`model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.ParameterGrid.html#sflearn.model_selection.ParameterGrid)(param_grid) | Grid of parameters with a discrete number of values for each. |
| [`model_selection.ParameterSampler`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.ParameterSampler.html#sflearn.model_selection.ParameterSampler)(...[, ...]) | Generator on parameters sampled from given distributions.    |
| [`model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.RandomizedSearchCV.html#sflearn.model_selection.RandomizedSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |
| [`model_selection.HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.HalvingRandomSearchCV.html#sflearn.model_selection.HalvingRandomSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |

### Model validation

| [`model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.cross_validate.html#sflearn.model_selection.cross_validate)(estimator, X) | Evaluate metric(s) by cross-validation and also record fit/score times. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.cross_val_predict.html#sflearn.model_selection.cross_val_predict)(estimator, X) | Generate cross-validated estimates for each input data point. |
| [`model_selection.cross_val_score`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.cross_val_score.html#sflearn.model_selection.cross_val_score)(estimator, X) | Evaluate a score by cross-validation.                        |
| [`model_selection.learning_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.learning_curve.html#sflearn.model_selection.learning_curve)(estimator, X, ...) | Learning curve.                                              |
| [`model_selection.permutation_test_score`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.permutation_test_score.html#sflearn.model_selection.permutation_test_score)(...) | Evaluate the significance of a cross-validated score with permutations. |
| [`model_selection.validation_curve`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.validation_curve.html#sflearn.model_selection.validation_curve)(estimator, ...) | Validation curve.                                            |

### Visualization

| [`model_selection.LearningCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sflearn.model_selection.LearningCurveDisplay.html#sflearn.model_selection.LearningCurveDisplay)(*, ...) | Learning Curve visualization. |
| ------------------------------------------------------------ | ----------------------------- |
|                                                              |                               |

## [`sflearn.multiclass`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.multiclass): Multiclass classification

### Multiclass classification strategies

- This module implements multiclass learning algorithms:

  one-vs-the-rest / one-vs-allone-vs-oneerror correcting output codes

The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. For example, it is possible to use these estimators to turn a binary classifier or a regressor into a multiclass classifier. It is also possible to use these estimators with multiclass estimators in the hope that their accuracy or runtime performance improves.

All classifiers in scikit-learn implement multiclass classification; you only need to use this module if you want to experiment with custom multiclass strategies.

The one-vs-the-rest meta-classifier also implements a `predict_proba` method, so long as such a method is implemented by the base classifier. This method returns probabilities of class membership in both the single label and multilabel case. Note that in the multilabel case, probabilities are the marginal probability that a given sample falls in the given class. As such, in the multilabel case the sum of these probabilities over all possible labels for a given sample *will not* sum to unity, as they do in the single label case.

**User guide:** See the [Multiclass classification](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification) section for further details.

| [`multiclass.OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.multiclass.OneVsRestClassifier.html#sflearn.multiclass.OneVsRestClassifier)(estimator, *) | One-vs-the-rest (OvR) multiclass strategy.          |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [`multiclass.OneVsOneClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.multiclass.OneVsOneClassifier.html#sflearn.multiclass.OneVsOneClassifier)(estimator, *) | One-vs-one multiclass strategy.                     |
| [`multiclass.OutputCodeClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.multiclass.OutputCodeClassifier.html#sflearn.multiclass.OutputCodeClassifier)(estimator, *) | (Error-Correcting) Output-Code multiclass strategy. |

## [`sflearn.multioutput`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.multioutput): Multioutput regression and classification

This module implements multioutput regression and classification.

The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. The meta-estimator extends single output estimators to multioutput estimators.

**User guide:** See the [Multilabel classification](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification), [Multiclass-multioutput classification](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-multioutput-classification), and [Multioutput regression](https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression) sections for further details.

| [`multioutput.ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sflearn.multioutput.ClassifierChain.html#sflearn.multioutput.ClassifierChain)(base_estimator, *) | A multi-label model that arranges binary classifiers into a chain. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`multioutput.MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.multioutput.MultiOutputRegressor.html#sflearn.multioutput.MultiOutputRegressor)(estimator, *) | Multi target regression.                                     |
| [`multioutput.MultiOutputClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.multioutput.MultiOutputClassifier.html#sflearn.multioutput.MultiOutputClassifier)(estimator, *) | Multi target classification.                                 |
| [`multioutput.RegressorChain`](https://scikit-learn.org/stable/modules/generated/sflearn.multioutput.RegressorChain.html#sflearn.multioutput.RegressorChain)(base_estimator, *) | A multi-label model that arranges regressions into a chain.  |

## [`sflearn.naive_bayes`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.naive_bayes): Naive Bayes

The [`sflearn.naive_bayes`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.naive_bayes) module implements Naive Bayes algorithms. These are supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions.

**User guide:** See the [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes) section for further details.

| [`naive_bayes.BernoulliNB`](https://scikit-learn.org/stable/modules/generated/sflearn.naive_bayes.BernoulliNB.html#sflearn.naive_bayes.BernoulliNB)(*[, alpha, ...]) | Naive Bayes classifier for multivariate Bernoulli models.    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`naive_bayes.CategoricalNB`](https://scikit-learn.org/stable/modules/generated/sflearn.naive_bayes.CategoricalNB.html#sflearn.naive_bayes.CategoricalNB)(*[, alpha, ...]) | Naive Bayes classifier for categorical features.             |
| [`naive_bayes.ComplementNB`](https://scikit-learn.org/stable/modules/generated/sflearn.naive_bayes.ComplementNB.html#sflearn.naive_bayes.ComplementNB)(*[, alpha, ...]) | The Complement Naive Bayes classifier described in Rennie et al. (2003). |
| [`naive_bayes.GaussianNB`](https://scikit-learn.org/stable/modules/generated/sflearn.naive_bayes.GaussianNB.html#sflearn.naive_bayes.GaussianNB)(*[, priors, ...]) | Gaussian Naive Bayes (GaussianNB).                           |
| [`naive_bayes.MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sflearn.naive_bayes.MultinomialNB.html#sflearn.naive_bayes.MultinomialNB)(*[, alpha, ...]) | Naive Bayes classifier for multinomial models.               |

## [`sflearn.neighbors`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.neighbors): Nearest Neighbors

The [`sflearn.neighbors`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.neighbors) module implements the k-nearest neighbors algorithm.

**User guide:** See the [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) section for further details.

| [`neighbors.BallTree`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.BallTree.html#sflearn.neighbors.BallTree)(X[, leaf_size, metric]) | BallTree for fast generalized N-point problems               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.KDTree.html#sflearn.neighbors.KDTree)(X[, leaf_size, metric]) | KDTree for fast generalized N-point problems                 |
| [`neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.KernelDensity.html#sflearn.neighbors.KernelDensity)(*[, bandwidth, ...]) | Kernel Density Estimation.                                   |
| [`neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.KNeighborsClassifier.html#sflearn.neighbors.KNeighborsClassifier)([...]) | Classifier implementing the k-nearest neighbors vote.        |
| [`neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.KNeighborsRegressor.html#sflearn.neighbors.KNeighborsRegressor)([n_neighbors, ...]) | Regression based on k-nearest neighbors.                     |
| [`neighbors.KNeighborsTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.KNeighborsTransformer.html#sflearn.neighbors.KNeighborsTransformer)(*[, mode, ...]) | Transform X into a (weighted) graph of k nearest neighbors.  |
| [`neighbors.LocalOutlierFactor`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.LocalOutlierFactor.html#sflearn.neighbors.LocalOutlierFactor)([n_neighbors, ...]) | Unsupervised Outlier Detection using the Local Outlier Factor (LOF). |
| [`neighbors.RadiusNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.RadiusNeighborsClassifier.html#sflearn.neighbors.RadiusNeighborsClassifier)([...]) | Classifier implementing a vote among neighbors within a given radius. |
| [`neighbors.RadiusNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.RadiusNeighborsRegressor.html#sflearn.neighbors.RadiusNeighborsRegressor)([radius, ...]) | Regression based on neighbors within a fixed radius.         |
| [`neighbors.RadiusNeighborsTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.RadiusNeighborsTransformer.html#sflearn.neighbors.RadiusNeighborsTransformer)(*[, ...]) | Transform X into a (weighted) graph of neighbors nearer than a radius. |
| [`neighbors.NearestCentroid`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.NearestCentroid.html#sflearn.neighbors.NearestCentroid)([metric, ...]) | Nearest centroid classifier.                                 |
| [`neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.NearestNeighbors.html#sflearn.neighbors.NearestNeighbors)(*[, n_neighbors, ...]) | Unsupervised learner for implementing neighbor searches.     |
| [`neighbors.NeighborhoodComponentsAnalysis`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.NeighborhoodComponentsAnalysis.html#sflearn.neighbors.NeighborhoodComponentsAnalysis)([...]) | Neighborhood Components Analysis.                            |

| [`neighbors.kneighbors_graph`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.kneighbors_graph.html#sflearn.neighbors.kneighbors_graph)(X, n_neighbors, *) | Compute the (weighted) graph of k-Neighbors for points in X. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`neighbors.radius_neighbors_graph`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.radius_neighbors_graph.html#sflearn.neighbors.radius_neighbors_graph)(X, radius, *) | Compute the (weighted) graph of Neighbors for points in X.   |
| [`neighbors.sort_graph_by_row_values`](https://scikit-learn.org/stable/modules/generated/sflearn.neighbors.sort_graph_by_row_values.html#sflearn.neighbors.sort_graph_by_row_values)(graph[, ...]) | Sort a sparse graph such that each row is stored with increasing values. |

## [`sflearn.neural_network`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.neural_network): Neural network models

The [`sflearn.neural_network`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.neural_network) module includes models based on neural networks.

**User guide:** See the [Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised) and [Neural network models (unsupervised)](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#neural-networks-unsupervised) sections for further details.

| [`neural_network.BernoulliRBM`](https://scikit-learn.org/stable/modules/generated/sflearn.neural_network.BernoulliRBM.html#sflearn.neural_network.BernoulliRBM)([n_components, ...]) | Bernoulli Restricted Boltzmann Machine (RBM). |
| ------------------------------------------------------------ | --------------------------------------------- |
| [`neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.neural_network.MLPClassifier.html#sflearn.neural_network.MLPClassifier)([...]) | Multi-layer Perceptron classifier.            |
| [`neural_network.MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.neural_network.MLPRegressor.html#sflearn.neural_network.MLPRegressor)([...]) | Multi-layer Perceptron regressor.             |

## [`sflearn.pipeline`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.pipeline): Pipeline

The [`sflearn.pipeline`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.pipeline) module implements utilities to build a composite estimator, as a chain of transforms and estimators.

**User guide:** See the [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) section for further details.

| [`pipeline.FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sflearn.pipeline.FeatureUnion.html#sflearn.pipeline.FeatureUnion)(transformer_list, *[, ...]) | Concatenates results of multiple transformer objects. |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [`pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sflearn.pipeline.Pipeline.html#sflearn.pipeline.Pipeline)(steps, *[, memory, verbose]) | Pipeline of transforms with a final estimator.        |

| [`pipeline.make_pipeline`](https://scikit-learn.org/stable/modules/generated/sflearn.pipeline.make_pipeline.html#sflearn.pipeline.make_pipeline)(*steps[, memory, verbose]) | Construct a `Pipeline` from the given estimators.     |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [`pipeline.make_union`](https://scikit-learn.org/stable/modules/generated/sflearn.pipeline.make_union.html#sflearn.pipeline.make_union)(*transformers[, n_jobs, ...]) | Construct a FeatureUnion from the given transformers. |

## [`sflearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.preprocessing): Preprocessing and Normalization

The [`sflearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.preprocessing) module includes scaling, centering, normalization, binarization methods.

**User guide:** See the [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) section for further details.

| [`preprocessing.Binarizer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.Binarizer.html#sflearn.preprocessing.Binarizer)(*[, threshold, copy]) | Binarize data (set feature values to 0 or 1) according to a threshold. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.FunctionTransformer.html#sflearn.preprocessing.FunctionTransformer)([func, ...]) | Constructs a transformer from an arbitrary callable.         |
| [`preprocessing.KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.KBinsDiscretizer.html#sflearn.preprocessing.KBinsDiscretizer)([n_bins, ...]) | Bin continuous data into intervals.                          |
| [`preprocessing.KernelCenterer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.KernelCenterer.html#sflearn.preprocessing.KernelCenterer)() | Center an arbitrary kernel matrix �.                         |
| [`preprocessing.LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.LabelBinarizer.html#sflearn.preprocessing.LabelBinarizer)(*[, neg_label, ...]) | Binarize labels in a one-vs-all fashion.                     |
| [`preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.LabelEncoder.html#sflearn.preprocessing.LabelEncoder)() | Encode target labels with value between 0 and n_classes-1.   |
| [`preprocessing.MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.MultiLabelBinarizer.html#sflearn.preprocessing.MultiLabelBinarizer)(*[, ...]) | Transform between iterable of iterables and a multilabel format. |
| [`preprocessing.MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.MaxAbsScaler.html#sflearn.preprocessing.MaxAbsScaler)(*[, copy]) | Scale each feature by its maximum absolute value.            |
| [`preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.MinMaxScaler.html#sflearn.preprocessing.MinMaxScaler)([feature_range, ...]) | Transform features by scaling each feature to a given range. |
| [`preprocessing.Normalizer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.Normalizer.html#sflearn.preprocessing.Normalizer)([norm, copy]) | Normalize samples individually to unit norm.                 |
| [`preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.OneHotEncoder.html#sflearn.preprocessing.OneHotEncoder)(*[, categories, ...]) | Encode categorical features as a one-hot numeric array.      |
| [`preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.OrdinalEncoder.html#sflearn.preprocessing.OrdinalEncoder)(*[, ...]) | Encode categorical features as an integer array.             |
| [`preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.PolynomialFeatures.html#sflearn.preprocessing.PolynomialFeatures)([degree, ...]) | Generate polynomial and interaction features.                |
| [`preprocessing.PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.PowerTransformer.html#sflearn.preprocessing.PowerTransformer)([method, ...]) | Apply a power transform featurewise to make data more Gaussian-like. |
| [`preprocessing.QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.QuantileTransformer.html#sflearn.preprocessing.QuantileTransformer)(*[, ...]) | Transform features using quantiles information.              |
| [`preprocessing.RobustScaler`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.RobustScaler.html#sflearn.preprocessing.RobustScaler)(*[, ...]) | Scale features using statistics that are robust to outliers. |
| [`preprocessing.SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.SplineTransformer.html#sflearn.preprocessing.SplineTransformer)([n_knots, ...]) | Generate univariate B-spline bases for features.             |
| [`preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.StandardScaler.html#sflearn.preprocessing.StandardScaler)(*[, copy, ...]) | Standardize features by removing the mean and scaling to unit variance. |

| [`preprocessing.add_dummy_feature`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.add_dummy_feature.html#sflearn.preprocessing.add_dummy_feature)(X[, value]) | Augment dataset with an additional dummy feature.            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`preprocessing.binarize`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.binarize.html#sflearn.preprocessing.binarize)(X, *[, threshold, copy]) | Boolean thresholding of array-like or scipy.sparse matrix.   |
| [`preprocessing.label_binarize`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.label_binarize.html#sflearn.preprocessing.label_binarize)(y, *, classes) | Binarize labels in a one-vs-all fashion.                     |
| [`preprocessing.maxabs_scale`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.maxabs_scale.html#sflearn.preprocessing.maxabs_scale)(X, *[, axis, copy]) | Scale each feature to the [-1, 1] range without breaking the sparsity. |
| [`preprocessing.minmax_scale`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.minmax_scale.html#sflearn.preprocessing.minmax_scale)(X[, ...]) | Transform features by scaling each feature to a given range. |
| [`preprocessing.normalize`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.normalize.html#sflearn.preprocessing.normalize)(X[, norm, axis, ...]) | Scale input vectors individually to unit norm (vector length). |
| [`preprocessing.quantile_transform`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.quantile_transform.html#sflearn.preprocessing.quantile_transform)(X, *[, ...]) | Transform features using quantiles information.              |
| [`preprocessing.robust_scale`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.robust_scale.html#sflearn.preprocessing.robust_scale)(X, *[, axis, ...]) | Standardize a dataset along any axis.                        |
| [`preprocessing.scale`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.scale.html#sflearn.preprocessing.scale)(X, *[, axis, with_mean, ...]) | Standardize a dataset along any axis.                        |
| [`preprocessing.power_transform`](https://scikit-learn.org/stable/modules/generated/sflearn.preprocessing.power_transform.html#sflearn.preprocessing.power_transform)(X[, method, ...]) | Parametric, monotonic transformation to make data more Gaussian-like. |

## [`sflearn.random_projection`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.random_projection): Random projection

Random Projection transformers.

Random Projections are a simple and computationally efficient way to reduce the dimensionality of the data by trading a controlled amount of accuracy (as additional variance) for faster processing times and smaller model sizes.

The dimensions and distribution of Random Projections matrices are controlled so as to preserve the pairwise distances between any two samples of the dataset.

The main theoretical result behind the efficiency of random projection is the [Johnson-Lindenstrauss lemma (quoting Wikipedia)](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma):

> In mathematics, the Johnson-Lindenstrauss lemma is a result concerning low-distortion embeddings of points from high-dimensional into low-dimensional Euclidean space. The lemma states that a small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The map used for the embedding is at least Lipschitz, and can even be taken to be an orthogonal projection.

**User guide:** See the [Random Projection](https://scikit-learn.org/stable/modules/random_projection.html#random-projection) section for further details.

| [`random_projection.GaussianRandomProjection`](https://scikit-learn.org/stable/modules/generated/sflearn.random_projection.GaussianRandomProjection.html#sflearn.random_projection.GaussianRandomProjection)([...]) | Reduce dimensionality through Gaussian random projection. |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`random_projection.SparseRandomProjection`](https://scikit-learn.org/stable/modules/generated/sflearn.random_projection.SparseRandomProjection.html#sflearn.random_projection.SparseRandomProjection)([...]) | Reduce dimensionality through sparse random projection.   |

| [`random_projection.johnson_lindenstrauss_min_dim`](https://scikit-learn.org/stable/modules/generated/sflearn.random_projection.johnson_lindenstrauss_min_dim.html#sflearn.random_projection.johnson_lindenstrauss_min_dim)(...) | Find a 'safe' number of components to randomly project to. |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
|                                                              |                                                            |

## [`sflearn.semi_supervised`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.semi_supervised): Semi-Supervised Learning

The [`sflearn.semi_supervised`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.semi_supervised) module implements semi-supervised learning algorithms. These algorithms utilize small amounts of labeled data and large amounts of unlabeled data for classification tasks. This module includes Label Propagation.

**User guide:** See the [Semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html#semi-supervised) section for further details.

| [`semi_supervised.LabelPropagation`](https://scikit-learn.org/stable/modules/generated/sflearn.semi_supervised.LabelPropagation.html#sflearn.semi_supervised.LabelPropagation)([kernel, ...]) | Label Propagation classifier.                      |
| ------------------------------------------------------------ | -------------------------------------------------- |
| [`semi_supervised.LabelSpreading`](https://scikit-learn.org/stable/modules/generated/sflearn.semi_supervised.LabelSpreading.html#sflearn.semi_supervised.LabelSpreading)([kernel, ...]) | LabelSpreading model for semi-supervised learning. |
| [`semi_supervised.SelfTrainingClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.semi_supervised.SelfTrainingClassifier.html#sflearn.semi_supervised.SelfTrainingClassifier)(...) | Self-training classifier.                          |

## [`sflearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.svm): Support Vector Machines

The [`sflearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.svm) module includes Support Vector Machine algorithms.

**User guide:** See the [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#svm) section for further details.

### Estimators

| [`svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.LinearSVC.html#sflearn.svm.LinearSVC)([penalty, loss, dual, tol, C, ...]) | Linear Support Vector Classification. |
| ------------------------------------------------------------ | ------------------------------------- |
| [`svm.LinearSVR`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.LinearSVR.html#sflearn.svm.LinearSVR)(*[, epsilon, tol, C, loss, ...]) | Linear Support Vector Regression.     |
| [`svm.NuSVC`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.NuSVC.html#sflearn.svm.NuSVC)(*[, nu, kernel, degree, gamma, ...]) | Nu-Support Vector Classification.     |
| [`svm.NuSVR`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.NuSVR.html#sflearn.svm.NuSVR)(*[, nu, C, kernel, degree, gamma, ...]) | Nu Support Vector Regression.         |
| [`svm.OneClassSVM`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.OneClassSVM.html#sflearn.svm.OneClassSVM)(*[, kernel, degree, gamma, ...]) | Unsupervised Outlier Detection.       |
| [`svm.SVC`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.SVC.html#sflearn.svm.SVC)(*[, C, kernel, degree, gamma, ...]) | C-Support Vector Classification.      |
| [`svm.SVR`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.SVR.html#sflearn.svm.SVR)(*[, kernel, degree, gamma, coef0, ...]) | Epsilon-Support Vector Regression.    |

| [`svm.l1_min_c`](https://scikit-learn.org/stable/modules/generated/sflearn.svm.l1_min_c.html#sflearn.svm.l1_min_c)(X, y, *[, loss, fit_intercept, ...]) | Return the lowest bound for C. |
| ------------------------------------------------------------ | ------------------------------ |
|                                                              |                                |

## [`sflearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.tree): Decision Trees

The [`sflearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.tree) module includes decision tree-based models for classification and regression.

**User guide:** See the [Decision Trees](https://scikit-learn.org/stable/modules/tree.html#tree) section for further details.

| [`tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.DecisionTreeClassifier.html#sflearn.tree.DecisionTreeClassifier)(*[, criterion, ...]) | A decision tree classifier.              |
| ------------------------------------------------------------ | ---------------------------------------- |
| [`tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.DecisionTreeRegressor.html#sflearn.tree.DecisionTreeRegressor)(*[, criterion, ...]) | A decision tree regressor.               |
| [`tree.ExtraTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.ExtraTreeClassifier.html#sflearn.tree.ExtraTreeClassifier)(*[, criterion, ...]) | An extremely randomized tree classifier. |
| [`tree.ExtraTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.ExtraTreeRegressor.html#sflearn.tree.ExtraTreeRegressor)(*[, criterion, ...]) | An extremely randomized tree regressor.  |

| [`tree.export_graphviz`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.export_graphviz.html#sflearn.tree.export_graphviz)(decision_tree[, ...]) | Export a decision tree in DOT format.                     |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`tree.export_text`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.export_text.html#sflearn.tree.export_text)(decision_tree, *[, ...]) | Build a text report showing the rules of a decision tree. |

### Plotting

| [`tree.plot_tree`](https://scikit-learn.org/stable/modules/generated/sflearn.tree.plot_tree.html#sflearn.tree.plot_tree)(decision_tree, *[, ...]) | Plot a decision tree. |
| ------------------------------------------------------------ | --------------------- |
|                                                              |                       |

## [`sflearn.utils`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.utils): Utilities

The [`sflearn.utils`](https://scikit-learn.org/stable/modules/classes.html#module-sflearn.utils) module includes various utilities.

**Developer guide:** See the [Utilities for Developers](https://scikit-learn.org/stable/developers/utilities.html#developers-utils) page for further details.

| [`utils.Bunch`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.Bunch.html#sflearn.utils.Bunch)(**kwargs) | Container object exposing keys as attributes. |
| ------------------------------------------------------------ | --------------------------------------------- |
|                                                              |                                               |

| [`utils.arrayfuncs.min_pos`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.arrayfuncs.min_pos.html#sflearn.utils.arrayfuncs.min_pos) | Find the minimum value of an array over positive values      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`utils.as_float_array`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.as_float_array.html#sflearn.utils.as_float_array)(X, *[, copy, ...]) | Convert an array-like to an array of floats.                 |
| [`utils.assert_all_finite`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.assert_all_finite.html#sflearn.utils.assert_all_finite)(X, *[, allow_nan, ...]) | Throw a ValueError if X contains NaN or infinity.            |
| [`utils.check_X_y`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.check_X_y.html#sflearn.utils.check_X_y)(X, y[, accept_sparse, ...]) | Input validation for standard estimators.                    |
| [`utils.check_array`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.check_array.html#sflearn.utils.check_array)(array[, accept_sparse, ...]) | Input validation on an array, list, sparse matrix or similar. |
| [`utils.check_scalar`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.check_scalar.html#sflearn.utils.check_scalar)(x, name, target_type, *) | Validate scalar parameters type and value.                   |
| [`utils.check_consistent_length`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.check_consistent_length.html#sflearn.utils.check_consistent_length)(*arrays) | Check that all arrays have consistent first dimensions.      |
| [`utils.check_random_state`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.check_random_state.html#sflearn.utils.check_random_state)(seed) | Turn seed into a np.random.RandomState instance.             |
| [`utils.class_weight.compute_class_weight`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.class_weight.compute_class_weight.html#sflearn.utils.class_weight.compute_class_weight)(...) | Estimate class weights for unbalanced datasets.              |
| [`utils.class_weight.compute_sample_weight`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.class_weight.compute_sample_weight.html#sflearn.utils.class_weight.compute_sample_weight)(...) | Estimate sample weights by class for unbalanced datasets.    |
| [`utils.deprecated`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.deprecated.html#sflearn.utils.deprecated)([extra]) | Decorator to mark a function or class as deprecated.         |
| [`utils.estimator_checks.check_estimator`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.estimator_checks.check_estimator.html#sflearn.utils.estimator_checks.check_estimator)([...]) | Check if estimator adheres to scikit-learn conventions.      |
| [`utils.estimator_checks.parametrize_with_checks`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.estimator_checks.parametrize_with_checks.html#sflearn.utils.estimator_checks.parametrize_with_checks)(...) | Pytest specific decorator for parametrizing estimator checks. |
| [`utils.estimator_html_repr`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.estimator_html_repr.html#sflearn.utils.estimator_html_repr)(estimator) | Build a HTML representation of an estimator.                 |
| [`utils.extmath.safe_sparse_dot`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.extmath.safe_sparse_dot.html#sflearn.utils.extmath.safe_sparse_dot)(a, b, *[, ...]) | Dot product that handle the sparse matrix case correctly.    |
| [`utils.extmath.randomized_range_finder`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.extmath.randomized_range_finder.html#sflearn.utils.extmath.randomized_range_finder)(A, *, ...) | Compute an orthonormal matrix whose range approximates the range of A. |
| [`utils.extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.extmath.randomized_svd.html#sflearn.utils.extmath.randomized_svd)(M, n_components, *) | Compute a truncated randomized SVD.                          |
| [`utils.extmath.fast_logdet`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.extmath.fast_logdet.html#sflearn.utils.extmath.fast_logdet)(A) | Compute logarithm of determinant of a square matrix.         |
| [`utils.extmath.density`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.extmath.density.html#sflearn.utils.extmath.density)(w, **kwargs) | Compute density of a sparse vector.                          |
| [`utils.extmath.weighted_mode`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.extmath.weighted_mode.html#sflearn.utils.extmath.weighted_mode)(a, w, *[, axis]) | Return an array of the weighted modal (most common) value in the passed array. |
| [`utils.gen_batches`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.gen_batches.html#sflearn.utils.gen_batches)(n, batch_size, *[, ...]) | Generator to create slices containing `batch_size` elements from 0 to `n`. |
| [`utils.gen_even_slices`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.gen_even_slices.html#sflearn.utils.gen_even_slices)(n, n_packs, *[, n_samples]) | Generator to create `n_packs` evenly spaced slices going up to `n`. |
| [`utils.graph.single_source_shortest_path_length`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.graph.single_source_shortest_path_length.html#sflearn.utils.graph.single_source_shortest_path_length)(...) | Return the length of the shortest path from source to all reachable nodes. |
| [`utils.indexable`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.indexable.html#sflearn.utils.indexable)(*iterables) | Make arrays indexable for cross-validation.                  |
| [`utils.metaestimators.available_if`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.metaestimators.available_if.html#sflearn.utils.metaestimators.available_if)(check) | An attribute that is available only if check returns a truthy value. |
| [`utils.multiclass.type_of_target`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.multiclass.type_of_target.html#sflearn.utils.multiclass.type_of_target)(y[, input_name]) | Determine the type of data indicated by the target.          |
| [`utils.multiclass.is_multilabel`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.multiclass.is_multilabel.html#sflearn.utils.multiclass.is_multilabel)(y) | Check if `y` is in a multilabel format.                      |
| [`utils.multiclass.unique_labels`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.multiclass.unique_labels.html#sflearn.utils.multiclass.unique_labels)(*ys) | Extract an ordered array of unique labels.                   |
| [`utils.murmurhash3_32`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.murmurhash3_32.html#sflearn.utils.murmurhash3_32) | Compute the 32bit murmurhash3 of key at seed.                |
| [`utils.resample`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.resample.html#sflearn.utils.resample)(*arrays[, replace, ...]) | Resample arrays or sparse matrices in a consistent way.      |
| [`utils._safe_indexing`](https://scikit-learn.org/stable/modules/generated/sflearn.utils._safe_indexing.html#sflearn.utils._safe_indexing)(X, indices, *[, axis]) | Return rows, items or columns of X using indices.            |
| [`utils.safe_mask`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.safe_mask.html#sflearn.utils.safe_mask)(X, mask) | Return a mask which is safe to use on X.                     |
| [`utils.safe_sqr`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.safe_sqr.html#sflearn.utils.safe_sqr)(X, *[, copy]) | Element wise squaring of array-likes and sparse matrices.    |
| [`utils.shuffle`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.shuffle.html#sflearn.utils.shuffle)(*arrays[, random_state, n_samples]) | Shuffle arrays or sparse matrices in a consistent way.       |
| [`utils.sparsefuncs.incr_mean_variance_axis`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.incr_mean_variance_axis.html#sflearn.utils.sparsefuncs.incr_mean_variance_axis)(X, ...) | Compute incremental mean and variance along an axis on a CSR or CSC matrix. |
| [`utils.sparsefuncs.inplace_column_scale`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.inplace_column_scale.html#sflearn.utils.sparsefuncs.inplace_column_scale)(X, scale) | Inplace column scaling of a CSC/CSR matrix.                  |
| [`utils.sparsefuncs.inplace_row_scale`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.inplace_row_scale.html#sflearn.utils.sparsefuncs.inplace_row_scale)(X, scale) | Inplace row scaling of a CSR or CSC matrix.                  |
| [`utils.sparsefuncs.inplace_swap_row`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.inplace_swap_row.html#sflearn.utils.sparsefuncs.inplace_swap_row)(X, m, n) | Swap two rows of a CSC/CSR matrix in-place.                  |
| [`utils.sparsefuncs.inplace_swap_column`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.inplace_swap_column.html#sflearn.utils.sparsefuncs.inplace_swap_column)(X, m, n) | Swap two columns of a CSC/CSR matrix in-place.               |
| [`utils.sparsefuncs.mean_variance_axis`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.mean_variance_axis.html#sflearn.utils.sparsefuncs.mean_variance_axis)(X, axis) | Compute mean and variance along an axis on a CSR or CSC matrix. |
| [`utils.sparsefuncs.inplace_csr_column_scale`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs.inplace_csr_column_scale.html#sflearn.utils.sparsefuncs.inplace_csr_column_scale)(X, ...) | Inplace column scaling of a CSR matrix.                      |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l1`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1.html#sflearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1) | Inplace row normalize using the l1 norm                      |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l2`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2.html#sflearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2) | Inplace row normalize using the l2 norm                      |
| [`utils.random.sample_without_replacement`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.random.sample_without_replacement.html#sflearn.utils.random.sample_without_replacement) | Sample integers without replacement.                         |
| [`utils.validation.check_is_fitted`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.validation.check_is_fitted.html#sflearn.utils.validation.check_is_fitted)(estimator) | Perform is_fitted validation for estimator.                  |
| [`utils.validation.check_memory`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.validation.check_memory.html#sflearn.utils.validation.check_memory)(memory) | Check that `memory` is joblib.Memory-like.                   |
| [`utils.validation.check_symmetric`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.validation.check_symmetric.html#sflearn.utils.validation.check_symmetric)(array, *[, ...]) | Make sure that array is 2D, square and symmetric.            |
| [`utils.validation.column_or_1d`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.validation.column_or_1d.html#sflearn.utils.validation.column_or_1d)(y, *[, dtype, ...]) | Ravel column or 1d numpy array, else raises an error.        |
| [`utils.validation.has_fit_parameter`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.validation.has_fit_parameter.html#sflearn.utils.validation.has_fit_parameter)(...) | Check whether the estimator's fit method supports the given parameter. |

Specific utilities to list scikit-learn components:

| [`utils.discovery.all_estimators`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.discovery.all_estimators.html#sflearn.utils.discovery.all_estimators)([type_filter]) | Get a list of all estimators from `sflearn`. |
| ------------------------------------------------------------ | -------------------------------------------- |
| [`utils.discovery.all_displays`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.discovery.all_displays.html#sflearn.utils.discovery.all_displays)() | Get a list of all displays from `sflearn`.   |
| [`utils.discovery.all_functions`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.discovery.all_functions.html#sflearn.utils.discovery.all_functions)() | Get a list of all functions from `sflearn`.  |

Utilities from joblib:

| [`utils.parallel.delayed`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.parallel.delayed.html#sflearn.utils.parallel.delayed)(function) | Decorator used to capture the arguments of a function.       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`utils.parallel_backend`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.parallel_backend.html#sflearn.utils.parallel_backend)(backend[, n_jobs, ...]) | Change the default backend used by Parallel inside a with block. |
| [`utils.register_parallel_backend`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.register_parallel_backend.html#sflearn.utils.register_parallel_backend)(name, factory) | Register a new Parallel backend factory.                     |

| [`utils.parallel.Parallel`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.parallel.Parallel.html#sflearn.utils.parallel.Parallel)([n_jobs, backend, ...]) | Tweak of [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#joblib.Parallel) that propagates the scikit-learn configuration. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

## Recently deprecated

### To be removed in 1.3

| [`utils.metaestimators.if_delegate_has_method`](https://scikit-learn.org/stable/modules/generated/sflearn.utils.metaestimators.if_delegate_has_method.html#sflearn.utils.metaestimators.if_delegate_has_method)(...) | Create a decorator for methods that are delegated to a sub-estimator. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

© 2007 - 2023, scikit-learn developers (BSD License). [Show this page source](https://scikit-learn.org/stable/_sources/modules/classes.rst.txt)
