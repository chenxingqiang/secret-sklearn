# API Reference

This is the class and function reference of scikit-learn. Please refer to the [full user guide](https://scikit-learn.org/stable/user_guide.html#user-guide) for further details, as the class and function raw specifications may not be enough to give full guidelines on their uses. For reference on concepts repeated across the API, see [Glossary of Common Terms and API Elements](https://scikit-learn.org/stable/glossary.html#glossary).



## [`sklearn.base`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base): Base classes and utility functions

Base classes for all estimators.

### Base classes

| [`base.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator) | Base class for all estimators in scikit-learn.               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`base.BiclusterMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BiclusterMixin.html#sklearn.base.BiclusterMixin) | Mixin class for all bicluster estimators in scikit-learn.    |
| [`base.ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html#sklearn.base.ClassifierMixin) | Mixin class for all classifiers in scikit-learn.             |
| [`base.ClusterMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html#sklearn.base.ClusterMixin) | Mixin class for all cluster estimators in scikit-learn.      |
| [`base.DensityMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.DensityMixin.html#sklearn.base.DensityMixin) | Mixin class for all density estimators in scikit-learn.      |
| [`base.RegressorMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html#sklearn.base.RegressorMixin) | Mixin class for all regression estimators in scikit-learn.   |
| [`base.TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html#sklearn.base.TransformerMixin) | Mixin class for all transformers in scikit-learn.            |
| [`base.OneToOneFeatureMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.OneToOneFeatureMixin.html#sklearn.base.OneToOneFeatureMixin) | Provides `get_feature_names_out` for simple transformers.    |
| [`base.ClassNamePrefixFeaturesOutMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassNamePrefixFeaturesOutMixin.html#sklearn.base.ClassNamePrefixFeaturesOutMixin) | Mixin class for transformers that generate their own names by prefixing. |
| [`feature_selection.SelectorMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectorMixin.html#sklearn.feature_selection.SelectorMixin) | Transformer mixin that performs feature selection given a support mask |

### Functions

| [`base.clone`](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html#sklearn.base.clone)(estimator, *[, safe]) | Construct a new unfitted estimator with the same parameters. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`base.is_classifier`](https://scikit-learn.org/stable/modules/generated/sklearn.base.is_classifier.html#sklearn.base.is_classifier)(estimator) | Return True if the given estimator is (probably) a classifier. |
| [`base.is_regressor`](https://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html#sklearn.base.is_regressor)(estimator) | Return True if the given estimator is (probably) a regressor. |
| [`config_context`](https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html#sklearn.config_context)(*[, assume_finite, ...]) | Context manager for global scikit-learn configuration.       |
| [`get_config`](https://scikit-learn.org/stable/modules/generated/sklearn.get_config.html#sklearn.get_config)() | Retrieve current values for configuration set by [`set_config`](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config). |
| [`set_config`](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config)([assume_finite, working_memory, ...]) | Set global scikit-learn configuration                        |
| [`show_versions`](https://scikit-learn.org/stable/modules/generated/sklearn.show_versions.html#sklearn.show_versions)() | Print useful debugging information"                          |



## [`sklearn.calibration`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.calibration): Probability Calibration

Calibration of predicted probabilities.

**User guide:** See the [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html#calibration) section for further details.

| [`calibration.CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV)([...]) | Probability calibration with isotonic regression or logistic regression. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

| [`calibration.calibration_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html#sklearn.calibration.calibration_curve)(y_true, y_prob, *) | Compute true and predicted probabilities for a calibration curve. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |



## [`sklearn.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster): Clustering

The [`sklearn.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) module gathers popular unsupervised clustering algorithms.

**User guide:** See the [Clustering](https://scikit-learn.org/stable/modules/clustering.html#clustering) and [Biclustering](https://scikit-learn.org/stable/modules/biclustering.html#biclustering) sections for further details.

### Classes

| [`cluster.AffinityPropagation`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation)(*[, damping, ...]) | Perform Affinity Propagation Clustering of data.             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)([...]) | Agglomerative Clustering.                                    |
| [`cluster.Birch`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch)(*[, threshold, ...]) | Implements the BIRCH clustering algorithm.                   |
| [`cluster.DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)([eps, min_samples, metric, ...]) | Perform DBSCAN clustering from vector array or distance matrix. |
| [`cluster.FeatureAgglomeration`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration)([n_clusters, ...]) | Agglomerate features.                                        |
| [`cluster.KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)([n_clusters, init, n_init, ...]) | K-Means clustering.                                          |
| [`cluster.BisectingKMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans)([n_clusters, init, ...]) | Bisecting K-Means clustering.                                |
| [`cluster.MiniBatchKMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans)([n_clusters, init, ...]) | Mini-Batch K-Means clustering.                               |
| [`cluster.MeanShift`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift)(*[, bandwidth, seeds, ...]) | Mean shift clustering using a flat kernel.                   |
| [`cluster.OPTICS`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS)(*[, min_samples, max_eps, ...]) | Estimate clustering structure from vector array.             |
| [`cluster.SpectralClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering)([n_clusters, ...]) | Apply clustering to a projection of the normalized Laplacian. |
| [`cluster.SpectralBiclustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html#sklearn.cluster.SpectralBiclustering)([n_clusters, ...]) | Spectral biclustering (Kluger, 2003).                        |
| [`cluster.SpectralCoclustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html#sklearn.cluster.SpectralCoclustering)([n_clusters, ...]) | Spectral Co-Clustering algorithm (Dhillon, 2001).            |

### Functions

| [`cluster.affinity_propagation`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.affinity_propagation.html#sklearn.cluster.affinity_propagation)(S, *[, ...]) | Perform Affinity Propagation Clustering of data.             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`cluster.cluster_optics_dbscan`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_dbscan.html#sklearn.cluster.cluster_optics_dbscan)(*, ...) | Perform DBSCAN extraction for an arbitrary epsilon.          |
| [`cluster.cluster_optics_xi`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_xi.html#sklearn.cluster.cluster_optics_xi)(*, reachability, ...) | Automatically extract clusters according to the Xi-steep method. |
| [`cluster.compute_optics_graph`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.compute_optics_graph.html#sklearn.cluster.compute_optics_graph)(X, *, ...) | Compute the OPTICS reachability graph.                       |
| [`cluster.dbscan`](https://scikit-learn.org/stable/modules/generated/dbscan-function.html#sklearn.cluster.dbscan)(X[, eps, min_samples, ...]) | Perform DBSCAN clustering from vector array or distance matrix. |
| [`cluster.estimate_bandwidth`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html#sklearn.cluster.estimate_bandwidth)(X, *[, quantile, ...]) | Estimate the bandwidth to use with the mean-shift algorithm. |
| [`cluster.k_means`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means)(X, n_clusters, *[, ...]) | Perform K-means clustering algorithm.                        |
| [`cluster.kmeans_plusplus`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html#sklearn.cluster.kmeans_plusplus)(X, n_clusters, *[, ...]) | Init n_clusters seeds according to k-means++.                |
| [`cluster.mean_shift`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.mean_shift.html#sklearn.cluster.mean_shift)(X, *[, bandwidth, seeds, ...]) | Perform mean shift clustering of data using a flat kernel.   |
| [`cluster.spectral_clustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html#sklearn.cluster.spectral_clustering)(affinity, *[, ...]) | Apply clustering to a projection of the normalized Laplacian. |
| [`cluster.ward_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.ward_tree.html#sklearn.cluster.ward_tree)(X, *[, connectivity, ...]) | Ward clustering based on a Feature matrix.                   |



## [`sklearn.compose`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose): Composite Estimators

Meta-estimators for building composite models with transformers

In addition to its current contents, this module will eventually be home to refurbished versions of Pipeline and FeatureUnion.

**User guide:** See the [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) section for further details.

| [`compose.ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer)(transformers, *[, ...]) | Applies transformers to columns of an array or pandas DataFrame. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`compose.TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor)([...]) | Meta-estimator to regress on a transformed target.           |

| [`compose.make_column_transformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer)(*transformers) | Construct a ColumnTransformer from the given transformers.   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`compose.make_column_selector`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector)([pattern, ...]) | Create a callable to select columns to be used with `ColumnTransformer`. |



## [`sklearn.covariance`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance): Covariance Estimators

The [`sklearn.covariance`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance) module includes methods and algorithms to robustly estimate the covariance of features given a set of points. The precision matrix defined as the inverse of the covariance is also estimated. Covariance estimation is closely related to the theory of Gaussian Graphical Models.

**User guide:** See the [Covariance estimation](https://scikit-learn.org/stable/modules/covariance.html#covariance) section for further details.

| [`covariance.EmpiricalCovariance`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html#sklearn.covariance.EmpiricalCovariance)(*[, ...]) | Maximum likelihood covariance estimator.                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`covariance.EllipticEnvelope`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html#sklearn.covariance.EllipticEnvelope)(*[, ...]) | An object for detecting outliers in a Gaussian distributed dataset. |
| [`covariance.GraphicalLasso`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html#sklearn.covariance.GraphicalLasso)([alpha, mode, ...]) | Sparse inverse covariance estimation with an l1-penalized estimator. |
| [`covariance.GraphicalLassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLassoCV.html#sklearn.covariance.GraphicalLassoCV)(*[, alphas, ...]) | Sparse inverse covariance w/ cross-validated choice of the l1 penalty. |
| [`covariance.LedoitWolf`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html#sklearn.covariance.LedoitWolf)(*[, store_precision, ...]) | LedoitWolf Estimator.                                        |
| [`covariance.MinCovDet`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html#sklearn.covariance.MinCovDet)(*[, store_precision, ...]) | Minimum Covariance Determinant (MCD): robust estimator of covariance. |
| [`covariance.OAS`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html#sklearn.covariance.OAS)(*[, store_precision, ...]) | Oracle Approximating Shrinkage Estimator.                    |
| [`covariance.ShrunkCovariance`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html#sklearn.covariance.ShrunkCovariance)(*[, ...]) | Covariance estimator with shrinkage.                         |

| [`covariance.empirical_covariance`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html#sklearn.covariance.empirical_covariance)(X, *[, ...]) | Compute the Maximum likelihood covariance estimator.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`covariance.graphical_lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.graphical_lasso.html#sklearn.covariance.graphical_lasso)(emp_cov, alpha, *) | L1-penalized covariance estimator.                           |
| [`covariance.ledoit_wolf`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html#sklearn.covariance.ledoit_wolf)(X, *[, ...]) | Estimate the shrunk Ledoit-Wolf covariance matrix.           |
| [`covariance.oas`](https://scikit-learn.org/stable/modules/generated/oas-function.html#sklearn.covariance.oas)(X, *[, assume_centered]) | Estimate covariance with the Oracle Approximating Shrinkage algorithm. |
| [`covariance.shrunk_covariance`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.shrunk_covariance.html#sklearn.covariance.shrunk_covariance)(emp_cov[, ...]) | Calculate a covariance matrix shrunk on the diagonal.        |



## [`sklearn.cross_decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition): Cross decomposition

**User guide:** See the [Cross decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition) section for further details.

| [`cross_decomposition.CCA`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html#sklearn.cross_decomposition.CCA)([n_components, ...]) | Canonical Correlation Analysis, also known as "Mode B" PLS. |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| [`cross_decomposition.PLSCanonical`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html#sklearn.cross_decomposition.PLSCanonical)([...]) | Partial Least Squares transformer and regressor.            |
| [`cross_decomposition.PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn.cross_decomposition.PLSRegression)([...]) | PLS regression.                                             |
| [`cross_decomposition.PLSSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html#sklearn.cross_decomposition.PLSSVD)([n_components, ...]) | Partial Least Square SVD.                                   |



## [`sklearn.datasets`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets): Datasets

The [`sklearn.datasets`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) module includes utilities to load datasets, including methods to load and fetch popular reference datasets. It also features some artificial data generators.

**User guide:** See the [Dataset loading utilities](https://scikit-learn.org/stable/datasets.html#datasets) section for further details.

### Loaders

| [`datasets.clear_data_home`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.clear_data_home.html#sklearn.datasets.clear_data_home)([data_home]) | Delete all the content of the data home cache.               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`datasets.dump_svmlight_file`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html#sklearn.datasets.dump_svmlight_file)(X, y, f, *[, ...]) | Dump the dataset in svmlight / libsvm file format.           |
| [`datasets.fetch_20newsgroups`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups)(*[, data_home, ...]) | Load the filenames and data from the 20 newsgroups dataset (classification). |
| [`datasets.fetch_20newsgroups_vectorized`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html#sklearn.datasets.fetch_20newsgroups_vectorized)(*[, ...]) | Load and vectorize the 20 newsgroups dataset (classification). |
| [`datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#sklearn.datasets.fetch_california_housing)(*[, ...]) | Load the California housing dataset (regression).            |
| [`datasets.fetch_covtype`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html#sklearn.datasets.fetch_covtype)(*[, data_home, ...]) | Load the covertype dataset (classification).                 |
| [`datasets.fetch_kddcup99`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html#sklearn.datasets.fetch_kddcup99)(*[, subset, ...]) | Load the kddcup99 dataset (classification).                  |
| [`datasets.fetch_lfw_pairs`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html#sklearn.datasets.fetch_lfw_pairs)(*[, subset, ...]) | Load the Labeled Faces in the Wild (LFW) pairs dataset (classification). |
| [`datasets.fetch_lfw_people`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html#sklearn.datasets.fetch_lfw_people)(*[, data_home, ...]) | Load the Labeled Faces in the Wild (LFW) people dataset (classification). |
| [`datasets.fetch_olivetti_faces`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html#sklearn.datasets.fetch_olivetti_faces)(*[, ...]) | Load the Olivetti faces data-set from AT&T (classification). |
| [`datasets.fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html#sklearn.datasets.fetch_openml)([name, version, ...]) | Fetch dataset from openml by name or dataset id.             |
| [`datasets.fetch_rcv1`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html#sklearn.datasets.fetch_rcv1)(*[, data_home, subset, ...]) | Load the RCV1 multilabel dataset (classification).           |
| [`datasets.fetch_species_distributions`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_species_distributions.html#sklearn.datasets.fetch_species_distributions)(*[, ...]) | Loader for species distribution dataset from Phillips et.    |
| [`datasets.get_data_home`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.get_data_home.html#sklearn.datasets.get_data_home)([data_home]) | Return the path of the scikit-learn data directory.          |
| [`datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer)(*[, return_X_y, ...]) | Load and return the breast cancer wisconsin dataset (classification). |
| [`datasets.load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)(*[, return_X_y, ...]) | Load and return the diabetes dataset (regression).           |
| [`datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)(*[, n_class, ...]) | Load and return the digits dataset (classification).         |
| [`datasets.load_files`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html#sklearn.datasets.load_files)(container_path, *[, ...]) | Load text files with categories as subfolder names.          |
| [`datasets.load_iris`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)(*[, return_X_y, as_frame]) | Load and return the iris dataset (classification).           |
| [`datasets.load_linnerud`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud)(*[, return_X_y, as_frame]) | Load and return the physical exercise Linnerud dataset.      |
| [`datasets.load_sample_image`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html#sklearn.datasets.load_sample_image)(image_name) | Load the numpy array of a single sample image.               |
| [`datasets.load_sample_images`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html#sklearn.datasets.load_sample_images)() | Load sample images for image manipulation.                   |
| [`datasets.load_svmlight_file`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html#sklearn.datasets.load_svmlight_file)(f, *[, ...]) | Load datasets in the svmlight / libsvm format into sparse CSR matrix. |
| [`datasets.load_svmlight_files`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_files.html#sklearn.datasets.load_svmlight_files)(files, *[, ...]) | Load dataset from multiple files in SVMlight format.         |
| [`datasets.load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine)(*[, return_X_y, as_frame]) | Load and return the wine dataset (classification).           |

### Samples generator

| [`datasets.make_biclusters`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_biclusters.html#sklearn.datasets.make_biclusters)(shape, n_clusters, *) | Generate a constant block diagonal structure array for biclustering. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`datasets.make_blobs`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs)([n_samples, n_features, ...]) | Generate isotropic Gaussian blobs for clustering.            |
| [`datasets.make_checkerboard`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_checkerboard.html#sklearn.datasets.make_checkerboard)(shape, n_clusters, *) | Generate an array with block checkerboard structure for biclustering. |
| [`datasets.make_circles`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles)([n_samples, shuffle, ...]) | Make a large circle containing a smaller circle in 2d.       |
| [`datasets.make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification)([n_samples, ...]) | Generate a random n-class classification problem.            |
| [`datasets.make_friedman1`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman1)([n_samples, ...]) | Generate the "Friedman #1" regression problem.               |
| [`datasets.make_friedman2`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html#sklearn.datasets.make_friedman2)([n_samples, noise, ...]) | Generate the "Friedman #2" regression problem.               |
| [`datasets.make_friedman3`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman3.html#sklearn.datasets.make_friedman3)([n_samples, noise, ...]) | Generate the "Friedman #3" regression problem.               |
| [`datasets.make_gaussian_quantiles`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles)(*[, mean, ...]) | Generate isotropic Gaussian and label samples by quantile.   |
| [`datasets.make_hastie_10_2`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html#sklearn.datasets.make_hastie_10_2)([n_samples, ...]) | Generate data for binary classification used in Hastie et al. 2009, Example 10.2. |
| [`datasets.make_low_rank_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_low_rank_matrix.html#sklearn.datasets.make_low_rank_matrix)([n_samples, ...]) | Generate a mostly low rank matrix with bell-shaped singular values. |
| [`datasets.make_moons`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons)([n_samples, shuffle, ...]) | Make two interleaving half circles.                          |
| [`datasets.make_multilabel_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_multilabel_classification.html#sklearn.datasets.make_multilabel_classification)([...]) | Generate a random multilabel classification problem.         |
| [`datasets.make_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression)([n_samples, ...]) | Generate a random regression problem.                        |
| [`datasets.make_s_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_s_curve.html#sklearn.datasets.make_s_curve)([n_samples, noise, ...]) | Generate an S curve dataset.                                 |
| [`datasets.make_sparse_coded_signal`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_coded_signal.html#sklearn.datasets.make_sparse_coded_signal)(n_samples, ...) | Generate a signal as a sparse combination of dictionary elements. |
| [`datasets.make_sparse_spd_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html#sklearn.datasets.make_sparse_spd_matrix)([dim, ...]) | Generate a sparse symmetric definite positive matrix.        |
| [`datasets.make_sparse_uncorrelated`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_uncorrelated.html#sklearn.datasets.make_sparse_uncorrelated)([...]) | Generate a random regression problem with sparse uncorrelated design. |
| [`datasets.make_spd_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html#sklearn.datasets.make_spd_matrix)(n_dim, *[, ...]) | Generate a random symmetric, positive-definite matrix.       |
| [`datasets.make_swiss_roll`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html#sklearn.datasets.make_swiss_roll)([n_samples, noise, ...]) | Generate a swiss roll dataset.                               |



## [`sklearn.decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition): Matrix Decomposition

The [`sklearn.decomposition`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) module includes matrix decomposition algorithms, including among others PCA, NMF or ICA. Most of the algorithms of this module can be regarded as dimensionality reduction techniques.

**User guide:** See the [Decomposing signals in components (matrix factorization problems)](https://scikit-learn.org/stable/modules/decomposition.html#decompositions) section for further details.

| [`decomposition.DictionaryLearning`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning)([...]) | Dictionary learning.                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`decomposition.FactorAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis)([n_components, ...]) | Factor Analysis (FA).                                        |
| [`decomposition.FastICA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA)([n_components, ...]) | FastICA: a fast algorithm for Independent Component Analysis. |
| [`decomposition.IncrementalPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA)([n_components, ...]) | Incremental principal components analysis (IPCA).            |
| [`decomposition.KernelPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)([n_components, ...]) | Kernel Principal component analysis (KPCA) [[R396fc7d924b8-1\]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#r396fc7d924b8-1). |
| [`decomposition.LatentDirichletAllocation`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation)([...]) | Latent Dirichlet Allocation with online variational Bayes algorithm. |
| [`decomposition.MiniBatchDictionaryLearning`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html#sklearn.decomposition.MiniBatchDictionaryLearning)([...]) | Mini-batch dictionary learning.                              |
| [`decomposition.MiniBatchSparsePCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html#sklearn.decomposition.MiniBatchSparsePCA)([...]) | Mini-batch Sparse Principal Components Analysis.             |
| [`decomposition.NMF`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF)([n_components, init, ...]) | Non-Negative Matrix Factorization (NMF).                     |
| [`decomposition.MiniBatchNMF`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html#sklearn.decomposition.MiniBatchNMF)([n_components, ...]) | Mini-Batch Non-Negative Matrix Factorization (NMF).          |
| [`decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)([n_components, copy, ...]) | Principal component analysis (PCA).                          |
| [`decomposition.SparsePCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html#sklearn.decomposition.SparsePCA)([n_components, ...]) | Sparse Principal Components Analysis (SparsePCA).            |
| [`decomposition.SparseCoder`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparseCoder.html#sklearn.decomposition.SparseCoder)(dictionary, *[, ...]) | Sparse coding.                                               |
| [`decomposition.TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)([n_components, ...]) | Dimensionality reduction using truncated SVD (aka LSA).      |

| [`decomposition.dict_learning`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning.html#sklearn.decomposition.dict_learning)(X, n_components, ...) | Solve a dictionary learning matrix factorization problem.    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`decomposition.dict_learning_online`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.dict_learning_online.html#sklearn.decomposition.dict_learning_online)(X[, ...]) | Solve a dictionary learning matrix factorization problem online. |
| [`decomposition.fastica`](https://scikit-learn.org/stable/modules/generated/fastica-function.html#sklearn.decomposition.fastica)(X[, n_components, ...]) | Perform Fast Independent Component Analysis.                 |
| [`decomposition.non_negative_factorization`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.non_negative_factorization.html#sklearn.decomposition.non_negative_factorization)(X) | Compute Non-negative Matrix Factorization (NMF).             |
| [`decomposition.sparse_encode`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.sparse_encode.html#sklearn.decomposition.sparse_encode)(X, dictionary, *) | Sparse coding.                                               |



## [`sklearn.discriminant_analysis`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis): Discriminant Analysis

Linear Discriminant Analysis and Quadratic Discriminant Analysis

**User guide:** See the [Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda) section for further details.

| [`discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)([...]) | Linear Discriminant Analysis.    |
| ------------------------------------------------------------ | -------------------------------- |
| [`discriminant_analysis.QuadraticDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)(*) | Quadratic Discriminant Analysis. |



## [`sklearn.dummy`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.dummy): Dummy estimators

**User guide:** See the [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) section for further details.

| [`dummy.DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier)(*[, strategy, ...]) | DummyClassifier makes predictions that ignore the input features. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`dummy.DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor)(*[, strategy, ...]) | Regressor that makes predictions using simple rules.         |



## [`sklearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble): Ensemble Methods

The [`sklearn.ensemble`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) module includes ensemble-based methods for classification, regression and anomaly detection.

**User guide:** See the [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#ensemble) section for further details.

| [`ensemble.AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)([estimator, ...]) | An AdaBoost classifier.                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`ensemble.AdaBoostRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor)([estimator, ...]) | An AdaBoost regressor.                                       |
| [`ensemble.BaggingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier)([estimator, ...]) | A Bagging classifier.                                        |
| [`ensemble.BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor)([estimator, ...]) | A Bagging regressor.                                         |
| [`ensemble.ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier)([...]) | An extra-trees classifier.                                   |
| [`ensemble.ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor)([n_estimators, ...]) | An extra-trees regressor.                                    |
| [`ensemble.GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)(*[, ...]) | Gradient Boosting for classification.                        |
| [`ensemble.GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)(*[, ...]) | Gradient Boosting for regression.                            |
| [`ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest)(*[, n_estimators, ...]) | Isolation Forest Algorithm.                                  |
| [`ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)([...]) | A random forest classifier.                                  |
| [`ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)([...]) | A random forest regressor.                                   |
| [`ensemble.RandomTreesEmbedding`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html#sklearn.ensemble.RandomTreesEmbedding)([...]) | An ensemble of totally random trees.                         |
| [`ensemble.StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier)(estimators[, ...]) | Stack of estimators with a final classifier.                 |
| [`ensemble.StackingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html#sklearn.ensemble.StackingRegressor)(estimators[, ...]) | Stack of estimators with a final regressor.                  |
| [`ensemble.VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier)(estimators, *[, ...]) | Soft Voting/Majority Rule classifier for unfitted estimators. |
| [`ensemble.VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor)(estimators, *[, ...]) | Prediction voting regressor for unfitted estimators.         |
| [`ensemble.HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor)([...]) | Histogram-based Gradient Boosting Regression Tree.           |
| [`ensemble.HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier)([...]) | Histogram-based Gradient Boosting Classification Tree.       |



## [`sklearn.exceptions`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions): Exceptions and warnings

The [`sklearn.exceptions`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.exceptions) module includes all custom warnings and error classes used across scikit-learn.

| [`exceptions.ConvergenceWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ConvergenceWarning.html#sklearn.exceptions.ConvergenceWarning) | Custom warning to capture convergence problems               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`exceptions.DataConversionWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataConversionWarning.html#sklearn.exceptions.DataConversionWarning) | Warning used to notify implicit data conversions happening in the code. |
| [`exceptions.DataDimensionalityWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.DataDimensionalityWarning.html#sklearn.exceptions.DataDimensionalityWarning) | Custom warning to notify potential issues with data dimensionality. |
| [`exceptions.EfficiencyWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.EfficiencyWarning.html#sklearn.exceptions.EfficiencyWarning) | Warning used to notify the user of inefficient computation.  |
| [`exceptions.FitFailedWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.FitFailedWarning.html#sklearn.exceptions.FitFailedWarning) | Warning class used if there is an error while fitting the estimator. |
| [`exceptions.NotFittedError`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html#sklearn.exceptions.NotFittedError) | Exception class to raise if estimator is used before fitting. |
| [`exceptions.UndefinedMetricWarning`](https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html#sklearn.exceptions.UndefinedMetricWarning) | Warning used when the metric is invalid                      |



## [`sklearn.experimental`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental): Experimental

The [`sklearn.experimental`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) module provides importable modules that enable the use of experimental features or estimators.

The features and estimators that are experimental aren’t subject to deprecation cycles. Use them at your own risks!

| [`experimental.enable_hist_gradient_boosting`](https://scikit-learn.org/stable/modules/generated/sklearn.experimental.enable_hist_gradient_boosting.html#module-sklearn.experimental.enable_hist_gradient_boosting) | This is now a no-op and can be safely removed from your code. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`experimental.enable_iterative_imputer`](https://scikit-learn.org/stable/modules/generated/sklearn.experimental.enable_iterative_imputer.html#module-sklearn.experimental.enable_iterative_imputer) | Enables IterativeImputer                                     |
| [`experimental.enable_halving_search_cv`](https://scikit-learn.org/stable/modules/generated/sklearn.experimental.enable_halving_search_cv.html#module-sklearn.experimental.enable_halving_search_cv) | Enables Successive Halving search-estimators                 |



## [`sklearn.feature_extraction`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction): Feature Extraction

The [`sklearn.feature_extraction`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction) module deals with feature extraction from raw data. It currently includes methods to extract features from text and images.

**User guide:** See the [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction) section for further details.

| [`feature_extraction.DictVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer)(*[, ...]) | Transforms lists of feature-value mappings to vectors. |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [`feature_extraction.FeatureHasher`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher)([...]) | Implements feature hashing, aka the hashing trick.     |



### From images

The [`sklearn.feature_extraction.image`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.image) submodule gathers utilities to extract features from images.

| [`feature_extraction.image.extract_patches_2d`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html#sklearn.feature_extraction.image.extract_patches_2d)(...) | Reshape a 2D image into a collection of patches.  |
| ------------------------------------------------------------ | ------------------------------------------------- |
| [`feature_extraction.image.grid_to_graph`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.grid_to_graph.html#sklearn.feature_extraction.image.grid_to_graph)(n_x, n_y) | Graph of the pixel-to-pixel connections.          |
| [`feature_extraction.image.img_to_graph`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.img_to_graph.html#sklearn.feature_extraction.image.img_to_graph)(img, *) | Graph of the pixel-to-pixel gradient connections. |
| [`feature_extraction.image.reconstruct_from_patches_2d`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.reconstruct_from_patches_2d.html#sklearn.feature_extraction.image.reconstruct_from_patches_2d)(...) | Reconstruct the image from all of its patches.    |
| [`feature_extraction.image.PatchExtractor`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html#sklearn.feature_extraction.image.PatchExtractor)(*[, ...]) | Extracts patches from a collection of images.     |



### From text

The [`sklearn.feature_extraction.text`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text) submodule gathers utilities to build feature vectors from text documents.

| [`feature_extraction.text.CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)(*[, ...]) | Convert a collection of text documents to a matrix of token counts. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`feature_extraction.text.HashingVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer)(*) | Convert a collection of text documents to a matrix of token occurrences. |
| [`feature_extraction.text.TfidfTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)(*) | Transform a count matrix to a normalized tf or tf-idf representation. |
| [`feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)(*[, ...]) | Convert a collection of raw documents to a matrix of TF-IDF features. |



## [`sklearn.feature_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection): Feature Selection

The [`sklearn.feature_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection) module implements feature selection algorithms. It currently includes univariate filter selection methods and the recursive feature elimination algorithm.

**User guide:** See the [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection) section for further details.

| [`feature_selection.GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect)([...]) | Univariate feature selector with configurable strategy.      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`feature_selection.SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile)([...]) | Select features according to a percentile of the highest scores. |
| [`feature_selection.SelectKBest`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)([score_func, k]) | Select features according to the k highest scores.           |
| [`feature_selection.SelectFpr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr)([score_func, alpha]) | Filter: Select the pvalues below alpha based on a FPR test.  |
| [`feature_selection.SelectFdr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr)([score_func, alpha]) | Filter: Select the p-values for an estimated false discovery rate. |
| [`feature_selection.SelectFromModel`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel)(estimator, *) | Meta-transformer for selecting features based on importance weights. |
| [`feature_selection.SelectFwe`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe)([score_func, alpha]) | Filter: Select the p-values corresponding to Family-wise error rate. |
| [`feature_selection.SequentialFeatureSelector`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector)(...) | Transformer that performs Sequential Feature Selection.      |
| [`feature_selection.RFE`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)(estimator, *[, ...]) | Feature ranking with recursive feature elimination.          |
| [`feature_selection.RFECV`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)(estimator, *[, ...]) | Recursive feature elimination with cross-validation to select features. |
| [`feature_selection.VarianceThreshold`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold)([threshold]) | Feature selector that removes all low-variance features.     |

| [`feature_selection.chi2`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)(X, y) | Compute chi-squared stats between each non-negative feature and class. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`feature_selection.f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)(X, y) | Compute the ANOVA F-value for the provided sample.           |
| [`feature_selection.f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)(X, y, *[, ...]) | Univariate linear regression tests returning F-statistic and p-values. |
| [`feature_selection.r_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.r_regression.html#sklearn.feature_selection.r_regression)(X, y, *[, ...]) | Compute Pearson's r for each features and the target.        |
| [`feature_selection.mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)(X, y, *) | Estimate mutual information for a discrete target variable.  |
| [`feature_selection.mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression)(X, y, *) | Estimate mutual information for a continuous target variable. |



## [`sklearn.gaussian_process`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process): Gaussian Processes

The [`sklearn.gaussian_process`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process) module implements Gaussian Process based regression and classification.

**User guide:** See the [Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process) section for further details.

| [`gaussian_process.GaussianProcessClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier)([...]) | Gaussian process classification (GPC) based on Laplace approximation. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`gaussian_process.GaussianProcessRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor)([...]) | Gaussian process regression (GPR).                           |

Kernels:

| [`gaussian_process.kernels.CompoundKernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.CompoundKernel.html#sklearn.gaussian_process.kernels.CompoundKernel)(kernels) | Kernel which is composed of a set of other kernels.          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`gaussian_process.kernels.ConstantKernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html#sklearn.gaussian_process.kernels.ConstantKernel)([...]) | Constant kernel.                                             |
| [`gaussian_process.kernels.DotProduct`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.DotProduct.html#sklearn.gaussian_process.kernels.DotProduct)([...]) | Dot-Product kernel.                                          |
| [`gaussian_process.kernels.ExpSineSquared`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html#sklearn.gaussian_process.kernels.ExpSineSquared)([...]) | Exp-Sine-Squared kernel (aka periodic kernel).               |
| [`gaussian_process.kernels.Exponentiation`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Exponentiation.html#sklearn.gaussian_process.kernels.Exponentiation)(...) | The Exponentiation kernel takes one base kernel and a scalar parameter � and combines them via |
| [`gaussian_process.kernels.Hyperparameter`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Hyperparameter.html#sklearn.gaussian_process.kernels.Hyperparameter)(...) | A kernel hyperparameter's specification in form of a namedtuple. |
| [`gaussian_process.kernels.Kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html#sklearn.gaussian_process.kernels.Kernel)() | Base class for all kernels.                                  |
| [`gaussian_process.kernels.Matern`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html#sklearn.gaussian_process.kernels.Matern)([...]) | Matern kernel.                                               |
| [`gaussian_process.kernels.PairwiseKernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html#sklearn.gaussian_process.kernels.PairwiseKernel)([...]) | Wrapper for kernels in sklearn.metrics.pairwise.             |
| [`gaussian_process.kernels.Product`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Product.html#sklearn.gaussian_process.kernels.Product)(k1, k2) | The `Product` kernel takes two kernels �1 and �2 and combines them via |
| [`gaussian_process.kernels.RBF`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF)([length_scale, ...]) | Radial basis function kernel (aka squared-exponential kernel). |
| [`gaussian_process.kernels.RationalQuadratic`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html#sklearn.gaussian_process.kernels.RationalQuadratic)([...]) | Rational Quadratic kernel.                                   |
| [`gaussian_process.kernels.Sum`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Sum.html#sklearn.gaussian_process.kernels.Sum)(k1, k2) | The `Sum` kernel takes two kernels �1 and �2 and combines them via |
| [`gaussian_process.kernels.WhiteKernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html#sklearn.gaussian_process.kernels.WhiteKernel)([...]) | White kernel.                                                |



## [`sklearn.impute`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute): Impute

Transformers for missing value imputation

**User guide:** See the [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html#impute) section for further details.

| [`impute.SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer)(*[, missing_values, ...]) | Univariate imputer for completing missing values with simple strategies. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`impute.IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)([estimator, ...]) | Multivariate imputer that estimates each feature from all the others. |
| [`impute.MissingIndicator`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator)(*[, missing_values, ...]) | Binary indicators for missing values.                        |
| [`impute.KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer)(*[, missing_values, ...]) | Imputation for completing missing values using k-Nearest Neighbors. |



## [`sklearn.inspection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection): Inspection

The [`sklearn.inspection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection) module includes tools for model inspection.

| [`inspection.partial_dependence`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html#sklearn.inspection.partial_dependence)(estimator, X, ...) | Partial dependence of `features`.                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`inspection.permutation_importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance)(estimator, ...) | Permutation importance for feature evaluation [[Rd9e56ef97513-BRE\]](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#rd9e56ef97513-bre). |

### Plotting

| [`inspection.DecisionBoundaryDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html#sklearn.inspection.DecisionBoundaryDisplay)(*, xx0, ...) | Decisions boundary visualization. |
| ------------------------------------------------------------ | --------------------------------- |
| [`inspection.PartialDependenceDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html#sklearn.inspection.PartialDependenceDisplay)(...[, ...]) | Partial Dependence Plot (PDP).    |



## [`sklearn.isotonic`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic): Isotonic regression

**User guide:** See the [Isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html#isotonic) section for further details.

| [`isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression)(*[, y_min, ...]) | Isotonic regression model. |
| ------------------------------------------------------------ | -------------------------- |
|                                                              |                            |

| [`isotonic.check_increasing`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.check_increasing.html#sklearn.isotonic.check_increasing)(x, y) | Determine whether y is monotonically correlated with x. |
| ------------------------------------------------------------ | ------------------------------------------------------- |
| [`isotonic.isotonic_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.isotonic_regression.html#sklearn.isotonic.isotonic_regression)(y, *[, ...]) | Solve the isotonic regression model.                    |



## [`sklearn.kernel_approximation`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation): Kernel Approximation

The [`sklearn.kernel_approximation`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_approximation) module implements several approximate kernel feature maps based on Fourier transforms and Count Sketches.

**User guide:** See the [Kernel Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation) section for further details.

| [`kernel_approximation.AdditiveChi2Sampler`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler)(*) | Approximate feature map for additive chi2 kernel.            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`kernel_approximation.Nystroem`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem)([kernel, ...]) | Approximate a kernel map using a subset of the training data. |
| [`kernel_approximation.PolynomialCountSketch`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.PolynomialCountSketch.html#sklearn.kernel_approximation.PolynomialCountSketch)(*) | Polynomial kernel approximation via Tensor Sketch.           |
| [`kernel_approximation.RBFSampler`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler)(*[, gamma, ...]) | Approximate a RBF kernel feature map using random Fourier features. |
| [`kernel_approximation.SkewedChi2Sampler`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html#sklearn.kernel_approximation.SkewedChi2Sampler)(*[, ...]) | Approximate feature map for "skewed chi-squared" kernel.     |



## [`sklearn.kernel_ridge`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge): Kernel Ridge Regression

Module [`sklearn.kernel_ridge`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.kernel_ridge) implements kernel ridge regression.

**User guide:** See the [Kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge) section for further details.

| [`kernel_ridge.KernelRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge)([alpha, kernel, ...]) | Kernel ridge regression. |
| ------------------------------------------------------------ | ------------------------ |
|                                                              |                          |



## [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model): Linear Models

The [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) module implements a variety of linear models.

**User guide:** See the [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#linear-model) section for further details.

The following subsections are only rough guidelines: the same estimator can fall into multiple categories, depending on its parameters.

### Linear classifiers

| [`linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)([penalty, ...]) | Logistic Regression (aka logit, MaxEnt) classifier.          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV)(*[, Cs, ...]) | Logistic Regression CV (aka logit, MaxEnt) classifier.       |
| [`linear_model.PassiveAggressiveClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier)(*) | Passive Aggressive Classifier.                               |
| [`linear_model.Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)(*[, penalty, alpha, ...]) | Linear perceptron classifier.                                |
| [`linear_model.RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier)([alpha, ...]) | Classifier using Ridge regression.                           |
| [`linear_model.RidgeClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV)([alphas, ...]) | Ridge classifier with built-in cross-validation.             |
| [`linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)([loss, penalty, ...]) | Linear classifiers (SVM, logistic regression, etc.) with SGD training. |
| [`linear_model.SGDOneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html#sklearn.linear_model.SGDOneClassSVM)([nu, ...]) | Solves linear One-Class SVM using Stochastic Gradient Descent. |

### Classical linear regressors

| [`linear_model.LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)(*[, ...]) | Ordinary least squares Linear Regression.                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)([alpha, fit_intercept, ...]) | Linear least squares with l2 regularization.                 |
| [`linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV)([alphas, ...]) | Ridge regression with built-in cross-validation.             |
| [`linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)([loss, penalty, ...]) | Linear model fitted by minimizing a regularized empirical loss with SGD. |

### Regressors with variable selection

The following estimators have built-in variable selection fitting procedures, but any estimator using a L1 or elastic-net penalty also performs variable selection: typically [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) or [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) with an appropriate penalty.

| [`linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)([alpha, l1_ratio, ...]) | Linear regression with combined L1 and L2 priors as regularizer. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.ElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV)(*[, l1_ratio, ...]) | Elastic Net model with iterative fitting along a regularization path. |
| [`linear_model.Lars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars)(*[, fit_intercept, ...]) | Least Angle Regression model a.k.a.                          |
| [`linear_model.LarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html#sklearn.linear_model.LarsCV)(*[, fit_intercept, ...]) | Cross-validated Least Angle Regression model.                |
| [`linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)([alpha, fit_intercept, ...]) | Linear Model trained with L1 prior as regularizer (aka the Lasso). |
| [`linear_model.LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV)(*[, eps, n_alphas, ...]) | Lasso linear model with iterative fitting along a regularization path. |
| [`linear_model.LassoLars`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars)([alpha, ...]) | Lasso model fit with Least Angle Regression a.k.a.           |
| [`linear_model.LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV)(*[, fit_intercept, ...]) | Cross-validated Lasso, using the LARS algorithm.             |
| [`linear_model.LassoLarsIC`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC)([criterion, ...]) | Lasso model fit with Lars using BIC or AIC for model selection. |
| [`linear_model.OrthogonalMatchingPursuit`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit)(*[, ...]) | Orthogonal Matching Pursuit model (OMP).                     |
| [`linear_model.OrthogonalMatchingPursuitCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV)(*) | Cross-validated Orthogonal Matching Pursuit model (OMP).     |

### Bayesian regressors

| [`linear_model.ARDRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression)(*[, n_iter, tol, ...]) | Bayesian ARD regression.   |
| ------------------------------------------------------------ | -------------------------- |
| [`linear_model.BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge)(*[, n_iter, tol, ...]) | Bayesian ridge regression. |

### Multi-task linear regressors with variable selection

These estimators fit multiple regression problems (or tasks) jointly, while inducing sparse coefficients. While the inferred coefficients may differ between the tasks, they are constrained to agree on the features that are selected (non-zero coefficients).

| [`linear_model.MultiTaskElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet)([alpha, ...]) | Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.MultiTaskElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV)(*[, ...]) | Multi-task L1/L2 ElasticNet with built-in cross-validation.  |
| [`linear_model.MultiTaskLasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso)([alpha, ...]) | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer. |
| [`linear_model.MultiTaskLassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV)(*[, eps, ...]) | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer. |

### Outlier-robust regressors

Any estimator using the Huber loss would also be robust to outliers, e.g. [`SGDRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor) with `loss='huber'`.

| [`linear_model.HuberRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor)(*[, epsilon, ...]) | L2-regularized linear regression model that is robust to outliers. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html#sklearn.linear_model.QuantileRegressor)(*[, ...]) | Linear regression model that predicts conditional quantiles. |
| [`linear_model.RANSACRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor)([estimator, ...]) | RANSAC (RANdom SAmple Consensus) algorithm.                  |
| [`linear_model.TheilSenRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor)(*[, ...]) | Theil-Sen Estimator: robust multivariate regression model.   |

### Generalized linear models (GLM) for regression

These models allow for response variables to have error distributions other than a normal distribution:

| [`linear_model.PoissonRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html#sklearn.linear_model.PoissonRegressor)(*[, alpha, ...]) | Generalized Linear Model with a Poisson distribution. |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [`linear_model.TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor)(*[, power, ...]) | Generalized Linear Model with a Tweedie distribution. |
| [`linear_model.GammaRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html#sklearn.linear_model.GammaRegressor)(*[, alpha, ...]) | Generalized Linear Model with a Gamma distribution.   |

### Miscellaneous

| [`linear_model.PassiveAggressiveRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor)(*[, ...]) | Passive Aggressive Regressor.                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`linear_model.enet_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.enet_path.html#sklearn.linear_model.enet_path)(X, y, *[, l1_ratio, ...]) | Compute elastic net path with coordinate descent.            |
| [`linear_model.lars_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html#sklearn.linear_model.lars_path)(X, y[, Xy, Gram, ...]) | Compute Least Angle Regression or Lasso path using the LARS algorithm [1]. |
| [`linear_model.lars_path_gram`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path_gram.html#sklearn.linear_model.lars_path_gram)(Xy, Gram, *, ...) | The lars_path in the sufficient stats mode [1].              |
| [`linear_model.lasso_path`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html#sklearn.linear_model.lasso_path)(X, y, *[, eps, ...]) | Compute Lasso path with coordinate descent.                  |
| [`linear_model.orthogonal_mp`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html#sklearn.linear_model.orthogonal_mp)(X, y, *[, ...]) | Orthogonal Matching Pursuit (OMP).                           |
| [`linear_model.orthogonal_mp_gram`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp_gram.html#sklearn.linear_model.orthogonal_mp_gram)(Gram, Xy, *) | Gram Orthogonal Matching Pursuit (OMP).                      |
| [`linear_model.ridge_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html#sklearn.linear_model.ridge_regression)(X, y, alpha, *) | Solve the ridge equation by the method of normal equations.  |



## [`sklearn.manifold`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold): Manifold Learning

The [`sklearn.manifold`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold) module implements data embedding techniques.

**User guide:** See the [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html#manifold) section for further details.

| [`manifold.Isomap`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap)(*[, n_neighbors, radius, ...]) | Isomap Embedding.                                           |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| [`manifold.LocallyLinearEmbedding`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding)(*[, ...]) | Locally Linear Embedding.                                   |
| [`manifold.MDS`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS)([n_components, metric, n_init, ...]) | Multidimensional scaling.                                   |
| [`manifold.SpectralEmbedding`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html#sklearn.manifold.SpectralEmbedding)([n_components, ...]) | Spectral embedding for non-linear dimensionality reduction. |
| [`manifold.TSNE`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)([n_components, perplexity, ...]) | T-distributed Stochastic Neighbor Embedding.                |

| [`manifold.locally_linear_embedding`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.locally_linear_embedding.html#sklearn.manifold.locally_linear_embedding)(X, *, ...) | Perform a Locally Linear Embedding analysis on the data.     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`manifold.smacof`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html#sklearn.manifold.smacof)(dissimilarities, *[, ...]) | Compute multidimensional scaling using the SMACOF algorithm. |
| [`manifold.spectral_embedding`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.spectral_embedding.html#sklearn.manifold.spectral_embedding)(adjacency, *[, ...]) | Project the sample on the first eigenvectors of the graph Laplacian. |
| [`manifold.trustworthiness`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html#sklearn.manifold.trustworthiness)(X, X_embedded, *[, ...]) | Indicate to what extent the local structure is retained.     |



## [`sklearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics): Metrics

See the [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation) section and the [Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics) section of the user guide for further details.



The [`sklearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) module includes score functions, performance metrics and pairwise metrics and distance computations.

### Model Selection Interface

See the [The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) section of the user guide for further details.

| [`metrics.check_scoring`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.check_scoring.html#sklearn.metrics.check_scoring)(estimator[, scoring, ...]) | Determine scorer from user options.                       |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`metrics.get_scorer`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer)(scoring) | Get a scorer from string.                                 |
| [`metrics.get_scorer_names`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer_names.html#sklearn.metrics.get_scorer_names)() | Get the names of all available scorers.                   |
| [`metrics.make_scorer`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer)(score_func, *[, ...]) | Make a scorer from a performance metric or loss function. |

### Classification metrics

See the [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) section of the user guide for further details.

| [`metrics.accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)(y_true, y_pred, *[, ...]) | Accuracy classification score.                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.auc`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)(x, y) | Compute Area Under the Curve (AUC) using the trapezoidal rule. |
| [`metrics.average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)(y_true, ...) | Compute average precision (AP) from prediction scores.       |
| [`metrics.balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)(y_true, ...) | Compute the balanced accuracy.                               |
| [`metrics.brier_score_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss)(y_true, y_prob, *) | Compute the Brier score loss.                                |
| [`metrics.class_likelihood_ratios`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.class_likelihood_ratios.html#sklearn.metrics.class_likelihood_ratios)(y_true, ...) | Compute binary classification positive and negative likelihood ratios. |
| [`metrics.classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)(y_true, y_pred, *) | Build a text report showing the main classification metrics. |
| [`metrics.cohen_kappa_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score)(y1, y2, *[, ...]) | Compute Cohen's kappa: a statistic that measures inter-annotator agreement. |
| [`metrics.confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)(y_true, y_pred, *) | Compute confusion matrix to evaluate the accuracy of a classification. |
| [`metrics.dcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.dcg_score.html#sklearn.metrics.dcg_score)(y_true, y_score, *[, k, ...]) | Compute Discounted Cumulative Gain.                          |
| [`metrics.det_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.det_curve.html#sklearn.metrics.det_curve)(y_true, y_score[, ...]) | Compute error rates for different probability thresholds.    |
| [`metrics.f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)(y_true, y_pred, *[, ...]) | Compute the F1 score, also known as balanced F-score or F-measure. |
| [`metrics.fbeta_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score)(y_true, y_pred, *, beta) | Compute the F-beta score.                                    |
| [`metrics.hamming_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss)(y_true, y_pred, *[, ...]) | Compute the average Hamming loss.                            |
| [`metrics.hinge_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss)(y_true, pred_decision, *) | Average hinge loss (non-regularized).                        |
| [`metrics.jaccard_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score)(y_true, y_pred, *[, ...]) | Jaccard similarity coefficient score.                        |
| [`metrics.log_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss)(y_true, y_pred, *[, eps, ...]) | Log loss, aka logistic loss or cross-entropy loss.           |
| [`metrics.matthews_corrcoef`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef)(y_true, y_pred, *) | Compute the Matthews correlation coefficient (MCC).          |
| [`metrics.multilabel_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html#sklearn.metrics.multilabel_confusion_matrix)(y_true, ...) | Compute a confusion matrix for each class or sample.         |
| [`metrics.ndcg_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html#sklearn.metrics.ndcg_score)(y_true, y_score, *[, k, ...]) | Compute Normalized Discounted Cumulative Gain.               |
| [`metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)(y_true, ...) | Compute precision-recall pairs for different probability thresholds. |
| [`metrics.precision_recall_fscore_support`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)(...) | Compute precision, recall, F-measure and support for each class. |
| [`metrics.precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)(y_true, y_pred, *[, ...]) | Compute the precision.                                       |
| [`metrics.recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)(y_true, y_pred, *[, ...]) | Compute the recall.                                          |
| [`metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)(y_true, y_score, *[, ...]) | Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)     from prediction scores. |
| [`metrics.roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)(y_true, y_score, *[, ...]) | Compute Receiver operating characteristic (ROC).             |
| [`metrics.top_k_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html#sklearn.metrics.top_k_accuracy_score)(y_true, y_score, *) | Top-k Accuracy classification score.                         |
| [`metrics.zero_one_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html#sklearn.metrics.zero_one_loss)(y_true, y_pred, *[, ...]) | Zero-one classification loss.                                |

### Regression metrics

See the [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) section of the user guide for further details.

| [`metrics.explained_variance_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score)(y_true, ...) | Explained variance regression score function.                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.max_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error)(y_true, y_pred) | The max_error metric calculates the maximum residual error.  |
| [`metrics.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)(y_true, y_pred, *) | Mean absolute error regression loss.                         |
| [`metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)(y_true, y_pred, *) | Mean squared error regression loss.                          |
| [`metrics.mean_squared_log_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html#sklearn.metrics.mean_squared_log_error)(y_true, y_pred, *) | Mean squared logarithmic error regression loss.              |
| [`metrics.median_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html#sklearn.metrics.median_absolute_error)(y_true, y_pred, *) | Median absolute error regression loss.                       |
| [`metrics.mean_absolute_percentage_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html#sklearn.metrics.mean_absolute_percentage_error)(...) | Mean absolute percentage error (MAPE) regression loss.       |
| [`metrics.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)(y_true, y_pred, *[, ...]) | �2 (coefficient of determination) regression score function. |
| [`metrics.mean_poisson_deviance`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html#sklearn.metrics.mean_poisson_deviance)(y_true, y_pred, *) | Mean Poisson deviance regression loss.                       |
| [`metrics.mean_gamma_deviance`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html#sklearn.metrics.mean_gamma_deviance)(y_true, y_pred, *) | Mean Gamma deviance regression loss.                         |
| [`metrics.mean_tweedie_deviance`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html#sklearn.metrics.mean_tweedie_deviance)(y_true, y_pred, *) | Mean Tweedie deviance regression loss.                       |
| [`metrics.d2_tweedie_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_tweedie_score.html#sklearn.metrics.d2_tweedie_score)(y_true, y_pred, *) | D^2 regression score function, fraction of Tweedie deviance explained. |
| [`metrics.mean_pinball_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html#sklearn.metrics.mean_pinball_loss)(y_true, y_pred, *) | Pinball loss for quantile regression.                        |
| [`metrics.d2_pinball_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_pinball_score.html#sklearn.metrics.d2_pinball_score)(y_true, y_pred, *) | �2 regression score function, fraction of pinball loss explained. |
| [`metrics.d2_absolute_error_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_absolute_error_score.html#sklearn.metrics.d2_absolute_error_score)(y_true, ...) | �2 regression score function,     fraction of absolute error explained. |

### Multilabel ranking metrics

See the [Multilabel ranking metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics) section of the user guide for further details.

| [`metrics.coverage_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.coverage_error.html#sklearn.metrics.coverage_error)(y_true, y_score, *[, ...]) | Coverage error measure.                  |
| ------------------------------------------------------------ | ---------------------------------------- |
| [`metrics.label_ranking_average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_average_precision_score.html#sklearn.metrics.label_ranking_average_precision_score)(...) | Compute ranking-based average precision. |
| [`metrics.label_ranking_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html#sklearn.metrics.label_ranking_loss)(y_true, y_score, *) | Compute Ranking loss measure.            |

### Clustering metrics

See the [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation) section of the user guide for further details.



The [`sklearn.metrics.cluster`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster) submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:

- supervised, which uses a ground truth class values for each sample.
- unsupervised, which does not and measures the ‘quality’ of the model itself.

| [`metrics.adjusted_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score)(...[, ...]) | Adjusted Mutual Information between two clusterings.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score)(labels_true, ...) | Rand index adjusted for chance.                              |
| [`metrics.calinski_harabasz_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score)(X, labels) | Compute the Calinski and Harabasz score.                     |
| [`metrics.davies_bouldin_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score)(X, labels) | Compute the Davies-Bouldin score.                            |
| [`metrics.completeness_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score)(labels_true, ...) | Compute completeness metric of a cluster labeling given a ground truth. |
| [`metrics.cluster.contingency_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.contingency_matrix.html#sklearn.metrics.cluster.contingency_matrix)(...[, ...]) | Build a contingency matrix describing the relationship between labels. |
| [`metrics.cluster.pair_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html#sklearn.metrics.cluster.pair_confusion_matrix)(...) | Pair confusion matrix arising from two clusterings [[R9ca8fd06d29a-1\]](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cluster.pair_confusion_matrix.html#r9ca8fd06d29a-1). |
| [`metrics.fowlkes_mallows_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score)(labels_true, ...) | Measure the similarity of two clusterings of a set of points. |
| [`metrics.homogeneity_completeness_v_measure`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html#sklearn.metrics.homogeneity_completeness_v_measure)(...) | Compute the homogeneity and completeness and V-Measure scores at once. |
| [`metrics.homogeneity_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score)(labels_true, ...) | Homogeneity metric of a cluster labeling given a ground truth. |
| [`metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score)(labels_true, ...) | Mutual Information between two clusterings.                  |
| [`metrics.normalized_mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score)(...[, ...]) | Normalized Mutual Information between two clusterings.       |
| [`metrics.rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score)(labels_true, labels_pred) | Rand index.                                                  |
| [`metrics.silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)(X, labels, *[, ...]) | Compute the mean Silhouette Coefficient of all samples.      |
| [`metrics.silhouette_samples`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples)(X, labels, *[, ...]) | Compute the Silhouette Coefficient for each sample.          |
| [`metrics.v_measure_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score)(labels_true, ...[, beta]) | V-measure cluster labeling given a ground truth.             |

### Biclustering metrics

See the [Biclustering evaluation](https://scikit-learn.org/stable/modules/biclustering.html#biclustering-evaluation) section of the user guide for further details.

| [`metrics.consensus_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.consensus_score.html#sklearn.metrics.consensus_score)(a, b, *[, similarity]) | The similarity of two sets of biclusters. |
| ------------------------------------------------------------ | ----------------------------------------- |
|                                                              |                                           |

### Distance metrics

| [`metrics.DistanceMetric`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html#sklearn.metrics.DistanceMetric) | DistanceMetric class |
| ------------------------------------------------------------ | -------------------- |
|                                                              |                      |

### Pairwise metrics

See the [Pairwise metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html#metrics) section of the user guide for further details.



| [`metrics.pairwise.additive_chi2_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.additive_chi2_kernel.html#sklearn.metrics.pairwise.additive_chi2_kernel)(X[, Y]) | Compute the additive chi-squared kernel between observations in X and Y. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.pairwise.chi2_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.chi2_kernel.html#sklearn.metrics.pairwise.chi2_kernel)(X[, Y, gamma]) | Compute the exponential chi-squared kernel between X and Y.  |
| [`metrics.pairwise.cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity)(X[, Y, ...]) | Compute cosine similarity between samples in X and Y.        |
| [`metrics.pairwise.cosine_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html#sklearn.metrics.pairwise.cosine_distances)(X[, Y]) | Compute cosine distance between samples in X and Y.          |
| [`metrics.pairwise.distance_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics)() | Valid metrics for pairwise_distances.                        |
| [`metrics.pairwise.euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances)(X[, Y, ...]) | Compute the distance matrix between each pair from a vector array X and Y. |
| [`metrics.pairwise.haversine_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html#sklearn.metrics.pairwise.haversine_distances)(X[, Y]) | Compute the Haversine distance between samples in X and Y.   |
| [`metrics.pairwise.kernel_metrics`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics)() | Valid metrics for pairwise_kernels.                          |
| [`metrics.pairwise.laplacian_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html#sklearn.metrics.pairwise.laplacian_kernel)(X[, Y, gamma]) | Compute the laplacian kernel between X and Y.                |
| [`metrics.pairwise.linear_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html#sklearn.metrics.pairwise.linear_kernel)(X[, Y, ...]) | Compute the linear kernel between X and Y.                   |
| [`metrics.pairwise.manhattan_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html#sklearn.metrics.pairwise.manhattan_distances)(X[, Y, ...]) | Compute the L1 distances between the vectors in X and Y.     |
| [`metrics.pairwise.nan_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html#sklearn.metrics.pairwise.nan_euclidean_distances)(X) | Calculate the euclidean distances in the presence of missing values. |
| [`metrics.pairwise.pairwise_kernels`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html#sklearn.metrics.pairwise.pairwise_kernels)(X[, Y, ...]) | Compute the kernel between arrays X and optional array Y.    |
| [`metrics.pairwise.polynomial_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html#sklearn.metrics.pairwise.polynomial_kernel)(X[, Y, ...]) | Compute the polynomial kernel between X and Y.               |
| [`metrics.pairwise.rbf_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel)(X[, Y, gamma]) | Compute the rbf (gaussian) kernel between X and Y.           |
| [`metrics.pairwise.sigmoid_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html#sklearn.metrics.pairwise.sigmoid_kernel)(X[, Y, ...]) | Compute the sigmoid kernel between X and Y.                  |
| [`metrics.pairwise.paired_euclidean_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_euclidean_distances.html#sklearn.metrics.pairwise.paired_euclidean_distances)(X, Y) | Compute the paired euclidean distances between X and Y.      |
| [`metrics.pairwise.paired_manhattan_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_manhattan_distances.html#sklearn.metrics.pairwise.paired_manhattan_distances)(X, Y) | Compute the paired L1 distances between X and Y.             |
| [`metrics.pairwise.paired_cosine_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_cosine_distances.html#sklearn.metrics.pairwise.paired_cosine_distances)(X, Y) | Compute the paired cosine distances between X and Y.         |
| [`metrics.pairwise.paired_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.paired_distances.html#sklearn.metrics.pairwise.paired_distances)(X, Y, *[, ...]) | Compute the paired distances between X and Y.                |
| [`metrics.pairwise_distances`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances)(X[, Y, metric, ...]) | Compute the distance matrix from a vector array X and optional Y. |
| [`metrics.pairwise_distances_argmin`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin.html#sklearn.metrics.pairwise_distances_argmin)(X, Y, *[, ...]) | Compute minimum distances between one point and a set of points. |
| [`metrics.pairwise_distances_argmin_min`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html#sklearn.metrics.pairwise_distances_argmin_min)(X, Y, *) | Compute minimum distances between one point and a set of points. |
| [`metrics.pairwise_distances_chunked`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html#sklearn.metrics.pairwise_distances_chunked)(X[, Y, ...]) | Generate a distance matrix chunk by chunk with optional reduction. |

### Plotting

See the [Visualizations](https://scikit-learn.org/stable/visualizations.html#visualizations) section of the user guide for further details.

| [`metrics.ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay)(...[, ...]) | Confusion Matrix visualization.                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`metrics.DetCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DetCurveDisplay.html#sklearn.metrics.DetCurveDisplay)(*, fpr, fnr[, ...]) | DET curve visualization.                                     |
| [`metrics.PrecisionRecallDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay)(precision, ...) | Precision Recall visualization.                              |
| [`metrics.PredictionErrorDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html#sklearn.metrics.PredictionErrorDisplay)(*, y_true, y_pred) | Visualization of the prediction error of a regression model. |
| [`metrics.RocCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html#sklearn.metrics.RocCurveDisplay)(*, fpr, tpr[, ...]) | ROC Curve visualization.                                     |
| [`calibration.CalibrationDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html#sklearn.calibration.CalibrationDisplay)(prob_true, ...) | Calibration curve (also known as reliability diagram) visualization. |



## [`sklearn.mixture`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture): Gaussian Mixture Models

The [`sklearn.mixture`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture) module implements mixture modeling algorithms.

**User guide:** See the [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html#mixture) section for further details.

| [`mixture.BayesianGaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture)(*[, ...]) | Variational Bayesian estimation of a Gaussian mixture. |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| [`mixture.GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)([n_components, ...]) | Gaussian Mixture.                                      |



## [`sklearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection): Model Selection

**User guide:** See the [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation), [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) and [Learning curve](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve) sections for further details.

### Splitter Classes

| [`model_selection.GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold)([n_splits]) | K-fold iterator variant with non-overlapping groups.         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit)([...]) | Shuffle-Group(s)-Out cross-validation iterator               |
| [`model_selection.KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)([n_splits, shuffle, ...]) | K-Folds cross-validator                                      |
| [`model_selection.LeaveOneGroupOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut)() | Leave One Group Out cross-validator                          |
| [`model_selection.LeavePGroupsOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut)(n_groups) | Leave P Group(s) Out cross-validator                         |
| [`model_selection.LeaveOneOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut)() | Leave-One-Out cross-validator                                |
| [`model_selection.LeavePOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html#sklearn.model_selection.LeavePOut)(p) | Leave-P-Out cross-validator                                  |
| [`model_selection.PredefinedSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html#sklearn.model_selection.PredefinedSplit)(test_fold) | Predefined split cross-validator                             |
| [`model_selection.RepeatedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold)(*[, n_splits, ...]) | Repeated K-Fold cross validator.                             |
| [`model_selection.RepeatedStratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold)(*[, ...]) | Repeated Stratified K-Fold cross validator.                  |
| [`model_selection.ShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit)([n_splits, ...]) | Random permutation cross-validator                           |
| [`model_selection.StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)([n_splits, ...]) | Stratified K-Folds cross-validator.                          |
| [`model_selection.StratifiedShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit)([...]) | Stratified ShuffleSplit cross-validator                      |
| [`model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html#sklearn.model_selection.StratifiedGroupKFold)([...]) | Stratified K-Folds iterator variant with non-overlapping groups. |
| [`model_selection.TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit)([n_splits, ...]) | Time Series cross-validator                                  |

### Splitter Functions

| [`model_selection.check_cv`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html#sklearn.model_selection.check_cv)([cv, y, classifier]) | Input checker utility for building a cross-validator.        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)(*arrays[, ...]) | Split arrays or matrices into random train and test subsets. |



### Hyper-parameter optimizers

| [`model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)(estimator, ...) | Exhaustive search over specified parameter values for an estimator. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.HalvingGridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV)(...[, ...]) | Search over specified parameter values with successive halving. |
| [`model_selection.ParameterGrid`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid)(param_grid) | Grid of parameters with a discrete number of values for each. |
| [`model_selection.ParameterSampler`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler)(...[, ...]) | Generator on parameters sampled from given distributions.    |
| [`model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |
| [`model_selection.HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV)(...[, ...]) | Randomized search on hyper parameters.                       |

### Model validation

| [`model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)(estimator, X) | Evaluate metric(s) by cross-validation and also record fit/score times. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`model_selection.cross_val_predict`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict)(estimator, X) | Generate cross-validated estimates for each input data point. |
| [`model_selection.cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)(estimator, X) | Evaluate a score by cross-validation.                        |
| [`model_selection.learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve)(estimator, X, ...) | Learning curve.                                              |
| [`model_selection.permutation_test_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html#sklearn.model_selection.permutation_test_score)(...) | Evaluate the significance of a cross-validated score with permutations. |
| [`model_selection.validation_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve)(estimator, ...) | Validation curve.                                            |

### Visualization

| [`model_selection.LearningCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LearningCurveDisplay.html#sklearn.model_selection.LearningCurveDisplay)(*, ...) | Learning Curve visualization. |
| ------------------------------------------------------------ | ----------------------------- |
|                                                              |                               |



## [`sklearn.multiclass`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multiclass): Multiclass classification

### Multiclass classification strategies

- This module implements multiclass learning algorithms:

  one-vs-the-rest / one-vs-allone-vs-oneerror correcting output codes

The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. For example, it is possible to use these estimators to turn a binary classifier or a regressor into a multiclass classifier. It is also possible to use these estimators with multiclass estimators in the hope that their accuracy or runtime performance improves.

All classifiers in scikit-learn implement multiclass classification; you only need to use this module if you want to experiment with custom multiclass strategies.

The one-vs-the-rest meta-classifier also implements a `predict_proba` method, so long as such a method is implemented by the base classifier. This method returns probabilities of class membership in both the single label and multilabel case. Note that in the multilabel case, probabilities are the marginal probability that a given sample falls in the given class. As such, in the multilabel case the sum of these probabilities over all possible labels for a given sample *will not* sum to unity, as they do in the single label case.

**User guide:** See the [Multiclass classification](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification) section for further details.

| [`multiclass.OneVsRestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier)(estimator, *) | One-vs-the-rest (OvR) multiclass strategy.          |
| ------------------------------------------------------------ | --------------------------------------------------- |
| [`multiclass.OneVsOneClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier)(estimator, *) | One-vs-one multiclass strategy.                     |
| [`multiclass.OutputCodeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html#sklearn.multiclass.OutputCodeClassifier)(estimator, *) | (Error-Correcting) Output-Code multiclass strategy. |



## [`sklearn.multioutput`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.multioutput): Multioutput regression and classification

This module implements multioutput regression and classification.

The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. The meta-estimator extends single output estimators to multioutput estimators.

**User guide:** See the [Multilabel classification](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification), [Multiclass-multioutput classification](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-multioutput-classification), and [Multioutput regression](https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression) sections for further details.

| [`multioutput.ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain)(base_estimator, *) | A multi-label model that arranges binary classifiers into a chain. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`multioutput.MultiOutputRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor)(estimator, *) | Multi target regression.                                     |
| [`multioutput.MultiOutputClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier)(estimator, *) | Multi target classification.                                 |
| [`multioutput.RegressorChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain)(base_estimator, *) | A multi-label model that arranges regressions into a chain.  |



## [`sklearn.naive_bayes`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes): Naive Bayes

The [`sklearn.naive_bayes`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes) module implements Naive Bayes algorithms. These are supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions.

**User guide:** See the [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes) section for further details.

| [`naive_bayes.BernoulliNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)(*[, alpha, ...]) | Naive Bayes classifier for multivariate Bernoulli models.    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`naive_bayes.CategoricalNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB)(*[, alpha, ...]) | Naive Bayes classifier for categorical features.             |
| [`naive_bayes.ComplementNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB)(*[, alpha, ...]) | The Complement Naive Bayes classifier described in Rennie et al. (2003). |
| [`naive_bayes.GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)(*[, priors, ...]) | Gaussian Naive Bayes (GaussianNB).                           |
| [`naive_bayes.MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)(*[, alpha, ...]) | Naive Bayes classifier for multinomial models.               |



## [`sklearn.neighbors`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors): Nearest Neighbors

The [`sklearn.neighbors`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors) module implements the k-nearest neighbors algorithm.

**User guide:** See the [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) section for further details.

| [`neighbors.BallTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree)(X[, leaf_size, metric]) | BallTree for fast generalized N-point problems               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`neighbors.KDTree`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree)(X[, leaf_size, metric]) | KDTree for fast generalized N-point problems                 |
| [`neighbors.KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html#sklearn.neighbors.KernelDensity)(*[, bandwidth, ...]) | Kernel Density Estimation.                                   |
| [`neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)([...]) | Classifier implementing the k-nearest neighbors vote.        |
| [`neighbors.KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)([n_neighbors, ...]) | Regression based on k-nearest neighbors.                     |
| [`neighbors.KNeighborsTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsTransformer.html#sklearn.neighbors.KNeighborsTransformer)(*[, mode, ...]) | Transform X into a (weighted) graph of k nearest neighbors.  |
| [`neighbors.LocalOutlierFactor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor)([n_neighbors, ...]) | Unsupervised Outlier Detection using the Local Outlier Factor (LOF). |
| [`neighbors.RadiusNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier)([...]) | Classifier implementing a vote among neighbors within a given radius. |
| [`neighbors.RadiusNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor)([radius, ...]) | Regression based on neighbors within a fixed radius.         |
| [`neighbors.RadiusNeighborsTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsTransformer.html#sklearn.neighbors.RadiusNeighborsTransformer)(*[, ...]) | Transform X into a (weighted) graph of neighbors nearer than a radius. |
| [`neighbors.NearestCentroid`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid)([metric, ...]) | Nearest centroid classifier.                                 |
| [`neighbors.NearestNeighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)(*[, n_neighbors, ...]) | Unsupervised learner for implementing neighbor searches.     |
| [`neighbors.NeighborhoodComponentsAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html#sklearn.neighbors.NeighborhoodComponentsAnalysis)([...]) | Neighborhood Components Analysis.                            |

| [`neighbors.kneighbors_graph`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html#sklearn.neighbors.kneighbors_graph)(X, n_neighbors, *) | Compute the (weighted) graph of k-Neighbors for points in X. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`neighbors.radius_neighbors_graph`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.radius_neighbors_graph.html#sklearn.neighbors.radius_neighbors_graph)(X, radius, *) | Compute the (weighted) graph of Neighbors for points in X.   |
| [`neighbors.sort_graph_by_row_values`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.sort_graph_by_row_values.html#sklearn.neighbors.sort_graph_by_row_values)(graph[, ...]) | Sort a sparse graph such that each row is stored with increasing values. |



## [`sklearn.neural_network`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network): Neural network models

The [`sklearn.neural_network`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network) module includes models based on neural networks.

**User guide:** See the [Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised) and [Neural network models (unsupervised)](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#neural-networks-unsupervised) sections for further details.

| [`neural_network.BernoulliRBM`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM)([n_components, ...]) | Bernoulli Restricted Boltzmann Machine (RBM). |
| ------------------------------------------------------------ | --------------------------------------------- |
| [`neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)([...]) | Multi-layer Perceptron classifier.            |
| [`neural_network.MLPRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)([...]) | Multi-layer Perceptron regressor.             |



## [`sklearn.pipeline`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline): Pipeline

The [`sklearn.pipeline`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline) module implements utilities to build a composite estimator, as a chain of transforms and estimators.

**User guide:** See the [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) section for further details.

| [`pipeline.FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion)(transformer_list, *[, ...]) | Concatenates results of multiple transformer objects. |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [`pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)(steps, *[, memory, verbose]) | Pipeline of transforms with a final estimator.        |

| [`pipeline.make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline)(*steps[, memory, verbose]) | Construct a `Pipeline` from the given estimators.     |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [`pipeline.make_union`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_union.html#sklearn.pipeline.make_union)(*transformers[, n_jobs, ...]) | Construct a FeatureUnion from the given transformers. |



## [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing): Preprocessing and Normalization

The [`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) module includes scaling, centering, normalization, binarization methods.

**User guide:** See the [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) section for further details.

| [`preprocessing.Binarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html#sklearn.preprocessing.Binarizer)(*[, threshold, copy]) | Binarize data (set feature values to 0 or 1) according to a threshold. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer)([func, ...]) | Constructs a transformer from an arbitrary callable.         |
| [`preprocessing.KBinsDiscretizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer)([n_bins, ...]) | Bin continuous data into intervals.                          |
| [`preprocessing.KernelCenterer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html#sklearn.preprocessing.KernelCenterer)() | Center an arbitrary kernel matrix �.                         |
| [`preprocessing.LabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer)(*[, neg_label, ...]) | Binarize labels in a one-vs-all fashion.                     |
| [`preprocessing.LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder)() | Encode target labels with value between 0 and n_classes-1.   |
| [`preprocessing.MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html#sklearn.preprocessing.MultiLabelBinarizer)(*[, ...]) | Transform between iterable of iterables and a multilabel format. |
| [`preprocessing.MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler)(*[, copy]) | Scale each feature by its maximum absolute value.            |
| [`preprocessing.MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)([feature_range, ...]) | Transform features by scaling each feature to a given range. |
| [`preprocessing.Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer)([norm, copy]) | Normalize samples individually to unit norm.                 |
| [`preprocessing.OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)(*[, categories, ...]) | Encode categorical features as a one-hot numeric array.      |
| [`preprocessing.OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder)(*[, ...]) | Encode categorical features as an integer array.             |
| [`preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures)([degree, ...]) | Generate polynomial and interaction features.                |
| [`preprocessing.PowerTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer)([method, ...]) | Apply a power transform featurewise to make data more Gaussian-like. |
| [`preprocessing.QuantileTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer)(*[, ...]) | Transform features using quantiles information.              |
| [`preprocessing.RobustScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler)(*[, ...]) | Scale features using statistics that are robust to outliers. |
| [`preprocessing.SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html#sklearn.preprocessing.SplineTransformer)([n_knots, ...]) | Generate univariate B-spline bases for features.             |
| [`preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)(*[, copy, ...]) | Standardize features by removing the mean and scaling to unit variance. |

| [`preprocessing.add_dummy_feature`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.add_dummy_feature.html#sklearn.preprocessing.add_dummy_feature)(X[, value]) | Augment dataset with an additional dummy feature.            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`preprocessing.binarize`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html#sklearn.preprocessing.binarize)(X, *[, threshold, copy]) | Boolean thresholding of array-like or scipy.sparse matrix.   |
| [`preprocessing.label_binarize`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html#sklearn.preprocessing.label_binarize)(y, *, classes) | Binarize labels in a one-vs-all fashion.                     |
| [`preprocessing.maxabs_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html#sklearn.preprocessing.maxabs_scale)(X, *[, axis, copy]) | Scale each feature to the [-1, 1] range without breaking the sparsity. |
| [`preprocessing.minmax_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html#sklearn.preprocessing.minmax_scale)(X[, ...]) | Transform features by scaling each feature to a given range. |
| [`preprocessing.normalize`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize)(X[, norm, axis, ...]) | Scale input vectors individually to unit norm (vector length). |
| [`preprocessing.quantile_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform)(X, *[, ...]) | Transform features using quantiles information.              |
| [`preprocessing.robust_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale)(X, *[, axis, ...]) | Standardize a dataset along any axis.                        |
| [`preprocessing.scale`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)(X, *[, axis, with_mean, ...]) | Standardize a dataset along any axis.                        |
| [`preprocessing.power_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform)(X[, method, ...]) | Parametric, monotonic transformation to make data more Gaussian-like. |



## [`sklearn.random_projection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection): Random projection

Random Projection transformers.

Random Projections are a simple and computationally efficient way to reduce the dimensionality of the data by trading a controlled amount of accuracy (as additional variance) for faster processing times and smaller model sizes.

The dimensions and distribution of Random Projections matrices are controlled so as to preserve the pairwise distances between any two samples of the dataset.

The main theoretical result behind the efficiency of random projection is the [Johnson-Lindenstrauss lemma (quoting Wikipedia)](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma):

> In mathematics, the Johnson-Lindenstrauss lemma is a result concerning low-distortion embeddings of points from high-dimensional into low-dimensional Euclidean space. The lemma states that a small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The map used for the embedding is at least Lipschitz, and can even be taken to be an orthogonal projection.

**User guide:** See the [Random Projection](https://scikit-learn.org/stable/modules/random_projection.html#random-projection) section for further details.

| [`random_projection.GaussianRandomProjection`](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection)([...]) | Reduce dimensionality through Gaussian random projection. |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`random_projection.SparseRandomProjection`](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html#sklearn.random_projection.SparseRandomProjection)([...]) | Reduce dimensionality through sparse random projection.   |

| [`random_projection.johnson_lindenstrauss_min_dim`](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html#sklearn.random_projection.johnson_lindenstrauss_min_dim)(...) | Find a 'safe' number of components to randomly project to. |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
|                                                              |                                                            |



## [`sklearn.semi_supervised`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised): Semi-Supervised Learning

The [`sklearn.semi_supervised`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.semi_supervised) module implements semi-supervised learning algorithms. These algorithms utilize small amounts of labeled data and large amounts of unlabeled data for classification tasks. This module includes Label Propagation.

**User guide:** See the [Semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html#semi-supervised) section for further details.

| [`semi_supervised.LabelPropagation`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html#sklearn.semi_supervised.LabelPropagation)([kernel, ...]) | Label Propagation classifier.                      |
| ------------------------------------------------------------ | -------------------------------------------------- |
| [`semi_supervised.LabelSpreading`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading)([kernel, ...]) | LabelSpreading model for semi-supervised learning. |
| [`semi_supervised.SelfTrainingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html#sklearn.semi_supervised.SelfTrainingClassifier)(...) | Self-training classifier.                          |



## [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm): Support Vector Machines

The [`sklearn.svm`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm) module includes Support Vector Machine algorithms.

**User guide:** See the [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html#svm) section for further details.

### Estimators

| [`svm.LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)([penalty, loss, dual, tol, C, ...]) | Linear Support Vector Classification. |
| ------------------------------------------------------------ | ------------------------------------- |
| [`svm.LinearSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)(*[, epsilon, tol, C, loss, ...]) | Linear Support Vector Regression.     |
| [`svm.NuSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC)(*[, nu, kernel, degree, gamma, ...]) | Nu-Support Vector Classification.     |
| [`svm.NuSVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR)(*[, nu, C, kernel, degree, gamma, ...]) | Nu Support Vector Regression.         |
| [`svm.OneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM)(*[, kernel, degree, gamma, ...]) | Unsupervised Outlier Detection.       |
| [`svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)(*[, C, kernel, degree, gamma, ...]) | C-Support Vector Classification.      |
| [`svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)(*[, kernel, degree, gamma, coef0, ...]) | Epsilon-Support Vector Regression.    |

| [`svm.l1_min_c`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.l1_min_c.html#sklearn.svm.l1_min_c)(X, y, *[, loss, fit_intercept, ...]) | Return the lowest bound for C. |
| ------------------------------------------------------------ | ------------------------------ |
|                                                              |                                |



## [`sklearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree): Decision Trees

The [`sklearn.tree`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree) module includes decision tree-based models for classification and regression.

**User guide:** See the [Decision Trees](https://scikit-learn.org/stable/modules/tree.html#tree) section for further details.

| [`tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)(*[, criterion, ...]) | A decision tree classifier.              |
| ------------------------------------------------------------ | ---------------------------------------- |
| [`tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)(*[, criterion, ...]) | A decision tree regressor.               |
| [`tree.ExtraTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html#sklearn.tree.ExtraTreeClassifier)(*[, criterion, ...]) | An extremely randomized tree classifier. |
| [`tree.ExtraTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html#sklearn.tree.ExtraTreeRegressor)(*[, criterion, ...]) | An extremely randomized tree regressor.  |

| [`tree.export_graphviz`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz)(decision_tree[, ...]) | Export a decision tree in DOT format.                     |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| [`tree.export_text`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html#sklearn.tree.export_text)(decision_tree, *[, ...]) | Build a text report showing the rules of a decision tree. |

### Plotting

| [`tree.plot_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree)(decision_tree, *[, ...]) | Plot a decision tree. |
| ------------------------------------------------------------ | --------------------- |
|                                                              |                       |



## [`sklearn.utils`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils): Utilities

The [`sklearn.utils`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.utils) module includes various utilities.

**Developer guide:** See the [Utilities for Developers](https://scikit-learn.org/stable/developers/utilities.html#developers-utils) page for further details.

| [`utils.Bunch`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch)(**kwargs) | Container object exposing keys as attributes. |
| ------------------------------------------------------------ | --------------------------------------------- |
|                                                              |                                               |

| [`utils.arrayfuncs.min_pos`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.arrayfuncs.min_pos.html#sklearn.utils.arrayfuncs.min_pos) | Find the minimum value of an array over positive values      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`utils.as_float_array`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.as_float_array.html#sklearn.utils.as_float_array)(X, *[, copy, ...]) | Convert an array-like to an array of floats.                 |
| [`utils.assert_all_finite`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.assert_all_finite.html#sklearn.utils.assert_all_finite)(X, *[, allow_nan, ...]) | Throw a ValueError if X contains NaN or infinity.            |
| [`utils.check_X_y`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html#sklearn.utils.check_X_y)(X, y[, accept_sparse, ...]) | Input validation for standard estimators.                    |
| [`utils.check_array`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html#sklearn.utils.check_array)(array[, accept_sparse, ...]) | Input validation on an array, list, sparse matrix or similar. |
| [`utils.check_scalar`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_scalar.html#sklearn.utils.check_scalar)(x, name, target_type, *) | Validate scalar parameters type and value.                   |
| [`utils.check_consistent_length`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_consistent_length.html#sklearn.utils.check_consistent_length)(*arrays) | Check that all arrays have consistent first dimensions.      |
| [`utils.check_random_state`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_random_state.html#sklearn.utils.check_random_state)(seed) | Turn seed into a np.random.RandomState instance.             |
| [`utils.class_weight.compute_class_weight`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html#sklearn.utils.class_weight.compute_class_weight)(...) | Estimate class weights for unbalanced datasets.              |
| [`utils.class_weight.compute_sample_weight`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html#sklearn.utils.class_weight.compute_sample_weight)(...) | Estimate sample weights by class for unbalanced datasets.    |
| [`utils.deprecated`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.deprecated.html#sklearn.utils.deprecated)([extra]) | Decorator to mark a function or class as deprecated.         |
| [`utils.estimator_checks.check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator)([...]) | Check if estimator adheres to scikit-learn conventions.      |
| [`utils.estimator_checks.parametrize_with_checks`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html#sklearn.utils.estimator_checks.parametrize_with_checks)(...) | Pytest specific decorator for parametrizing estimator checks. |
| [`utils.estimator_html_repr`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_html_repr.html#sklearn.utils.estimator_html_repr)(estimator) | Build a HTML representation of an estimator.                 |
| [`utils.extmath.safe_sparse_dot`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.safe_sparse_dot.html#sklearn.utils.extmath.safe_sparse_dot)(a, b, *[, ...]) | Dot product that handle the sparse matrix case correctly.    |
| [`utils.extmath.randomized_range_finder`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_range_finder.html#sklearn.utils.extmath.randomized_range_finder)(A, *, ...) | Compute an orthonormal matrix whose range approximates the range of A. |
| [`utils.extmath.randomized_svd`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html#sklearn.utils.extmath.randomized_svd)(M, n_components, *) | Compute a truncated randomized SVD.                          |
| [`utils.extmath.fast_logdet`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.fast_logdet.html#sklearn.utils.extmath.fast_logdet)(A) | Compute logarithm of determinant of a square matrix.         |
| [`utils.extmath.density`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.density.html#sklearn.utils.extmath.density)(w, **kwargs) | Compute density of a sparse vector.                          |
| [`utils.extmath.weighted_mode`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.weighted_mode.html#sklearn.utils.extmath.weighted_mode)(a, w, *[, axis]) | Return an array of the weighted modal (most common) value in the passed array. |
| [`utils.gen_batches`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_batches.html#sklearn.utils.gen_batches)(n, batch_size, *[, ...]) | Generator to create slices containing `batch_size` elements from 0 to `n`. |
| [`utils.gen_even_slices`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.gen_even_slices.html#sklearn.utils.gen_even_slices)(n, n_packs, *[, n_samples]) | Generator to create `n_packs` evenly spaced slices going up to `n`. |
| [`utils.graph.single_source_shortest_path_length`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph.single_source_shortest_path_length.html#sklearn.utils.graph.single_source_shortest_path_length)(...) | Return the length of the shortest path from source to all reachable nodes. |
| [`utils.indexable`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.indexable.html#sklearn.utils.indexable)(*iterables) | Make arrays indexable for cross-validation.                  |
| [`utils.metaestimators.available_if`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.metaestimators.available_if.html#sklearn.utils.metaestimators.available_if)(check) | An attribute that is available only if check returns a truthy value. |
| [`utils.multiclass.type_of_target`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.type_of_target.html#sklearn.utils.multiclass.type_of_target)(y[, input_name]) | Determine the type of data indicated by the target.          |
| [`utils.multiclass.is_multilabel`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.is_multilabel.html#sklearn.utils.multiclass.is_multilabel)(y) | Check if `y` is in a multilabel format.                      |
| [`utils.multiclass.unique_labels`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.multiclass.unique_labels.html#sklearn.utils.multiclass.unique_labels)(*ys) | Extract an ordered array of unique labels.                   |
| [`utils.murmurhash3_32`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.murmurhash3_32.html#sklearn.utils.murmurhash3_32) | Compute the 32bit murmurhash3 of key at seed.                |
| [`utils.resample`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html#sklearn.utils.resample)(*arrays[, replace, ...]) | Resample arrays or sparse matrices in a consistent way.      |
| [`utils._safe_indexing`](https://scikit-learn.org/stable/modules/generated/sklearn.utils._safe_indexing.html#sklearn.utils._safe_indexing)(X, indices, *[, axis]) | Return rows, items or columns of X using indices.            |
| [`utils.safe_mask`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_mask.html#sklearn.utils.safe_mask)(X, mask) | Return a mask which is safe to use on X.                     |
| [`utils.safe_sqr`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.safe_sqr.html#sklearn.utils.safe_sqr)(X, *[, copy]) | Element wise squaring of array-likes and sparse matrices.    |
| [`utils.shuffle`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html#sklearn.utils.shuffle)(*arrays[, random_state, n_samples]) | Shuffle arrays or sparse matrices in a consistent way.       |
| [`utils.sparsefuncs.incr_mean_variance_axis`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.incr_mean_variance_axis.html#sklearn.utils.sparsefuncs.incr_mean_variance_axis)(X, ...) | Compute incremental mean and variance along an axis on a CSR or CSC matrix. |
| [`utils.sparsefuncs.inplace_column_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_column_scale.html#sklearn.utils.sparsefuncs.inplace_column_scale)(X, scale) | Inplace column scaling of a CSC/CSR matrix.                  |
| [`utils.sparsefuncs.inplace_row_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_row_scale.html#sklearn.utils.sparsefuncs.inplace_row_scale)(X, scale) | Inplace row scaling of a CSR or CSC matrix.                  |
| [`utils.sparsefuncs.inplace_swap_row`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_row.html#sklearn.utils.sparsefuncs.inplace_swap_row)(X, m, n) | Swap two rows of a CSC/CSR matrix in-place.                  |
| [`utils.sparsefuncs.inplace_swap_column`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_swap_column.html#sklearn.utils.sparsefuncs.inplace_swap_column)(X, m, n) | Swap two columns of a CSC/CSR matrix in-place.               |
| [`utils.sparsefuncs.mean_variance_axis`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.mean_variance_axis.html#sklearn.utils.sparsefuncs.mean_variance_axis)(X, axis) | Compute mean and variance along an axis on a CSR or CSC matrix. |
| [`utils.sparsefuncs.inplace_csr_column_scale`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs.inplace_csr_column_scale.html#sklearn.utils.sparsefuncs.inplace_csr_column_scale)(X, ...) | Inplace column scaling of a CSR matrix.                      |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l1`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l1) | Inplace row normalize using the l1 norm                      |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l2`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2.html#sklearn.utils.sparsefuncs_fast.inplace_csr_row_normalize_l2) | Inplace row normalize using the l2 norm                      |
| [`utils.random.sample_without_replacement`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.random.sample_without_replacement.html#sklearn.utils.random.sample_without_replacement) | Sample integers without replacement.                         |
| [`utils.validation.check_is_fitted`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html#sklearn.utils.validation.check_is_fitted)(estimator) | Perform is_fitted validation for estimator.                  |
| [`utils.validation.check_memory`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_memory.html#sklearn.utils.validation.check_memory)(memory) | Check that `memory` is joblib.Memory-like.                   |
| [`utils.validation.check_symmetric`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_symmetric.html#sklearn.utils.validation.check_symmetric)(array, *[, ...]) | Make sure that array is 2D, square and symmetric.            |
| [`utils.validation.column_or_1d`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.column_or_1d.html#sklearn.utils.validation.column_or_1d)(y, *[, dtype, ...]) | Ravel column or 1d numpy array, else raises an error.        |
| [`utils.validation.has_fit_parameter`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.has_fit_parameter.html#sklearn.utils.validation.has_fit_parameter)(...) | Check whether the estimator's fit method supports the given parameter. |

Specific utilities to list scikit-learn components:

| [`utils.discovery.all_estimators`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_estimators.html#sklearn.utils.discovery.all_estimators)([type_filter]) | Get a list of all estimators from `sklearn`. |
| ------------------------------------------------------------ | -------------------------------------------- |
| [`utils.discovery.all_displays`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_displays.html#sklearn.utils.discovery.all_displays)() | Get a list of all displays from `sklearn`.   |
| [`utils.discovery.all_functions`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.discovery.all_functions.html#sklearn.utils.discovery.all_functions)() | Get a list of all functions from `sklearn`.  |

Utilities from joblib:

| [`utils.parallel.delayed`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel.delayed.html#sklearn.utils.parallel.delayed)(function) | Decorator used to capture the arguments of a function.       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`utils.parallel_backend`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel_backend.html#sklearn.utils.parallel_backend)(backend[, n_jobs, ...]) | Change the default backend used by Parallel inside a with block. |
| [`utils.register_parallel_backend`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.register_parallel_backend.html#sklearn.utils.register_parallel_backend)(name, factory) | Register a new Parallel backend factory.                     |

| [`utils.parallel.Parallel`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel.Parallel.html#sklearn.utils.parallel.Parallel)([n_jobs, backend, ...]) | Tweak of [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html#joblib.Parallel) that propagates the scikit-learn configuration. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

## Recently deprecated

### To be removed in 1.3

| [`utils.metaestimators.if_delegate_has_method`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.metaestimators.if_delegate_has_method.html#sklearn.utils.metaestimators.if_delegate_has_method)(...) | Create a decorator for methods that are delegated to a sub-estimator. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

© 2007 - 2023, scikit-learn developers (BSD License). [Show this page source](https://scikit-learn.org/stable/_sources/modules/classes.rst.txt)