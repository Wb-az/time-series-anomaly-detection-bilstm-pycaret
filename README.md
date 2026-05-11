<h1 align="center">Sensor Signal Anomaly Detection for Predictive Maintenance</h1>
<h2 align="center">PyCaret vs Custom BiLSTM on the NASA IMS Bearing Dataset</h2>

## Executive Summary

This project compares six unsupervised anomaly detection models implemented through PyCaret with a custom Bidirectional LSTM (BiLSTM) autoencoder built in PyTorch for detecting abnormal patterns in bearing vibration sensor data.

Using the NASA IMS bearing dataset, the study evaluates whether lower-complexity, low-code methods can achieve performance comparable to a custom deep learning model on unseen data. The project focuses on predictive maintenance and operational reliability, where early detection of abnormal sensor behaviour can support fault diagnosis and reduce unplanned downtime.

Key findings:
- BiLSTM and iForest showed comparable performance on the independent test dataset.
- MCD and SVM performed strongly on cluster quality metrics in the test set.
- Some PyCaret models were less robust on unseen data and over-flagged anomalies.
- Low-code anomaly detection methods can be competitive, depending on data quality, contamination, and evaluation criteria.

---

## Introduction

Industrial systems generate continuous streams of time-series sensor data. Detecting abnormal behaviour early is important for predictive maintenance, fault diagnosis, and operational reliability.

This project benchmarks six PyCaret-based unsupervised anomaly detection models against a custom Bidirectional LSTM (BiLSTM) autoencoder implemented in PyTorch, using vibration sensor data from the NASA IMS bearing dataset.

The goal is to assess whether low-code anomaly detection methods can provide performance comparable to a custom deep learning approach while reducing implementation complexity and development effort.

---

## Problem Statement

Anomaly detection in industrial sensor data is challenging because:

- labelled anomalies are often unavailable
- contamination levels may be unknown
- model performance can vary across unseen datasets
- methods that perform well on selected training subsets may not generalise well in practice

Several prior studies have reported strong results for bearing fault detection using deep learning and semi-supervised approaches. However, many of these studies rely on selected training subsets or limited portions of the available data, making it difficult to assess generalisability or compare against lower-complexity approaches.

---

## Objective

The aim of this study is to compare classical unsupervised anomaly detection models with a custom BiLSTM autoencoder on the same bearing sensor datasets.

Specifically, the project evaluates whether PyCaret-based approaches can achieve performance similar to, or better than, a neural-network-based approach while requiring less implementation effort.

Model performance is assessed using:
- cluster quality metrics
- non-parametric statistical comparison
- anomaly detection behaviour on both training and independent test datasets

The statistical comparison includes the Friedman test and Friedman-Conover post hoc analysis.

---

## Dataset

The data were sourced from [Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) and are derived from the NASA Acoustics and Vibration Database.

The NASA IMS bearing dataset comprises **three run-to-failure experiments** collected from a bearing test rig under different fault conditions. The data consist of vibration sensor readings stored as **ASCII text files**, with each file containing a **1-second signal snapshot of 20,480 data points** sampled at **20 kHz**. Files were generally recorded at **10-minute intervals**, with one exception in Set 1 where the first 43 files were recorded every 5 minutes.

Each channel corresponds to a specific bearing position, making the dataset suitable for studying **anomaly detection, fault diagnosis, and predictive maintenance** in rotating machinery.

### Dataset Summary

- **Set 1** was recorded from **22 October 2003 to 25 November 2003** and contains **2,156 files** across **8 channels**.  
  Channel arrangement: Bearing 1 = Ch 1–2, Bearing 2 = Ch 3–4, Bearing 3 = Ch 5–6, Bearing 4 = Ch 7–8.  
  By the end of the experiment, **bearing 3 developed an inner race defect** and **bearing 4 developed a roller element defect**.

- **Set 2** was recorded from **12 February 2004 to 19 February 2004** and contains **984 files** across **4 channels**.  
  Channel arrangement: Bearing 1 = Ch 1, Bearing 2 = Ch 2, Bearing 3 = Ch 3, Bearing 4 = Ch 4.  
  By the end of the experiment, **bearing 1 developed an outer race failure**.

- **Set 3** was recorded from **4 March 2004 to 4 April 2004** and contains **4,448 files** across **4 channels**.  
  Channel arrangement: Bearing 1 = Ch 1, Bearing 2 = Ch 2, Bearing 3 = Ch 3, Bearing 4 = Ch 4.  
  By the end of the experiment, **bearing 3 developed an outer race failure**.

In this project, the experiments were used to compare anomaly detection behaviour across training and independent test conditions, <br> with averaged signal representations used for model comparison and visualisation.

Only Sets 2 and 3 were used and denoised using Fast Fourier Transforms (fft) for training and testing. Figure 1 shows the average signal representations of theses datasets before denoising.
Figure 1 shows the datasets’ average signal representation for the model comparisons.


<p align="center">
  Train dataset - Dataset 2 
</p>

<p align="center">
  <img height="200" src="plots/avg_dataset2.png" width="600" alt="Anomaly distribution on train dataset"/>
</p>
<p align="center">
  Test dataset - Dataset 3 
</p>
<p align="center">
  <img height="200" src="plots/avg_dataset3.png" width="600" alt="Anomaly distribution on train dataset"/>
<p align="center">
Figure 1. Average signal: dataset 2 and dataset 3.
</p>

---

## Methods

### PyCaret Models
PyCaret was selected as a low-code framework that automates several elements of the anomaly detection workflow. Table 1 shows the unsupervised anomaly detection models available through PyCaret were evaluated.

__Table 1__: PyCaret-based anomaly detection models.

| ID        | Name                              | Reference                 |
|-----------|-----------------------------------|---------------------------|
| cluster   | Clustering-Based Local Outlier    | pyod.models.cblof.CBLOF   |
| iforest   | Isolation Forest                  | pyod.models.iforest.IForest |
| histogram | Histogram-Based Outlier Detection | pyod.models.hbos.HBOS     |
| knn       | K-Nearest Neighbours Detector     | pyod.models.knn.KNN       |
| svm       | One-Class SVM Detector            | pyod.models.ocsvm.OCSVM   |
| mcd       | Minimum Covariance Determinant    | pyod.models.mcd.MCD       |



### Custom BiLSTM Autoencoder

A custom Bidirectional LSTM autoencoder was implemented in PyTorch to encode and reconstruct the vibration signal input.

All experiments (Table 2) were run with:
- 50 epochs
- learning rate of 2e-4
- batch size of 32
- one BiLSTM layer with 32 hidden units
- dropout of 0.1

__Table 2:__ BILSTM Experimental setup.

| Exp | Model  | Loss       | Optimiser |
|-----|--------|------------|-----------|
| 1   | bilstm | mae_loss   | adam      |
| 2   | bilstm | huber_loss | adam      |
| 3   | bilstm | mae_loss   | adamw     |
| 4   | bilstm | huber_loss | adamw     |

---

## Evaluation Approach

The models were compared using both cluster quality metrics and non-parametric statistical analysis.

### Cluster Quality Metrics

- **Silhouette Score** measures how well-separated the clusters are. Higher values indicate better-defined clusters.
- **Calinski-Harabasz Index** measures between-cluster separation relative to within-cluster dispersion. Higher values indicate better-defined clusters.
- **Davies-Bouldin Index** measures similarity between each cluster and its most similar cluster. Lower values indicate better separation.

### Statistical Comparison

Because cluster metrics can favour specific cluster shapes, model comparison also included:
- **Friedman test**
- **Friedman-Conover post hoc pairwise analysis**

This provided a more robust comparison of relative model performance across experiments.

---

## Results Overview

### Anomalies Detected

Table 3 summarises the number of anomalies detected by each model on the training and test datasets.

__Table 3:__ Anomalies detected by each model.

| Model     | Anomalies - </br>training dataset | Anomalies- </br>Test dataset |
|-----------|:---------------------------------:|:----------------------------:|
| Cluster   |                50                 |              -               |
| Histogram |                50                 |              -               |
| iforest   |                50                 |             187              |
| KNN       |                50                 |              -               |
| MCD       |                50                 |              78              |
| SVM       |                50                 |              95              |
| Exp-01    |                99                 |             190              |
| Exp-02    |                98                 |             191              |
| Exp-03    |                99                 |             190              |
| Exp-04    |                104                |             191              |

### Training Dataset

On the training dataset:
- SVM, MCD, and Histogram achieved the strongest cluster metric values.
- iForest also showed competitive separation performance.
- All BiLSTM experiments performed differently from the PyCaret models under the Friedman-Conover comparison.
- Among the BiLSTM experiments, there was no significant difference between the four configurations.
- Exp-04 ranked highest overall on the training set, although it was not significantly different from the other BiLSTM configurations.
- In terms of early detection, Exp-02 and Exp-04 identified the anomaly approximately 22 hours in advance, while the clustering-based model detected it approximately 14 hours and 40 minutes in advance.

### Test Dataset

On the independent test dataset:
- MCD and SVM achieved the strongest cluster metric values.
- iForest performed competitively and showed robustness relative to changes in dataset size and unknown contamination.
- The Friedman test rejected the null hypothesis that all models performed equivalently.
- The Friedman-Conover comparison showed no significant difference among Exp-01 to Exp-04 and iForest at the 95% confidence level.
- MCD and SVM did not differ significantly from each other, but they differed from other model groups in the comparison.
- The BiLSTM configurations detected anomalies with lead times of up to approximately 27 hours, while iForest detected anomalies approximately 25 hours and 30 minutes in advance.

---

## Visual Results

### PyCaret Models

<p align="center">
  Train dataset - Dataset 2 (avg_df2)
</p>

<p align="center">
  <img height="1200" src="plots/scatter_anomalies.png" width="600" alt="Anomaly distribution on train dataset"/>
</p>
<p align="center">
Figure 2. Anomalies detected by the PyCaret models on the training dataset.
</p>

 
<p></p>
<p align="center">
Test dataset - Dataset 3 (avg_df3)
</p>
<p align="center">
  <img alt="Fig" height="500" src="plots/merged_test_anomaly_prediction.png" width="600"/> 
</p>

<p align="center">
  Figure 3. Anomalies detected by the PyCaret models on the test dataset.
</p>


### BiLSTM Models


<p align="center">  
  Train dataset - Dataset 2 (avg_df2)
</p>

<p align="center">
<img src="plots/merged_bilstm_val_pred_anom_exp1.png" width="400" height="150" alt="train_anomaly exp1"/>A <img height="150" src="plots/merged_bilstm_val_pred_anom_exp2.png" title="train anomaly exp2" 
width="400"/>B
<img alt="train anomalies exp3" height="150" src="plots/merged_bilstm_val_pred_anom_exp3.png" width="400"/>C <img height="150" src="plots/merged_bilstm_val_pred_anom_exp4.png" width="400" title="train anomalies exp4"/>B
</p>

<p align="center">
  Figure 4. Anomalies distribution detected on the training dataset. The experimental setup is outlined in Table 2. A. Exp-01, B.Exp-02, C. Exp-03, D.Exp-04
</p>

<br>
<p align="center">
  Test dataset - Dataset 3 (avg_df3)
</p>
<p align="center">
<img src="plots/merged_test_bilstm_anom_exp1.png" width="400" height="150" alt="test_anomaly exp1"/>A <img src="plots/merged_test_bilstm_anom_exp2.png" width="400" height="150" alt="test_anomaly exp2"/>B
<img src="plots/merged_test_bilstm_anom_exp3.png" width="400" height="150" alt="test_anomaly exp3"/>C <img src="plots/merged_test_bilstm_anom_exp4.png" width="400" height="150" alt="test_anomaly exp4"/>D
</p>

<p align="center">
Figure 5. Anomalies distribution detected on the test dataset. The experimental setup is outlined in Table 2. A. Exp-01, B.Exp-02, C. Exp-03, D.Exp-04.
</p>

---

## Cluster Metrics

### Training Dataset

The result of the performnace metrics used to evaluate the clusters on the training dataset is shown in Table 4.

__Table 4__: Cluster metrics on the training dataset.

|index|silhoutte|calinski\_harabasz|davies\_bouldin|
|---|---|---|---|
|cluster|0\.7762|646\.9312|0\.8175|
|histogram|0\.8124|1001\.4351|0\.6754|
|iforest|0\.8124|992\.9829|0\.6791|
|knn|0\.8017|897\.689|0\.7167|
|mcd|0\.8124|1004\.5345|0\.6739|
|svm|0\.814|1011\.3661|0\.6723|
|exp1|0\.7442|879\.5458|0\.7615|
|exp2|0\.747|892\.5695|0\.7562|
|exp3|0\.7442|879\.5458|0\.7615|
|exp4|0\.7386|875\.7697|0\.764|

### Test Dataset

__Table 5__: Cluster metrics on the test dataset.

|index|silhoutte|calinski\_harabasz|davies\_bouldin|
|---|---|---|---|
|iforest|0\.923|7629\.4839|0\.7218|
|mcd|0\.9557|15873\.0144|0\.3554|
|svm|0\.9533|16564\.1182|0\.3939|
|exp1|0\.9263|8549\.185|0\.6857|
|exp2|0\.9256|8389\.7269|0\.692|
|exp3|0\.9267|8656\.1093|0\.6818|
|exp4|0\.9256|8433\.0381|0\.6902|

---

## Non-Parametric Statistical Comparison

### Model Ranking

__Table 6__: Models performance ranking for the training and test datasets.

| Model     | Training Ranks | Test Ranks |
|-----------|:--------------:|:----------:|
| Exp-01    |    0.5647      |  0.5738    |
| Exp-02    |    0.5642      |  0.5739    |
| Exp-03    |    0.5647      |  0.5738    |
| Exp-04    |    0.5673      |  0.5739    |
| Cluster   |    0.5398      |     -      |
| Histogram |    0.5398      |     -      |
| iForest   |    0.5398      |  0.5735    |
| KNN       |    0.5398      |     -      |
| MCD       |    0.5398      |  0.5663    |
| SVM       |    0.5398     |   0.5649    |

### Training Results

- Friedman chi-square test: p-value = 8.85e-75, so the null hypothesis was rejected.
- Post hoc Friedman-Conover analysis showed significant differences between the BiLSTM experiments and the PyCaret models.
- The four BiLSTM configurations were not significantly different from one another.

<p align="center">
<img height="250" src="plots/posthoc_train_models.png" width="350" title="train_post"/>
<img height="100" src="plots/critical_dif_train_models.png" width="400" title="train_post"/>
  

Figure 6. Conover-Friedman posthoc comparison test results and critical distance on the unseen test data by model.
</p>

### Test Results

- Friedman chi-square test: p-value = 6.51e-19, so the null hypothesis was rejected.
- Post hoc Friedman-Conover analysis showed no significant difference among Exp-01 to Exp-04 and iForest.
- MCD and SVM formed a separate strong-performing group in the comparison.

* Non-parametri test - Friedman-chisquare pvalue = 6.51e-19. Therefore the H<sub>0</sub> is rejected.
* Posthoc- Friedman-Conover pairwise comparison
* Critical Difference Diagram


<p align="center">
<img height="250" src="plots/posthoc_test_models.png" width="350" title="test_post"/>
<img height="100" src="plots/critical_dif_test_models.png" width="400" title="test_post"/>
  

Figure 7. Conover-Friedman posthoc comparison test results and critical distance on the unseen test data by model.

---

## Conclusion

This project compared six PyCaret anomaly detection models with a custom BiLSTM autoencoder on the NASA IMS bearing dataset.

Overall, Exp-04 performed strongly across both datasets, while iForest showed particularly robust behaviour on the independent test dataset. MCD and SVM also achieved strong cluster metric performance on the test set.

Some PyCaret models, including Histogram, Cluster, and KNN, were excluded from the final test comparison because they flagged an excessively large proportion of the test observations as anomalous, making them less reliable on unseen data.

The results suggest that selected PyCaret anomaly detection models and the custom BiLSTM autoencoder can achieve comparable performance for detecting abnormal bearing sensor behaviour. This is important because lower-complexity methods may offer a practical alternative to custom deep learning approaches in some operational settings.

Further work is needed to assess whether these models can detect failures days or weeks in advance across additional unseen datasets and under different contamination and operating conditions.

## Environment

This project was developed in PyCharm using Python 3.11.

Main libraries used:
- PyTorch
- PyCaret
- PyOD
- scikit-learn
- pandas
- NumPy
- SciPy
- matplotlib
- Plotly
- scikit-posthocs

Install dependencies with:
```bash
pip install -r requirements.txt
```

**Note:** PyCaret can be version-sensitive, so it helps to pin dependencies versions once the project runs cleanly.

---

## References

[1] [PyCaret Documentation](https://pycaret.gitbook.io/docs/).

[2] https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf.

[3] https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770.

[4] https://sabtainahmadml.medium.com/condition-monitoring-through-diagnosis-of-anomalies-lstm-based-unsupervised-ml-approach-5f0565735dff.

[5] Zhang, R.; Peng, Z.; Wu, L.; Yao, B.; Guan, Y. Fault Diagnosis from Raw Sensor Data Using Deep Neural Networks Considering Temporal Coherence. Sensors 2017, 17, 549. https://www.mdpi.com/1424-8220/17/3/549.

[6] Goldstein, M. and Uchida, S. (2016) ‘A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data’, PLOS ONE, 11(4). doi:10.1371/journal.pone.0152173.

[7] K. Choi, J. Yi, C. Park and S. Yoon, "Deep Learning for Anomaly Detection in Time-Series Data: Review, Analysis, and Guidelines," IEEE Access, vol. 9, pp. 120043-120065, 2021, doi: 10.1109/ACCESS.2021.3107975.

