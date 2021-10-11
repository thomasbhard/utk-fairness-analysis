

# UTK Face Fairness

## Testing the AIF360 toolkit on a face detection problem

Most of the existing material on using fairness toolkits in practice is focusing on the same datasets (e.g., [COMPAS](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis), [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)), [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)), which are then used in “allocation-harm” type problems, where one outcome is considered “good” and the other “bad” (e.g., low/high risk of recidivism, good/bad credit risk). The objective of this work is to investigate whether current fairness toolkits are suitable for other tasks like image classification. More precisely, gender detection using the [UTKFace dataset](https://susanqq.github.io/UTKFace/). This includes measuring fairness and bias as well as bias mitigation at the different stages of the ML pipeline, namely pre-, in-, and post-processing. The task at hand differs from the previous works in two main ways:

1.)	The input data is composed of images as opposed to numerical features. This does not only call for different model architectures but also limits the use of algorithms that make use of or change the features.

2.)	There is no “good” or “bad” label, and the interesting metric is how well the face detection works for different groups. This means that the attribute for which fairness is considered is not included in the dataset but rather a result of the prediction. 

The project focuses on the [AIF360](https://aif360.mybluemix.net/) platform as it offers a wide variety of algorithms for both measuring and mitigating bias. It should give insight into how well current fairness toolkits can be used for tasks that differ from the prime examples and how they could be improved to translate to a wider range of problems.
