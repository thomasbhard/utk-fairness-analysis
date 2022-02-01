# Roadmap


### 1.) Implement manual reweighing strategy:

Reweigh samples according to some metric calculated on a previous testset. For example: Assigning groups the inverse of the disparate impact

### 2.) In-processing algorithms:

Fitting the AIF360 in-processing algorithms to work for this kind of problem. Most importantly implementing a "moment".

### 3.) Post-processing algorithms:

See if any of AIF360 post-processing algorithms are suitable

### 4.) Formulate the problem as a regression task:

Not converting the predictions to a binary label but instead using AIF360's regression tools.

### 5.) Looking into age and race predictions:

Applying the results for the gender prediction to age and race.

