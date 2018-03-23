We would like make formal complaint due to erroneous claims of reviewer #1.
Three strong claims are made which shows both a lack of understanding of the
problem and challenges tackled in our research and a lack of knowledge in
statistics. Therefore, we provide a constructive feedback to put some light.

* So overall, the comparison between the model and the machine learning results
  is probably severely biased. As a baseline, the mechanical model is from
  equations of classical mechanics that without doubt can be considered as
  truthful.

We are surprised about the reviewer's comments which are erroneous. We are not
stating at any point that the mathematical model based on classical mechanic is
wrong: the laws of physic make it truthful as mentioned by the reviewer.

During simulation or with highly constraint experimental conditions in which
all parameters are known, the mathematical model based on physic will perform
perfectly (and probably the machine learning approach as well).

However, in real-world conditions, some assumptions are made upon the
parameters due to a limited knowledge regarding the parameters --- i.e. missing
information, parameters assumed to be constant --- (refer to the comment below
for example). In these conditions, the predictions will be noisy even if the
model itself is truthful. In addition, this lack or limited amount of
information is the same for both models.

Therefore, we strongly state that there is no bias (if the reviewer refers to
favor one of the model), severe or minor, towards a particular model. Such a
claim by the reviewer should be back up theoretically which is not the case in
this review. We are opened to any founded criticism which could improve our
experiments.

Following this remark, we would like to raise 2 questions: (i) what is actually
the bias that the reviewer is stating and (ii) the reviewer mentions "As a
baseline" referring to the mathematical model for which the meaning is not
clear. To clarify, this model cannot be used as a baseline (ground-truth) since
some parameters are measured (including some observational errors) and other
parameters are fixed based on some assumptions due to lack of information.

* Errors in slope may cause the model to perform poorly, and model taken from
  the literature may not be appropriate for the rider/bikes/roads used in the
  study.
  
Errors in the slope is due to error in the GPS measurements. Such errors are
known as observational or measurement errors [1] which are always part of
experiments in real-world conditions. In addition, the slope is computed from
the elevation measurements acquired with a sampling rate of 1Hz. Moreover, it
has been smoothed to reduce the noise. Therefore, the induced error can be
considered as negligible or inferior to measurement provided by inclinometer.
It should be noted that both models mathematical and machine learning models
will be influenced.

As mentioned by the reviewer, our primary goal is to be able to estimate power
in a framework similar to Strava. In this regard, our experiment makes
equivalent assumptions than the ones in Strava [2]. We define the weight for
each rider specifically. In the mathematical model, the weight of each rider is
set specifically while the parameters linked to the road and bike types are set
constant across rider.

As a conclusion, we have a limited knowledge to predict power which is actually
the true challenge of the proposed experiments.

* In the mathematical model there are missing terms: (i) changes in kinetic
  energy, (ii) frictional losses in wheel bearing, and (iii) frictional loss in
  the drive chain. In particular, the first part (power required to accelerate,
  saved when decelerating) will change the results significantly.

It is true that we dismissed these different terms. We consider the frictions
(ii) and (iii) to be negligible --- those frictions are reported to account for
around 2.5% [6]. It should be noted that we used different power-meters
(e.g. rotor, powertap). In this regard, the power data use as ground-truth do
not incorporate the frictional loss of the drive chain.

Regarding the kinetic change, this term is not always included [3] in the
model. However, we included this term in some preliminary experiments which
turn out to be detrimental to predict power. The overall results including the
kinetic term are R2=-0.45 / MAE=57.43 W while excluding the kinetic term the
results are R2=-0.26 / MAE=55.19 W. Note that this experiment can be reproduced
by setting `use_acceleration=True` in `strava_power_model` function in our code
(refer to
https://github.com/scikit-cycling/research/blob/master/power_regression/mathematical_model.py#L73).
We did not include these experiments due to page limitation. However, we are
inclined to include this extra experiment if the page limit is lift up.

* How was the 3-fold validation carried out? The straightforward way to select
  2/3 of the feature vectors for training, and 1/3 for testing, would be
  flawed.
  
The statement of the reviewer is completely erroneous and baseless which make
us question expertise of the reviewer in the field of statistics / machine
learning.

Cross-validation is a standard procedure in statistics and machine learning. We
are using 3 folds with 2/3 of samples for training and 1/3 of samples for
testing. The experiment is repeated 3 times such that each fold is used as a
testing fold. It allows to robustly estimate the average performance and the
confidence bounds of the model [4]. In a similar manner, we could have use a
10-fold cross-validation or any equivalent validation scheme.

The only concern regarding cross-validation that could be raised is about the
sample sizes and its influence on bias finding [5]. However, our sample size is
around 3 millions samples which is above than the critical threshold of small
sample size leading to large error bars.
 
* The text is unclear about which 48 features have been used, please
  clarify. The table 1 should also list the parameters used in the equation.
  
Due to page constraints, we did not explain fully those parts. It is true that
increasing will benefit to readers. We are inclined of such changes.

To be complete, there is initially 8 features: elevation, cadence, heart-rate,
speed, distance, gradient of elevation, gradient of the heart-rate, and the
acceleration. We additionally compute the gradient for each of these features
for different period (t - t-1; t - t-2; ... t - t-5). Therefore, we obtain 6
(original feature + 5 gradient for different periods) x 8 = 48 features from
which the gradient boosting will find the most significant when building the
decision trees.

* It is unclear how the method can be modified to estimate SC_x using machine
  learning, as there cannot be a training set by measuring SC_x as easily as
  power in this contribution.
  
This part is stated as future work and need more investigation. However, one of
the requirement will be to acquired a data set containing all necessary
information. In this regard, we will need to acquire data containing the same
information than in this study (i.e. speed, heart-rate, power, speed, cadence,
elevation, distance). Subsequently, we need to estimate SCx during these
acquisitions. In this regard, we need temperature, pressure (sometime provided
also by the computer bike) and wind speed measurements (with an additional
anemometer). SCx would be estimated as presented in [7-8]. Note that these
works are ongoing research.


[1] Dodge, Yadolah, ed. The Oxford dictionary of statistical terms. Oxford
University Press on Demand, 2006.

[2] https://support.strava.com/hc/en-us/articles/216917107-How-Strava-Calculates-Power

[3] Grappe, Fred. Cyclisme et optimisation de la performance: science et
méthodologie de l’entraînement. De Boeck Supérieur, 2009.

[4] Seni, Giovanni, and John F. Elder. "Ensemble methods in data mining:
improving accuracy through combining predictions." Synthesis Lectures on Data
Mining and Knowledge Discovery 2.1 (2010): 1-126.

[5] Varoquaux, Gaël. "Cross-validation failure: small sample sizes lead to
large error bars." NeuroImage (2017).

[6] Martin, James C., Douglas L. Milliken, John E. Cobb, Kevin L. McFadden, and
Andrew R. Coggan "Validation of a mathematical model for road cycling power."
Journal of applied biomechanics 14.3 (1998): 276-291.

[7] Voiry, Matthieu, Lemaitre Cedric, and Andre Cyrille. "Toward a robust and
inexpensive method to assess the aerodynamic drag of cyclists." Journal of
Science and Cycling 6.3 (2017).

[8] Voiry, Matthieu, Lemaitre Cedric, and Andre Cyrille."First evaluation of an
automated system for cyclist’s aerodynamic drag assessment", Submitted to
Science and Cycling (2018)
