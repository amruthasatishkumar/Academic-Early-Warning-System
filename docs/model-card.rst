Model card: synthetic four-outcome demonstration
================================================

Intended use
------------

Demonstrate a reproducible student-course early-warning method, time-aware
feature preparation and honest explanation of one model. Not suitable for
real student decisions, denial of opportunities, automated outreach or
institutional performance claims without new institution-approved work.

Targets and data
----------------

One mutually exclusive course outcome is modeled: Excellent, Pass, Fail or
Withdrawn. Confirmed post-checkpoint course withdrawal takes precedence over
grades. Other historical labels require qualified, finalized course grades.
Unknown/unqualified or already-withdrawn-before-checkpoint records are not
coerced to Fail. Current term records contain no published future truth.

The fixed-seed simulator generates correlated engagement, submissions, grade
and withdrawal history with source revisions. It also generates extensions,
delayed grading, excusals, paper work, repeated attempts, anonymous/external
requests and incomplete gradebooks. Hidden latent variables stay inside the
generator. Synthetic correlations were designed, not discovered in an
independent institutional population.

Inputs and training protocol
----------------------------

Sixteen features describe enrolled days, due work, completion, known online
missingness, late work, unknown paper work, excusals, grading coverage,
available grade level/recent level/trend, attempts, activity counts/change
and time since last activity. Missingness remains unknown until training-only
median imputation with missing indicators. Standard scaling is also fit only
on the training rows. All observations satisfy effective and recorded
availability cutoffs; current labels are forbidden.

Terms one and two train candidate pipelines. Term three has one fixed
stratified enrollment split: one fold fits sigmoid calibrators, the other
compares logistic candidates. The candidates use the configured regularization
values and unweighted/balanced classes. A training-prevalence classifier is a
comparison only. Selection is highest macro one-vs-rest average precision,
then lowest multiclass log loss, among logistic candidates. Exactly one
calibrated logistic classifier is published; no separate class-specific
student workflow or thresholded withdrawal workflow is published.

Calibration uses the supported scikit-learn 1.2 prefit API. The underlying
four-class model is fit on training only; per-class sigmoid calibration sees
only the calibration fold, followed by normalization to a four-class simplex.
Balancing changes the fitting population, and calibration does not guarantee
accurate real-world risk.

Evaluation interpretation
-------------------------

Results include a four-class report and confusion matrix, macro F1, multiclass
log loss, per-class average precision, prevalence, Brier error and calibration
bins. Returning/unseen student slices are reported only when all classes are
present; otherwise the slice is explicitly not evaluable.

Model choice and evaluation on the same decision fold introduce selection
optimism. Minority class counts can be small. The provided term four was
previously exposed in the prototype; optional evaluation of it is diagnostic
only and defaults off. It is not a newly independent test. No prospective,
institutional or independently validated accuracy is claimed. High recall
alone, synthetic feature significance and favorable charts are not acceptance.
Four modeled outcomes do not guarantee that every current batch has four
different argmax labels. Missing minority-class predictions are a limitation
to examine, not a reason to force results or change the synthetic labels.

Exact explanation units
-----------------------

For outcome k, the underlying multinomial classifier score is::

    s_k(x) = b_k + sum_j w_kj * z_j(x)

Here z is the fitted imputer-and-scaler output, including missing indicators.
Let r be the mean z over the actual training rows. The reference score is::

    reference_k = b_k + sum_j w_kj * r_j

For an original feature f, group its transformed value and any associated
missing indicator into one contribution::

    contribution_f,k(x) = sum_(j belonging to f) w_kj * (z_j(x) - r_j)
    reference_k + sum_f contribution_f,k(x) = s_k(x)

The code verifies this identity numerically for every explained row and
outcome. Local explanations normally use the predicted outcome, with an
optional selectable outcome. Positive terms raise that outcome's underlying
score relative to the training reference; negative terms lower it.
Top-three drivers are a display summary, not the complete additive equation.

Global per-outcome importance is::

    mean_(x in training) abs(contribution_f,k(x))

Grouping occurs **before** the absolute value and averaging. The same training
reference is used for all current explanations. These are underlying
multinomial **linear class-score units**, sometimes called logits, before
sigmoid calibration and final normalization. An individual class score is not
a calibrated log-odds probability. Contributions are not probability-point
changes, percentage contributions, SHAP values, causal effects or evidence
that changing a feature causes a different outcome.

Publication and provenance
--------------------------

The content-derived model version includes source-code hash, training hash,
calibration/decision hashes and row membership, model configuration, selected
candidate and library versions. Each publication retains its run identity,
source version, exact snapshot paths and observed counts in the current
manifest and an immutable publication manifest.

The exploration reader reloads the publication's own fitted model and
training reference, checks training provenance, and verifies that the model
reproduces every current predicted probability and class. Registration is
always disabled. Licensing and organizational release approval remain the
owner's responsibility.
