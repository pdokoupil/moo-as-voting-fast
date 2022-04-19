# Towards Results-level Proportionality for Multi-objective Recommender Systems
## Abstract
The main focus of our work is the problem of multiple objectives
optimization (MOO) while providing a final list of recommenda-
tions to the user. Currently, system designers can tune MOO by
setting importance of individual objectives, usually in some kind of
weighted average setting. However, this does not have to translate
into the presence of such objectives in the final results. In contrast,
in our work we would like to allow system designers or end-users
to directly quantify the required relative ratios of individual objec-
tives in the resulting recommendations, e.g., the final results should
have 60% relevance, 30% diversity and 10% novelty. If individual
objectives are transformed to represent quality on the same scale,
these result conditioning expressions may greatly contribute to-
wards recommendations tuneability and explainability as well as
user’s control over recommendations.
To achieve this task, we propose an iterative algorithm inspired
by the mandates allocation problem in public elections. The algo-
rithm is applicable as long as per-item marginal gains of individual
objectives can be calculated. Effectiveness of the algorithm is evalu-
ated on several settings of relevance-novelty-diversity optimization
problem. Furthermore, we also outline several options to scale indi-
vidual objectives to represent similar value for the user.

## Contacts
- Ladislav Peška - *ladislav.peska at matfyz.cuni.cz*
- Patrik Dokoupil - *patrik.dokoupil at matfyz.cuni.cz*

## About this repository
This repository contains source code and other supplementary material for the article Towards Results-level Proportionality for Multi-objective Recommender Systems.

### Project structure
- *mandate_allocation/* directory contains source codes of all the mandate allocation methods
    - *exactly_proportional_fuzzy_dhondt.py* implements EP-FuzzDA algorithm
    - *exactly_proportional_fuzzy_dhondt_2.py* implements RLprop algorithm
    - *fai_strategy.py* implements FAI algorithm
    - *probabilistic_fai_strategy.py* implements the Probabilistic algorithm
    - *sainte_lague_method.py* implements Sainte Lague algorithm
    - *weighted_average_strategy.py* implements wAVG algorithm
- *normalization/* directory contains sources for the individual normalization methods
    - *cdf.py* - wrapper around sklearn's QuantileTransformer
    - *standardization.py* - wrapper around sklearn's StandardScaler
    - *identity.py* - does not produce any normalization
    - *cdf_threshold_shift.py* - experimental, not used for producing the results in the paper
    - *robust_scaler.py* - experimental, not used for producing the results in the paper
- *support/* directory contains sources of the supports functions
    - *intra_list_diversity_support.py* calculates support for ILD diversity objective
    - *popularity_complement_support.py* calculates support for PC novelty objective
    - *rating_based_relevance_support.py* calculates support for relevance objective
- *main.py* - contains the implementation of the recommender which utilizes the abovementioned components + their entrypoint.
- *run_experiments.py* is convenience wrapper around *main.py* that allows to easily run multiple experiments (based on different combinations of input parameters).
- *MLproject* - mlflow project file
- *util.py* - contains some utility functions that were used for processing the results

### Requirements
The project was tested with Python 3.9.10 and the following dependencies installed. Slightly older versions of Python3 may also work.
- matplotlib==3.5.1
- caserecommender==1.1.1 (ensured installation of sklearn and some other dependencies)
- mlflow==1.23.1 (mlflow is used for tracking the results)
- flask==2.0.3
<br>
<br>

You can create *Dockerfile* from the following snippet and use it to build Docker container where the project can be executed:
___
FROM python:3.9.10-slim-buster<br>
<br>
RUN python -m pip install --upgrade pip<br>
<br>
RUN pip install matplotlib==3.5.1<br>
RUN pip install caserecommender==1.1.1<br>
RUN pip install mlflow==1.23.1<br>
RUN pip install flask==2.0.3<br>
<br>
RUN apt-get update<br>
RUN apt-get install -y git
___
<br>

### Parameters
Below is the list of parameters accepted by *main.py* script. Note that parameters for *run_experiments.py* are slightly different, they usually expect list of values which is then passed one-by-one to the *main.py* (We omit parameters for *run_experiments.py* script here for clarity and because they should be easily understood from the script itself).
- **--k** is the length of the recommendation list, default value is 10
- **--train_path** is path of the file containing the training data sessions. Expected format of the file is "user item rating" (without quotes) on each line, separated by tabs (\t).
- **--test_path** path of the file with test sessions.
- **--seed** the random seed, default value is 42
- **--normalization** the normalization that should be used. Values used in experiments are "cdf" and "standardization"
- **--algorithm** the mandate allocation algorithm, name should be one of {exactly_proportional_fuzzy_dhondt, exactly_proportional_fuzzy_dhondt_2, fai_strategy, probabilistic_fai_strategy, sainte_lague_method, weighted_average_strategy}
- **--masking-value** implementation detail, do not modify this value (it is some small constant that is used to "mask out" items already seen/recommended by/to the users)
- **--baseline** underlying (baseline) RS algorithm, should be either MatrixFactorization or ItemKNN
- **--metadata_path** path to the file with metadata, expected to be used in conjuction with **--diversity cb** parameter. In experiments, we used this with Movielens dataset (movies.dat) where each line consisted of "movie id::movie name::genre_1|genre_2|...|genre_n".
- **--diversity** specifies the type of diversity metric being used. Should be either "cf" (collaborative -> features == columns in user-rating matrix) or "cb" (content-based -> features == one-hot encoded item genres/categories)
- **--shift** specifies the shift value that should be used on the resulting normalized support value.
- **--cache_dir** specifies cache directory where the algorithm can save some data. Be aware that this directory should be different for every dataset.
- **--artifact_dir** specifies directory where the results will be saved. Usually this should not be set and it will be inferred automatically based on Mlflow RUN_ID and value of **--output_path_prefix**
- **--discounts** dicount values (string of 3 float values in interval [0, 1], separated by comma, e.g. "1,1,1") for individual objectives.

### Running the project
Example run is (when current directory is the repository directory):
*python3 ./run_experiments.py --experiment_label "ml-1m-standardization-cf-dhondt2" --algorithms "exactly_proportional_fuzzy_dhondt_2" --normalizations "standardization" --weights "1.0,0.0,0.0;0.8,0.1,0.1;0.5,0.25,0.25;0.3,0.3,0.3;0.6,0.3,0.1;0.6,0.1,0.3" --train_path "C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0/train.dat" --test_path "C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0/test.dat" --diversities "cf" --baselines "MatrixFactorization;ItemKNN" --seeds "42" --shifts "0.0;-0.5;2.0" --cache_dir "C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0" --discounts "1,1,1;0.85,0.85,0.85" --mlflow_project_path . --mlflow_tracking_uri "http://127.0.0.1:5000" --output_path_prefix .*

or

Windows Powershell:<br>
*$Env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"; mlflow run --no-conda . --experiment-name "moo-as-voting-fast" -P train_path="C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0/train.dat" -P test_path="C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0/test.dat" -P weights="0.33,0.33,0.33" -P seed=42 -P normalization="cdf" -P algorithm="exactly_proportional_fuzzy_dhondt_2" -P diversity="cf" -P baseline="MatrixFactorization" -P shift=0.0 -P discounts=1,1,1*

Linux:<br>
*MLFLOW_TRACKING_URI="http://127.0.0.1:5000" mlflow run --no-conda . --experiment-name "moo-as-voting-fast" -P train_path="C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0/train.dat" -P test_path="C:/Users/PD/Downloads/filmtrust-folds/randomfilmtrustfolds/0/test.dat" -P weights="0.33,0.33,0.33" -P seed=42 -P normalization="cdf" -P algorithm="exactly_proportional_fuzzy_dhondt_2" -P diversity="cf" -P baseline="MatrixFactorization" -P shift=0.0 -P discounts=1,1,1*

**Important note**: running mlflow server is needed.<br> 
See (https://www.mlflow.org/docs/latest/tracking.html) for details
You can run it as *mlflow server* and then run the commands above. Also note that python3 command must be on PATH (having just "python" on path is not enough, mlflow will complain on Windows)


### Results
All the results can be found at the following address:

### Final note about implementation
More generic and customizable implementation can be found in the following repository https://github.com/pdokoupil/moo-as-voting It should be mentioned that the previously mentioned implementation was not used to generate the results available in the paper. Also note that although that implementation is more generic, it is significantly slower compared to the current implementation.