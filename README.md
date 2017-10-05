# bayesian-psych
### A library for easy creation of bayesian decision making models and classifiers

## Availible Models:
 - Inter Temporal Choice Heuristics (ITCH)
 - Exponential Discounting
 - Hyperbolic Discounting
 - Quasi-Hyperbolic Discounting
 

## Quick Example
  Coming Soon


## Data Column Meaning 
|Column             | Meaning |
| ----------------- | -------- |
|1-p1               | one minus "more certain" outcome|
|1-p2               | one minus "less certain" outcome|
|LL                 | TRUE = choose Larger Later reward       FALSE = choose Smaller Sooner Reward|
|df_num             | convenience variable distinguish each subject, discounting, reward case|
|discount_type      | discount-type (i.e. time, probability, or effort)|
|e1                 | smaller effort level|
|e2                 | larger effort level|
|is_test            | TRUE = data is part of testing set      FALSE = data is part of training set|
|key                | participant number + discount_type + reward_type|
|participant        | participant number|
|reward_type        | reward-type (i.e. social, health, money)|
|t1                 | smaller time delay|
|t2                 | larger time delay|
|x1                 | smaller reward|
|x2                 | larger reward|
|t1_n               | smaller time delay normalized|
|t2_n               | larger time delay normalized|
|x1_n               | smaller reward normalized|
|x2_n               | larger reward normalized |

## Installation 

```bash
git clone https://github.com/mhamilton723/bayesian-psych.git
cd bayesian-psych/src
python 2_model_comparison.py
```

If you encounter errors, pip install the required packages. Run the code in the numerical order provided. You do not need to run 1_parse_raw_data.py since the full data set cannot be made publically available due to Institutional Review Board and HIPAA Privacy Rules. 


