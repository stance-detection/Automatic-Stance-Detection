Score for fold 3 was the highest with 0.704

This is the score on the test set (using GradientBoostOptimizer). We can see that a lot of related are classified as unrelated and vice versa, which was not the case with the baseline classifier that gave a very good accuracy for classifying related or unrelated. This classifier gives a higher weight to classifying correctly in the related class.

Way ahead: Use the baseline classifier for classifying related and unrelated and then use those results to classify related news into the three classes ['agree','disagree','discuss'].

-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    199    |     3     |    155    |    405    |
-------------------------------------------------------------
| disagree  |    12     |     9     |    39     |    102    |
-------------------------------------------------------------
|  discuss  |    68     |    12     |   1061    |    659    |
-------------------------------------------------------------
| unrelated |    136    |    11     |    127    |   6624    |
-------------------------------------------------------------
Score: 2997.25 out of 4448.5	(67.37664381252108%)

