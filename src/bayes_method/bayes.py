"""
We could instead ask the classifier
to give us a best guess about the class
and assign a probability estimate to that best guess.

Naïve Bayes
Pros: Works with a small amount of data, handles multiple classes
Cons: Sensitive to how the input data is prepared
Works with: Nominal values

rule for classifier:
* If p1(x, y) > p2(x, y), then the class is 1.
* If p2(x, y) > p1(x, y), then the class is 2.
"""
