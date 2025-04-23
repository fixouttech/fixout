<a href="http://fixout.fr"><img alt="fixout_logo" src="https://asilvaguilherme4.files.wordpress.com/2023/08/fixout-1.png?w=128"></a>

<b>Algorithmic inspection for trustworthy ML models</b>

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<ul>
  <li><a href="https://groups.google.com/g/fixout" target="_blank" rel="noopener">Community</a></li>
  <li><a href="https://fixouttech.github.io/fixout_api_docs" target="_blank" rel="noopener">Documentation</a></li>
  <li><a href="https://fixout.fr/blog/" target="_blank" rel="noopener">Blog</a></li>
</ul>


# Install

Install the latest version of FixOut from PyPI using 

```shell
pip install fixout
```


# Getting started

How to start analysing a simple model (let's say you have trained a binary classifier on the [German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)):


```python
from fixout.artifact import FixOutArtifact
from fixout.runner import FixOutRunner

fxo = FixOutRunner("Credit Risk Assessment (German Credit)") 

# Indicate the sensitive features
sensitive_features = ["foreignworker","statussex"] 

# Create a FixOut Artifact with your model and data
fxa = FixOutArtifact(model=model,
                      training_data=(X_train,y_train), 
                      testing_data=[(X_test,y_test,"Testing")],
                      features_name=features_name,
                      sensitive_features=sensitive_features,
                      dictionary=dic)
```

## Using a Jupyter Notebook

Then run the inspection with the method `runJ`
```python
fxo.runJ(fxa, show=False)
```

You can now check the calculated fairness metrics by using the method `fairness`.

```python
fxo.fairness()
```

![Fairness metrics](/img/fair_metrics.png)

## In your quality management code
 
If you prefer to integrate FixOut in your code to check, then run the inspection by calling `run`
```python
fxo.run(fxa, show=True)
```

In this case, you can access the generated dashboard at <a href="http://localhost:5000" target="_blank" rel="noopener">http://localhost:5000</a> ;)

You should be able to see an interface similar to the following 

![FixOut interface](/img/interface_data_2.PNG)
