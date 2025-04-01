<img alt="fixout_logo" src="https://asilvaguilherme4.files.wordpress.com/2023/08/fixout-1.png?w=128">

<b>Algorithmic inspection for trustworthy ML models</b>

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<ul>
  <li><a href="https://fixout.fr" target="_blank" rel="noopener">Website</a></li>
  <li><a href="https://fixouttech.github.io/fixout_api_docs" target="_blank" rel="noopener">Documentation</a></li>
  <li><a href="https://fixout.fr/blog/" target="_blank" rel="noopener">Blog</a></li>
</ul>

# Getting started

How to start analysing a simple model (let's say you have trained a binary classifier on the [German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)):


```python
from fixout.artifact import FixOutArtifact
from fixout.runner import FixOutRunner

fxo = FixOutRunner("Credit Risk Assessment (German Credit)") 

# Indicate the sensitive features
sensitive_features = ["foreignworker","telephone","statussex"] 

# Create a FixOut Artifact with your model and data
fxa = FixOutArtifact(model=model,
                      training_data=(X_train,y_train), 
                      testing_data=[(X_test,y_test,"Testing")],
                      features_name=features_name,
                      sensitive_features=sensitive_features,
                      dictionary=dic)
```

Then run the inspection
```python
fxo.run(fxa, show=True)
```

Finally, you can access the generated dashboard at <a href="http://localhost:5000" target="_blank" rel="noopener">http://localhost:5000</a> ;)

You should be able to see an interface similar to the following 

![FixOut interface](/img/interface_data.PNG)
