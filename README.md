<img alt="fixout_logo" src="https://asilvaguilherme4.files.wordpress.com/2023/08/fixout-1.png?w=128">

<b>Algorithmic inspection for trustworthy ML models</b>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/fixouttech/fixout/blob/main/LICENSE)

<ul>
  <li><a href="https://fixout.fr" target="_blank" rel="noopener">Website</a></li>
  <li><a href="https://fixouttech.github.io/fixout_api_docs" target="_blank" rel="noopener">Documentation</a></li>
  <li><a href="https://fixout.fr/blog/" target="_blank" rel="noopener">Blog</a></li>
</ul>

# Getting started

How to start analysing a simple model:


```python
from connector import FixOutConnector, FixOutArtifact

fixout = FixOutHelper("Credit Risk Assessment") 

sensitive_features = [(19,0,"foreignworker"), 
                      (18,1,"telephone"), 
                      (8,2,"statussex")] 

fxa = FixOutArtifact(model=model,
                         X_train=X_train.tolist(), 
                         y_train=y_train,
                         X_test=X_test.tolist(),
                         y_test=y_test,
                         features_name=features_name,
                         sensitive_features=sensitive_features)


```

Then run the inspection
```python
fx.load(fxa)
```

Finally, ask for a report.
```python
fx.disconnect(sendMail=True)
```


Access the generated report at <a href="http://localhost:5000" target="_blank" rel="noopener">http://localhost:5000</a> ;)
