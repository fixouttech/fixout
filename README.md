<img alt="fixout_logo" src="https://asilvaguilherme4.files.wordpress.com/2023/08/fixout-1.png?w=100">

<b>Algorithmic inspection for trustworthy ML models</b>

<div align="center">

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/fixouttech/fixout/blob/main/LICENSE)

</div>

<ul>
  <li><a href="" target="_blank" rel="noopener">Website</a></li>
  <li><a href="" target="_blank" rel="noopener">Documentation</a></li>
  <li><a href="" target="_blank" rel="noopener">Blog</a></li>
</ul>

# Getting started

How to start analysing a simple model:


```python
 from connector import FixOutConnector, FixOutArtifact

 fx = FixOutConnector('localhost', 9090)
 fx.connect("753159","Report 1") # token and name of the report

 sensitive_features = [(19,0,"foreignworker"), 
                       (18,1,"telephone"), 
                       (8,2,"statussex")] 

 fxa = FixOutArtifact(model=model,
                      X_train=X_train, 
                      y_train=y_train,
                      X_test=X_test,
                      y_test=y_test,
                      features_name=features_name,
                      sensitive_features=sensitive_features,
                      nonnumeric_features=[])

 fx.load(fxa)
```

Then run the inspection
```python
 fx.inspect([FairMetricEnum.DP, FairMetricEnum.EO, FairMetricEnum.PE])
 fx.explain()
```
