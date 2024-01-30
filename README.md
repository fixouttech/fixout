# fixout

The algorithmic inspection for trustworthy ML models

<div align="center">

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/fixouttech/fixout/blob/main/LICENSE)

</div>


Documentation
Website
Blog

# Getting started

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
