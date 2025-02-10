# ExternalExplainers
External explainer implementations for [PD-EXPLAIN](https://github.com/analysis-bots/pd-explain).\
While you can use the explainer implementations in this repository directly, it is recommended to use them through the PD-EXPLAIN library,
for a much better and more user-friendly experience.
## Included Explainers
### Outlier explainer
This explainer is based on the [SCORPION](https://sirrice.github.io/files/papers/scorpion-vldb13.pdf) paper.\
Its goal is to provide explanations for outliers in the data, explaining why a certain data point is an outlier.\
This explainer is meant to work on series created as a result of groupby + aggregation operations.\
Explainer author: [@Itay Elyashiv](https://github.com/ItayELY)
