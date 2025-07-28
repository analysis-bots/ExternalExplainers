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
### MetaInsight explainer (Beta)
This explainer is based on the [MetaInsight](https://dl.acm.org/doi/abs/10.1145/3448016.3457267?casa_token=QWDjnCLOY3AAAAAA:sMFcURRijjH_1aDGzOkwspGJKANrnJWA5-uZNipzI_lh719s_uv9MDzg1H9UXJKIVRW0Q6_p7a-6qw) paper.\
Its goal is to find the most interesting patterns in the data, displaying them and the exceptions to those patterns in a user-friendly way.\
This explainer can work on any multidimensional data.\
Please note that this explainer is still in beta, and may not work as expected, and may be subject to changes in the future.\
Explainer author: [@Yuval Uner](https://github.com/YuvalUner)
