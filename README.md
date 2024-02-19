<h1 align="center">TSASD</h1>
<h2 align="center">
Understanding Time Series Anomaly State Detection through One-Class Classification</h2>

### Abstract

For a long time, research on time series anomaly detection has mainly focused on finding outliers within a given time series. Admittedly, this is consistent with some practical problems, but in other practical application scenarios, people are concerned about: assuming a standard time series is given, how to judge whether another test time series deviates from the standard time series, which is more similar to the problem discussed in one-class classification (OCC).
Therefore, in this article, we try to re-understand and define the time series anomaly detection problem through OCC, which we call 'time series anomaly state detection problem'.
We first use stochastic processes and hypothesis testing to strictly define the 'time series anomaly state detection problem', and its corresponding anomalies. Then, we use the time series classification dataset to construct an artificial dataset corresponding to the problem. We compile 38 anomaly detection algorithms and correct some of the algorithms to adapt to handle this problem. 
Finally, through a large number of experiments, we fairly compare the actual performance of various time series anomaly detection algorithms, providing insights and directions for future research by researchers. 

If you use TSASD in your project or research, cite the following two papers:

* [Zhou2024Understanding](https://arxiv.org/pdf/2402.02007v1.pdf)
