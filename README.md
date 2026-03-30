# Objective

In this project, I'm building a real-time stock prediction dashboard utilizing Kafka for real-time processing and the Alpaca stock websocket & REST API. As an exercise, I will also be using zookeeper to better understand legacy code, as well as the control center for monitoring and schema registry for data governance to mimic a production environment.

In terms of the model, I will be using LSTM model for its ability to work well with times series data considering its long and short term memory cells.

![lstm stock demo](/imgs/lstm_stock_model_demo.gif)

## TODO:

- Make websocket_prediction_server.py have graceful shutdown when Ctrl+C
- Make script that automates running all of the scripts so don't have to run them individually
- Have this hosted (maybe using Terraform)
- Have MLFlow for testability
- Account for model drift (maybe have it so if it reaches a threshold, it triggers automated retraining)


# Layout

![layout](/imgs/lstm_stock_diagram.drawio.png)

### Model Training

1. **[download_alpaca_data]** requests training data from alpaca
    - `download_alpaca_data.py` -> `/data/raw/[TICKER]_2y_1Min.csv`
2. **[feature_eng]** creates features based on OHCLV data
    - `/data/raw/[TICKER]_2y_1Min.csv` -> `feature_eng.py` -> `/data/features/[TICKER]_features.parquet`
3. **[scaling_sequence_pipeline]** creates scalers for the LSTM model
    -  `/data/features/[TICKER]_features.parquet` -> `scaling_sequence_pipeline.py` -> `/models/scalers.pkl`, {`/data/sequences/train.npz`, `/data/sequences/val.npz`, `/data/sequences/test.npz`}
4. **[lstm_model]** creates the LSTM model's architecture
5. **[training]** trains the LSTM model, keeping track of progress via checkpoints
    - {`data/sequences/train.npz`, `data/sequences/val.npz`, `data/sequences/test.npz`} -> `training.py` -> `/models/best_model.pt`, `/checkpoints`

### Model Inference
1. **[alpaca_kafka_bridge]** connects the Alpaca websocket (producer) to the Kafka topic `stock_ohlcv`
    - `alpaca_kafka_bridge.py` (producer) -> `stock_ohlcv` (input_topic)
2. **[kafka_inference_service]** ingests from the `stock_ohlcv` topic, makes predictions, and outputs to the `stock_predictions` topic. Here is when it uses the LSTM model and scaler to make predictions. Since it ingests from a kafka topic, but also gives data to another topic, it is therefore both consumer & producer
    - `stock_ohlcv` (input_topic), `/models/best_model.pt`, `/models/scalers.pkl` -> `kafka_inference_service.py` (consumer & producer) -> `stock_predictions` (output topic)
3. **[websocket_prediction_server]** (consumer) streams the `stock_predictions` topic to `realtime_dashboard.html`
    - `stock_predictions` (output topic) -> `websocket_prediction_server.py` -> `realtime_dashboard.html`
4. **[realtime_dashboard]** provides real-time visualizations

Note:

- We employ the use of Zookeeper as the coordinator for Kafka (although nowadays there's the KRaft mode i.e. Kafka can coordinate itself, we use Zookeeper as a learning exercsie), schema registry for mangaging schemas, and control center for health monitoring. All of which aren't strictly required to use Kafka, but again, were for learning

# Running

(In the future, this may be replaced with an automated script)


### Model Training

0. Create a `.env` file and paste your `ALPACA_API_KEY=<ALPACA_API_KEY>` and `ALPACA_SECRET_KEY=<ALPACA_SECRET_KEY>`
    - run `pip install -r requirements.txt`
1. run `download_alpaca_data.py`
2. run `feature_eng.py`
3. run `scaling_sequence_pipeline.py`
4. run `training.py`
    - this file references `lstm_model.py`

### Model Inference

0. We first spin up the docker container `docker compose up -d`
    - Verify if it's healthy via `docker compose ps`
    - This starts up kafka, zookeeper, schema registry, control center
1. run `kafka_inference_service.py`
    - we run this before the first producer (`alpaca_kafka_bridge.py`) so we don't miss any data when we actually start it
2. run `websocket_prediction_server.py`
3. open `realtime_dashboard.html`
4. finally, run `alpaca_kafka_bridge.py` to begin streaming data

<br/><br/><br/>

# Contents
1. [Docker Config](#docker)
2. [Feature Engineering](#feature-engineering)
3. [Scaling features](#scaling-features)
4. [LSTM model architecture](#lstm-model)

<br/><br/><br/>

# [Docker](/docker-compose.yml)
Zookeeper
- Zookeeper official doc (https://hub.docker.com/r/confluentinc/cp-zookeeper)

- Example configuration code I based it off was [from the docker source code](https://github.com/confluentinc/cp-all-in-one/blob/8.0.0-post/cp-all-in-one/docker-compose.yml)
    - using zookeeper
    - using Schema Registry
    - using Control Center


[Full Docker Configuration Documentation](https://docs.confluent.io/platform/current/installation/docker/config-reference.html)


After setting up the configuration in the [docker yml file](/docker-compose.yml), I ran `docker compose up -d` and accessed the running system
[local confluent page spun up by docker](/imgs/1.%20confluent.png)


# [Feature Engineering](/src/training/feature_eng.py)
Here, we would like to preprocess the data as well as calculate certain metrics to increase the chances of the LSTM model being able to find patterns.

Note the type of data we will be intaking is OHLCV data (Open, High, Low, Close, Volume).

All windows are calibrated for **1-minute bars**: 60 bars = 1 hour, 390 bars = 1 full trading day.

## Price-based features
- `returns`
- `log_returns`

$$\text{returns} = (close_t - close_{t-1}) / close_{t-1}$$
$$\text{log.returns} = ln((close_t - close_{t-1}) / close_{t-1})$$

The change in closing price.

Normally, including both would be redundant and introduce multi-collinearity into the model. However, it is different here since our model is a neural network. My reasoning for this is that returns would capture short-term momentum while log of returns will capture more long-term momentum due its smoothing nature.

Additionally, if both features are close to the same price, then it provides a stronger indication.

- `intraday_range`

$$intraday.range = (high - low) / close$$

The volatility of a candlestick bar, normalized over the opening stock price of the trading day (or whatever time it's aggregated if it is). This captures the price changes in the market in an efficient way for the model.

- `gap`

$$(open_t - close_{t-1}) / close_{t-1}$$

The volatility of the entire trading day, normalized over the ending stock price of the day. This is useful for capturing market overnight/pre-market information (e.g. Apple bankrupts overnight, then price change will be reflected at the start of the trading day). At 1-minute resolution this feature is only non-zero on the first bar of each session.

- `close_position`

$$(close_t - low_t) / (high_t - low_t)$$

The interval scale of the bar between $[0,1]$, where it's bearish when closer to $0$, and bullish for $1$. The closing price of a stock doesn't change unless it is the lowest price, so this metric takes advantage of that to capture the sort of "bearishness" of the market.


## Technical indicators

This is typically done on a trading day scale than a candlestick bar scale. Windows are scaled to preserve their intended time horizons at 1-minute resolution.

- `sma_windows`
    - `close_to_sma`
    - `sma_60_390_cross`

SMA windows sum up the n bars before it to get an average for a window/interval of n bars, showing any small trends that may be appearing. Note this uses the closing price. Windows used: **5** (5 min), **15** (15 min), **60** (1 hour), **390** (1 full trading day).

The `close_to_sma` feature just shows how far the current stock's price is from the $SMA_n$ average, normalized as a percentage.

The `sma_60_390_cross` is the intraday equivalent of the golden/death cross, with a formula of $SMA_{60} / SMA_{390} - 1$. This metric accounts for intraday regime changes where when $SMA_{60} > SMA_{390}$ it indicates the last hour is running above the daily average (bullish), while vice versa is bearish.

- `ema_windows`

$$
\alpha * close_t + (1-\alpha) * EMA_{t-1} \ \ \text{ where }\ \  \alpha = 2/(n+1)
$$

Similar to SMA, except it places higher emphasis on more recent prices. Windows: **12** (12 min), **26** (26 min), **60** (1 hour).

- `rsi_window`

$$RS = \text{Average.Gain}_n / \text{Average.Loss}_n$$
$$RSI = 100 - (100 / (1 + RS))$$

The Relative Strength Index (overbought if more than 70, underbought if less than 30).

- `macd_windows`

$$\text{MACD.Line} = EMA_{12} - EMA_{26}$$
$$\text{Signal.Line} = EMA_9$$
$$\text{Histogram} = \text{MACD.Line} - \text{Signal.Line}$$

The MACD line uses a smaller time frame EMA and a larger time frame EMA and subtracts them to see if there is positive momentum, i.e. the smaller time frame EMA average is larger.
This is compared against an EMA (usually EMA_9), which serves as the "standard" average, or the signal line.
Under the the histogram, once there is enough positive momentum, MACD line will be greater than the signal line; indicating there is considerable positive momentum.

- `bb_width`
- `bb_position`

$$\text{middle.Band} = SMA_{60}$$
$$\text{upper.Band} = \text{middle.Band} + 2\sigma$$
$$\text{lower.Band} = \text{middle.Band} - 2\sigma$$

$$\text{bb.Width} = (bb.Upper - \text{bb.Lower}) / \text{bb.Middle}$$
$$\text{bb.Position} = (close - \text{bb.Lower}) / (\text{bb.Upper} - \text{bb.Lower})$$

The Bollinger bands (bb) are based on the normal distribution property that 95% of data falls within 2 standard deviations of the mean, and uses this fact to measure recent volatility by noting that any stock price outside of the 2 standard deviations likely indicates momentum. The bb width shows the average of this range, while the bb position is the normalized location of the current price relative to said range, where it's bullish when greater than 1 and bearish when below 0.


## Volatility features  
Along with returns and momentum, we want to measure how volatile the stock is.

- `volatility_windows`

$$
\sigma_{\text{historical}} = std(\text{log.Returns}_n) * \sqrt{k}
$$

This metric looks at the volatility of the returns and does so by looking at the standard deviation for k periods, where $k$ = number of trading periods in a year, as we want to see the volatility of the returns annualized. For 1-minute bars, $k = 252 \times 390 = 98,280$ trading minutes per year. Windows: **15, 30, 60** minutes.

- `parkinson_vol`

$$
\sqrt{[\frac{1}{4ln(2)}] * \frac{1}{n}[\sum ln(\frac{high_i}{low_i})^2]} * \sqrt{k}
$$

The Parkinson volatility estimator measures volatility with 5x more accuracy due to it using the High and Low metric for more information, as well as it using a scaling factor based on Brownian motion ($\frac{1}{4ln(2)}$); all annualized with $k = 98,280$. Windows: **15, 30, 60** minutes.

- `gk_vol`

$$
\sqrt{\frac{1}{n}\sum[ [0.5 ln(\frac{high}{low})^2] - [2ln(2)-1] * [ln(\frac{close}{open})^2] ]} * \sqrt{k}
$$

The Garman-Klass volatility estimator extends upon the Parkinson volatility estimator and uses even more metrics (OLHC) to get more information, as well as a scaling factor; all annualized with $k = 98,280$. Window: **60** minutes.


## Volume features

Along with the volatility of the stock price, we also want to keep track of the weight of the stock i.e. its volume.

- `volume_ma`
The moving average for the volume for the past n minutes. Windows: **15, 30, 60** minutes.

- `relative_volume`

$$
\text{volume}_t / \text{MA.Volume}_n
$$

The relative volume metric: the percentage of the current stock's volume over the moving average of the volume.

- `price_to_vwap`

$$VWAP = \sum(\text{typical.Price}_i * \text{volume}_i) / \sum(\text{volume}_i)$$
$$\ \ \text{ where }\ \  \text{typical.Price} = \frac{1}{3}[\text{high} + \text{low} + \text{close}]$$

The Volume Weighted Average Price learns during a trading day and works by taking the average of the typical candlestick bar's price, weighted by cumulative volume traded for the current trading day. The reason for the cumulative volume is to capture **inertia**, where stronger inertia means stronger bullish/bearish movements instead of just volatile sideways movement.

At the beginning of a trading day, volatility is high and inertia is lowest as the price is uncertain. However as time goes on, the stock price becomes more certain, and inertia is stronger; something that institutional investors typically use to their advantage to buy/sell when the stock price is below/above vwap.

VWAP resets at market open every day (cumulative daily reset), which is what institutional traders actually use as a benchmark. At 1-minute resolution this feature is particularly meaningful: `price_to_vwap` is noisy early in the session due to low accumulated volume, but becomes one of the strongest signals in the feature set by midday.

- `obv_divergence`

$$OBV_t = OBV_{t-1} + \text{sign}(\text{close}_t - \text{close}_{t-1}) * \text{volume}_t$$
$$OBV.Divergence = (OBV_t - MA.OBV_m)/OBV_t \ \ \text{ where m is the sliding window of the OBV moving avg}$$

The On-Balance Volume divergence indicator learns during a trading day by measuring the momentum of a candlestick bar price via comparing the previous OBV value to the volume-weighted change in candlestick bar prices. Note that the $\text{sign}$ is positive if $Close_t > Close_{t-1}$, and negative otherwise.

After getting $OBV_t$, it then compares the percentage difference between $OBV_t$ and OBV moving avg. The idea is that volume indicates stock price, and so if the volume of the current stock price balloons such that the  $\text{sign}(\text{close}_t - \text{close}_{t-1}) * \text{volume}_t$ portion of the equation is large, then the OBV divergence will show a larger percentage difference; signalling that the stock has significant volume backing its price. MA window: **60 bars**.

- `money_flow_ratio`

$$\text{money.flow} = \text{close} * \text{volume}$$
$$\text{money.flow.ratio} = MA(\text{money.flow})_{60} / MA(\text{money.flow})_{120}$$

The money flow ratio takes the moving average of the stock value (close * volume) for n bars, over the same moving average but with double the window, essentially comparing the current hour's flow to the prior two hours.

- `vw_return`
$[(close_t - close_{t-1}) / close_{t-1}] * volume_t$

Returns percentage difference but weighted by volume.

## Temporal features
Market prices can vary depending on the time of the day, week, season.
In order to account for that, we also include features regarding the time/date.

- `hour_sin`, `hour_cos`

$$\text{hour.sin} = sin((2\pi * hour) / 24)$$
$$\text{hour.cos} = cos((2\pi * hour) / 24)$$

When incorporating time, we use sine and cosine as to represent time wrapping around because for e.g. hour=23 and hour=0 should be 1 hour away from each other, not 23 hours away from each other.

- `minute_sin`, `minute_cos`

$$\text{minute.sin} = sin((2\pi * minute) / 60)$$
$$\text{minute.cos} = cos((2\pi * minute) / 60)$$

At 1-minute resolution, every bar has a distinct minute value, so the model can now learn intraday patterns at the minute level — for example, that the `:00` bar (top of the hour) tends to have a volume spike, or that the `:30` bar is typically quieter.

- `dow_sin`, `dow_cos`

$$\text{dow.sin} = sin((2\pi * dow) / 7)$$
$$\text{dow.cos} = cos((2\pi * dow) / 7)$$

We apply the same but for the day of the week with its 7 total days.

- `dom_sin`, `dom_cos`

$$\text{dom.sin} = sin((2\pi * dom) / 31)$$
$$\text{dom.cos} = cos((2\pi * dom) / 31)$$$


We apply the same but for the day of the month with its 31 total days, which although can differ depending on month and if its a leap year such that it can be $30$ or $28$ (if February), it doesn't matter much as the model will still learn that larger values in $\text{dom.sin}$ and $\text{dom.cos}$ tend to correlate with the end of the month.

- `is_market_open`
- `is_opening_30min`
- `is_closing_30min`

Boolean flags for if the bar falls during open market hours, the opening 30 minutes (9:30–10:00), and/or the closing 30 minutes (15:30–16:00). At 1-minute resolution the model sees all 30 bars of the opening and closing windows individually, letting it learn the characteristic volatility patterns of each session boundary.


## Lag features for LSTM

Including the lagged features provides a good way for the model to remember certain data more clearly, and force it to place higher emphasis on recent data rather than older and staler data.

- `return_lag` (lags 1, 3, 5 minutes)
- `rel_volume_lag` (lag 1 minute)
- `rsi_lag` (lag 1 minute)

The return, relative volume, and rsi metrics lagged by n bars.

<br/>

When ran, this file creates the features based on the data given and produces parquet file(s) in the [/data/raw] folder.



# [Scaling features](/src/training/scaling_sequence_pipeline.py)

This file references the parquet file(s) containing the features in the [/data/raw] folder

<br/>

In order to feed our features into our model, we need to scale them such that they're normalized to prevent any one feature from dominating the output due their units being inherently larger.

Here we split before scaling to ensure no data leakage (i.e. test data accidentally leaks into train data via scaling), then use Robust Scaler to scale instead of say Standard or MinMax Scaler as financial data can have fat tails, so it needs to be robust to outliers like Robust Scaler which uses the IQR.

Since this is a times series prediction model that predicts per minute, we segregate the data into sliding windows of 60 data points to feed as input.

<br/>

This file outputs as npz files (`train.npz`, `val.npz`, `test.npz`).


# [LSTM model](/src/training/lstm_model.py)

Note that we work with financial data which may be volatile, and thus our model must be able to generalize and robust to outliers.

## Architecture

For this LSTM, we set up its layers as:

- LSTM Layer 1
- Dropout
- LSTM Layer 2
- Attention Layer
- Layer Normalization
- Fully Connected (FC) Layer 1
- ReLU Activation funct
- Dropout
- Fully Connected (FC) Layer 2

### Reasoning

### LSTM layers

In financial trading data, there are the short-term fluctuations, the noise. However, these small waves generally follow the long-term fluctuations, i.e. the structural changes in the market. This leads us to having only 2 hidden layers (`num_layers`): the first is for small swings, and the second is for larger trends. 

To account for complex patterns, we choose a `hidden_size` of `128` as middle ground where a `<=64` size may have too much bias while a `>=1024` size may have too much variance. Although, due to the volatile nature of stocks, a `hidden_size` of `64` is also good.

### Dropout layer

Due to our `hidden_size` choice and the volatile nature of stocks, we require a `dropout` rate which during training, randomly shuts down a percentage of the neurons for each layer to make it more robust. We set a moderate `dropout` rate of $20\%$, however if we later notice too much variance, we can increase this percentage.

This dropout layer is placed in between the LSTM layers so that e.g. layer 2's cells intuitively learn the patterns from layer 1 from different angles, as it's forced to with limited cells to work with after dropout.

### Attention layer

We could simply end it here and get the output vector, however since the `hidden_size` is so big, and we have many features (+$60$), the weights may get diluted by the time it reaches the output layer; e.g. bar 1's influence on the features in the output layer is smaller than the most recent bar. This is because it's an LSTM architecture, whose very nature is sequential, using short-term and long-term memory in this design.

We can combat this by using a Multi-headed Attention layer with Query, Key, Value parameters (seen in Transformers) to see the which feature is more similar to which; and provides stronger weighting to the one that's most popular. The term "similar" being different in each attention head, where one can focus on volatility spikes, another on trend reversals, etc. 

### Layer Normalization

After going through the various layers, each with their weights and biases and non-linear functions, the values in the output layer begin to change. Therefore, we scale it within the neural network by using normalization.

We use `layer_norm` instead of `batch_norm` since it requires looking across all layers to normalize a weight, versus `layer_norm` normalizing its weights within their own layer. This is better since our layers can depend on each other, so normalizing across layers can mix up patterns (i.e. the short-term patterns of layer 1 with the long-term patterns of layer 2).

### Fully Connected (FC) Layers

As mentioned earlier, the LSTM architecture is sequential in nature due to its use of short and long-term memory cells, which are repeatedly tuned via the input cells, weights and biases. This allows it to sequentially learn patterns.

However, this does not predict a one stock price like we want, so to finally do this, we use Fully Connected layers a.k.a the Dense layers; which we commonly saw in those beginner Neural Network diagrams.

We map it with the first FC layer, introduce non-linearity via ReLU activation function, sneak in another Dropout layer, then pass it onto the second FC layer for the final output (whose patterns are strengthened/generalized due to the Dropout layer).


# [Training the model](/src/training/training.py)

We include checkpoints for when the training file runs this script, we don't have to restart the process in case of inteference.



# [Alpaca kafka connection to websocket](/src/alpaca_kafka_bridge.py)


# [Kafka inference](/src/kafka_inference_service.py)


# [Websocket connection for dashboard](/src/websocket_prediction_server.py)





