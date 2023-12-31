# 梯度下降及其變體

梯度下降是一種最優化算法，用於尋找函數的最小值。有多種梯度下降的變體，以下是一些常見的方法：

### [標準梯度下降（Batch Gradient Descent）](https://github.com/orange1314/Gradient-Descent/tree/main/Batch%20Gradient%20Descent)

- **優點：** 全局收斂，對小批量數據進行高效處理。
- **缺點：** 對大數據集的計算效率較低。

### [隨機梯度下降（Stochastic Gradient Descent，SGD）](https://github.com/orange1314/Gradient-Descent/tree/main/SGD)
- **優點：** 每次僅使用一個樣本進行更新，計算效率較高。
- **缺點：** 較不穩定，可能會在收斂過程中波動較大。

### [小批量梯度下降（Mini-batch Gradient Descent）](https://github.com/orange1314/Gradient-Descent/tree/main/SGD)

- 綜合了標準梯度下降和隨機梯度下降的優點，使用一小批量樣本進行更新。

### [動量梯度下降（Momentum）](https://github.com/orange1314/Gradient-Descent/tree/main/Momentum)

- 引入動量項，模擬物體運動的慣性，有助於克服收斂過程中的震盪。

### [Adagrad](https://github.com/orange1314/Gradient-Descent/tree/main/Adagrad)

- 自適應學習率的方法，根據參數的更新情況調整學習率。

### [RMSprop](https://github.com/orange1314/Gradient-Descent/blob/main/RMSprop/README.md)

- 修正了Adagrad的一些問題，使學習率能夠適應不同參數的更新情況。

### [Adam](https://github.com/orange1314/Gradient-Descent/edit/main/Adam/README.md)

- 結合了動量梯度下降和RMSprop的優勢，廣泛應用且性能優越。

### [Adadelta](https://github.com/orange1314/Gradient-Descent/tree/main/Adadelta)

- 類似於RMSprop，但更進一步解決了學習率過度下降的問題。

這些梯度下降法都是為了更有效地找到函數的最小值而提出的。Adam通常被視為一種表現優秀的算法，因為它綜合了多種優勢。然而，對於特定問題，不同的梯度下降法可能會有不同的效果，需要根據實際情況進行選擇。



```python

```
