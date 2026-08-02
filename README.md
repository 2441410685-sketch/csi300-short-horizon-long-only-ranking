# CSI300 Short-Horizon Long-Only Ranker

> 沪深300成分股短线多头选股策略：使用 186 维量价因子与 LightGBM LambdaRank，预测未来五个交易日最值得持有的 5 只股票。

## 项目简介

项目以每天的沪深300可交易成分股作为一个排序横截面。模型不预测精确收益，而是直接学习股票之间的相对优先级；信号日收盘后生成 Top5，组合等权配置，目标持有区间为下一交易日开盘至第五个交易日开盘。

完整链路包括历史成分下载、行情清洗、因子生成、时间序列回测、LambdaRank 训练和最新选股。训练、验证与测试严格按日期推进，所有随机源固定为 `2026`。

## 最近一年样本外结果

回测区间为 2025-07-29 至 2026-07-24，共 240 个信号日。

| 指标 | 结果 |
|---|---:|
| Top5 五日平均收益 | 1.656% |
| 五日口径年化夏普率 | 1.913 |
| 相对候选池平均 Alpha | 1.515% |
| Alpha 年化信息比率 | 1.955 |
| Top5 胜率 | 59.58% |
| 跑赢候选池比例 | 61.67% |

Alpha 是同一信号日 Top5 五日收益减去全部可交易候选股的等权五日收益，不是 CAPM Alpha。结果未扣除交易成本；五日标签存在重叠，因此不能把 240 个信号收益直接连乘为净值。详细方法、逐折结果和局限见 [模型项目报告](reports/model_report.md)。

## 方法概览

- 股票池：逐交易日历史沪深300成分，过滤停牌、ST 与无成交股票；
- 标签：`(open_T+5 - open_T+1) / open_T+1`；
- 因子：Alpha158 与 28 个技术指标，共 186 维；
- 模型：LightGBM `lambdarank`，每日横截面为一个 query；
- 标签编码：每日收益百分位映射为 0～9 级，指数增益为 `2^level-1`；
- 目标聚焦：`lambdarank_truncation_level=5`；
- 选轮：训练尾部 120 日作为内层验证，间隔 5 个交易日，只按验证 Top5 收益的五轮移动平均选树轮数；
- 回测：扩展训练窗口，12 折 × 20 个测试交易日，覆盖最近一年。

## 源码函数索引

以下索引覆盖 `src/csi300_ranker` 中的全部具名函数。`main()` 是对应模块的命令行入口；以下划线包围的 `evaluate` 是验证指标内部使用的闭包，不需要直接调用。

### `config.py`：项目配置

| 函数 | 作用 |
|---|---|
| `ensure_directories()` | 创建数据、因子、模型、输出和报告目录。 |

### `download.py`：数据更新

| 函数 | 作用 |
|---|---|
| `parse_args()` | 读取行情下载的开始和结束日期。 |
| `result_frame()` | 将 BaoStock 查询游标转换为 pandas DataFrame，并处理接口错误。 |
| `load_trade_dates()` | 查询给定区间内的真实开市日期。 |
| `load_membership()` | 逐交易日下载当时的沪深300成分，保留点时股票池。 |
| `load_prices()` | 为历史成分股完整下载前复权日线，保持复权快照一致。 |
| `main()` | 登录 BaoStock，依次更新交易日、历史成分和行情文件。 |

### `data.py`：数据清洗与标签

| 函数 | 作用 |
|---|---|
| `normalize_stock_id()` | 将交易所代码统一为带前导零的六位股票代码。 |
| `clean_prices()` | 转换字段类型，删除无效、重复或不满足 OHLC 关系的记录。 |
| `future_labels()` | 在完整交易日历上计算 T+1 开盘到 T+5 开盘收益。 |
| `build_samples()` | 合并历史成分、可交易状态和未来标签，生成建模样本。 |
| `main()` | 读取原始数据并写出清洗行情和样本表。 |

### `features.py`：186维因子

| 函数 | 作用 |
|---|---|
| `rolling_slope()` | 计算指定窗口内价格序列的线性回归斜率。 |
| `baseline_rsquared()` | 按冻结特征口径计算趋势拟合度。 |
| `exponential_average()` | 计算带完整暖启动窗口的指数移动平均。 |
| `relative_strength_index()` | 计算 Wilder 风格 RSI。 |
| `stochastic_kd()` | 计算 9-3-3 随机指标的 K、D 值。 |
| `average_true_range()` | 计算 Wilder 风格 ATR。 |
| `on_balance_volume()` | 按收盘涨跌方向累计成交量，得到 OBV。 |
| `engineer_stock()` | 针对单只股票计算 Alpha158 和 28 个技术指标。 |
| `build_feature_table()` | 逐股票计算因子并恢复为日期—股票排序的完整表。 |
| `main()` | 从清洗行情生成 186 维 float32 Parquet 因子缓存。 |

### `dataset.py`：模型矩阵

| 函数 | 作用 |
|---|---|
| `load_samples()` | 读取动态样本；推理模式不会加载未来标签。 |
| `load_features()` | 按固定顺序读取 186 个模型因子。 |
| `training_matrix()` | 合并指定区间内的成熟标签与同日因子。 |
| `prediction_matrix()` | 读取信号日可交易成分及其同日因子，不接触未来字段。 |
| `latest_signal_date()` | 返回当前样本表中的最新交易日。 |
| `mature_label_cutoff()` | 从信号日向前移动 5 个交易日，确定最后成熟标签日。 |

### `metrics.py`：评价指标

| 函数 | 作用 |
|---|---|
| `query_sizes()` | 统计每日股票数量，形成 LightGBM ranking group。 |
| `daily_metrics()` | 逐日计算 Rank IC、Top5 收益、候选池收益和 Alpha。 |
| `annualized_ratio()` | 将五日收益序列的均值/标准差比率年化。 |
| `summarize_daily()` | 汇总收益、夏普率、Alpha、胜率、Rank IC 和 ICIR。 |
| `lightgbm_validation_metric()` | 创建每轮训练使用的 Top5 收益与 Rank IC 回调。 |
| `lightgbm_validation_metric.<evaluate>()` | 按每日 query 边界切分预测，并返回 LightGBM 可识别的验证指标。 |

### `model.py`：LambdaRank

| 函数 | 作用 |
|---|---|
| `parameters()` | 返回确定性 LightGBM LambdaRank 参数。 |
| `relevance_labels()` | 将每日未来收益百分位映射为 0～9 级相关性标签。 |
| `ranking_dataset()` | 创建以交易日为 query 的 LightGBM Dataset。 |
| `inner_split()` | 在训练尾部留出 120 日验证集，并设置 5 日间隔。 |
| `choose_iteration()` | 仅按验证 Top5 收益的五轮移动平均选择树轮数。 |

### `train.py`：最终训练

| 函数 | 作用 |
|---|---|
| `parse_args()` | 读取信号日、训练起点、树轮数和 CPU 线程数。 |
| `fit_with_validation()` | 完成内层验证选轮，并在全部外层成熟样本上重新训练。 |
| `feature_importance()` | 将模型增益转换为按占比排序的因子重要性表。 |
| `main()` | 完成标签截断、训练、模型保存和训练清单生成。 |

### `predict.py`：最新选股

| 函数 | 作用 |
|---|---|
| `parse_args()` | 读取可选信号日，未指定时使用最新交易日。 |
| `rank_selection()` | 按分数降序选择 Top5，并生成排名和等权权重。 |
| `main()` | 加载模型执行无标签推理，写出最新多头名单。 |

### `backtest.py`：时间序列回测

| 函数 | 作用 |
|---|---|
| `parse_args()` | 读取回测线程数和最大树轮数。 |
| `fold_summary()` | 将逐日结果汇总为逐折收益、Alpha、胜率和年化比率。 |
| `main()` | 执行 12 折扩展窗口训练，生成 OOF 预测和回测报告。 |

### `utils.py`：通用工具

| 函数 | 作用 |
|---|---|
| `set_seed()` | 统一 Python 和 NumPy 随机种子。 |
| `save_csv()` | 以 UTF-8 编码稳定写出 CSV。 |
| `save_json()` | 以带缩进的 UTF-8 格式写出 JSON。 |
| `file_sha256()` | 计算模型或数据文件的 SHA-256。 |

## 快速开始

要求 Python 3.10。从项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。

完整更新与训练：

```bash
python -m csi300_ranker.download --start-date 2022-05-01
python -m csi300_ranker.data
python -m csi300_ranker.features
python -m csi300_ranker.backtest
python -m csi300_ranker.train
python -m csi300_ranker.predict
```

最新名单保存在 `outputs/latest_selection.csv`。仓库内的 `examples/latest_selection.csv` 是 2026-07-31 的可复现实例。

## Docker

```bash
docker build -t csi300-ranker .
docker run --rm -v "${PWD}:/workspace" csi300-ranker \
  python -m csi300_ranker.predict
```

## 项目结构

```text
src/csi300_ranker/   数据、因子、训练、回测与预测源码
models/              最终模型及训练口径
reports/             最近一年回测结果与模型报告
examples/            最新选股示例
data/                本地行情和建模样本（不提交大文件）
artifacts/           本地因子缓存（不提交大文件）
outputs/             每次运行产生的最新名单
tests/               标签、指标与排序单元测试
```

## 研究声明

本项目用于量化研究和工程展示，不构成投资建议。历史样本外表现不能保证未来收益，真实交易前应单独评估费用、冲击、涨跌停和容量约束。
