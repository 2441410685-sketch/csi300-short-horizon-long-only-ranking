# 数据目录

仓库不提交大体积行情和因子缓存。运行 `download`、`data` 与 `features` 后会依次生成：

- `raw/hs300_membership_daily.csv`：逐交易日的历史沪深300成分；
- `raw/hs300_daily_raw.csv`：历史成分股后复权日线；
- `processed/prices_daily_clean.csv`：清洗后的日线；
- `processed/samples_with_label.csv`：动态成分、可交易标记和成熟标签；
- `../artifacts/features_float32.parquet`：186 维模型因子。

数据更新采用前复权完整刷新，目的是保持复权尺度一致；历史成分按日期查询，避免用当前成分名单回填过去。
