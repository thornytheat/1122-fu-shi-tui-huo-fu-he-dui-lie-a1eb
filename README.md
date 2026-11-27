# 服饰退货复核队列

面向退货质控的本地工作台。内置业务样例覆盖服饰退货验包、概率校准与限额人工复核。样例采用确定性构造，适合复现实验和核查边界行为。

## 启动

```sh
docker compose up --build -d
```

打开 `http://127.0.0.1:18201/` 查看退货复核队列。左侧六个入口可展开运行证据。调整「复核名额（单）」，点击「分配复核」比较处置结果。结束后执行 `docker compose down --remove-orphans`。

## 独立业务模块

- 验包流水 `parcel_scan`
- 换店去重 `shop_identity`
- 退货特征 `return_signals`
- 验包评分 `inspection_score`
- 复核配额 `desk_quota`
- 退货复盘 `return_replay`

## 查看运行结果

- `GET /api/overview` 返回默认运行及其指纹。
- `POST /api/scenario` 接收 `{"capacity": 3}`。入口接受 1–100 的整数，应用样例时使用该业务额度。
- `GET /api/stages/return_replay` 查看该环节的独立证据，支持 `?capacity=3`。

```sh
docker compose exec -T app python -m returnscope.delivery evaluate --capacity 3
docker compose exec -T app pytest -q
```

数据和模型约定见 [业务契约](docs/SPEC.md)。模块、例外规则与验证入口见 [业务深度](docs/DEPTH.md)。称重设备更换后的基线重估步骤见 [值班操作](docs/RUNBOOK.md)。

## 称重误差基线

- 误差基线按「门店称重设备 × 商品组」建立，设备服役区间登记在样例顶层 `scales` 台账；旧设备退役后历史样本只保留在原设备编号下，不进入新设备基线。
- 新设备某商品组样本不足最小样本量时，包裹以 `INSUFFICIENT_DEVICE_SAMPLES` 转人工，不借用其他门店、旧设备或全局样本。
- 训练向量与在线评分共用同一 `baseline_id`，指纹不一致会抛出 `WEIGHT_BASELINE_DRIFT_BETWEEN_TRAIN_AND_SERVE`。
- `python -I tools/verify_baseline.py` 可复核设备隔离、人工兜底、训练/在线同口径与重算幂等；`python -I tools/rebuild_sample_data.py` 确定性重建样例数据。
