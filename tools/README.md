# 本机工具脚本（不参与容器运行）

容器内请使用 `requirements.txt` 锁定的 pytest 与 Flask；本目录脚本只依赖 Python 标准库，
方便在没有第三方包的沙箱里复现实验与验收。

- `run_tests.py`：最小测试发现/运行器，配合 `pytest_shim/` 提供 `pytest.approx`、
  `pytest.raises`、`pytest.mark.parametrize` 的等价实现，运行 `tests/` 下除 Web 交付
  （依赖 Flask）外的全部用例。
- `rebuild_sample_data.py`：读取 `sample_data/western_region_return_counter.json`，
  按称重设备台账重建设备标注、新设备称重与 `learned.weight_z`，固定随机种子，可重复执行。
- `verify_baseline.py`：只读验收，证明设备隔离、样本不足转人工、训练/在线同基线指纹、
  重算幂等与设备冲突拒绝五项性质。

运行方式：

```sh
python -I tools/run_tests.py
python -I tools/rebuild_sample_data.py
python -I tools/verify_baseline.py
```
