# 随机性分析

本文件夹包含圆形钢散射体/环氧基体声子晶体随机性分析所用的代码、
结果文件和论文图像。材料属性随机性采用直接抽样计算；几何随机性采用
节点法描述圆形边界的随机扰动。

## 文件内容

- `code/node_pwe_repro.py`  
  节点法 PWE 求解器及结构因子计算函数。

- `code/circle_material_oat_direct_uncertainty_order10.py`  
  材料属性单因素随机性分析。每次只改变一个材料属性，其余三个属性
  保持标称值不变；每组 300 个样本均直接调用节点法 PWE 计算，
  不使用插值。

- `code/circle_material_sobol_uncertainty_order10.py`  
  材料属性 Sobol 全局灵敏度分析。四个材料输入因子同时在
  \(\pm5\%\) 范围内变化，采用 Saltelli 型采样设计，并直接调用
  节点法 PWE 计算一阶和总阶 Sobol 指数。

- `code/circle_geometry_uncertainty_order10.py`  
  几何随机性分析。通过平滑径向 Fourier 扰动生成非规则圆形边界，
  并用节点法离散为多边形。

- `code/circle_uncertainty_order10.py` and
  `code/circle_material_oat_uncertainty_order10.py`  
  材料随机性与几何随机性分析中复用的辅助脚本。

- `figures/material_oat_uncertainty_order10_n300.pdf`  
  四个材料属性 \(E_s\)、\(\rho_s\)、\(E_e\)、\(\rho_e\) 的随机性分布图。

- `figures/material_sobol_uncertainty_order10_n1024.pdf`  
  四个材料属性对下带隙边界、上带隙边界和带隙宽度的一阶与总阶
  Sobol 灵敏度指数。

- `figures/geometry_uncertainty_examples.pdf`  
  标称圆形、5% 几何扰动和 10% 几何扰动示意图。

- `figures/geometry_uncertainty_order10_n2000.pdf`  
  5% 和 10% 几何扰动下的带隙边界与带宽分布图。

- `results/*.json` and `results/*.csv`  
  随机性分析的统计结果汇总和 Sobol 设计点逐样本计算结果。

## 复现说明

代码使用 Python 3.11 运行，依赖：

```text
numpy
scipy
matplotlib
```

材料属性随机性分析可在仓库根目录运行：

```powershell
py -3.11 ".\随机性分析\code\circle_material_oat_direct_uncertainty_order10.py" --samples 300 --workers 6
```

材料属性 Sobol 全局灵敏度分析可运行：

```powershell
py -3.11 ".\随机性分析\code\circle_material_sobol_uncertainty_order10.py" --base-samples 1024 --workers 10
```

5% 几何随机性分析可运行：

```powershell
py -3.11 ".\随机性分析\code\circle_geometry_uncertainty_order10.py" --samples 2000 --workers 6 --rel-bound 0.05 --coefficient-sigma 0.08
```

10% 几何随机性分析可运行：

```powershell
py -3.11 ".\随机性分析\code\circle_geometry_uncertainty_order10.py" --samples 2000 --workers 6 --rel-bound 0.10 --coefficient-sigma 0.16
```
