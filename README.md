# LAMMPS Data Viewer v2

`lammps_data_viewer_v2.py` 是一个面向 LAMMPS data 文件的桌面可视化与轻量编辑工具，主要用于原子、键、角、二面角和 improper 的查看、选择、筛选、修改和另存。

## 启动

```powershell
python .\lammps_data_viewer_v2.py
```

也可以将 `.data/.dat/.lmp/.txt` 格式的 LAMMPS data 文件拖拽到窗口中打开。

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

如果只运行源码，通常需要 `numpy`、`PySide6`、`pyvista`、`pyvistaqt`、`vtk`。如果需要导入力场 Excel 或打包 exe，还需要 `openpyxl` 和 `pyinstaller`。

## 打包 exe

项目提供 onedir 打包脚本：

```powershell
.\build_viewer_onedir.ps1
```

生成结果位于：

```text
dist\LammpsDataViewer\LammpsDataViewer.exe
```

建议分发整个 `dist\LammpsDataViewer` 文件夹，不建议使用 PyInstaller `--onefile`，因为本程序依赖 PySide6、VTK、PyVista，onefile 启动通常更慢。

## 主要功能

- 读取 LAMMPS data 文件中的 `Masses`、`Atoms`、`Bonds`、`Angles`、`Dihedrals`、`Impropers`、`CS-Info` 等信息。
- 使用球棍模型显示原子和键，并支持盒子显示、in-cell/default 显示模式。
- 支持 atom、bond、angle、dihedral、improper 的点选、查找、隐藏、隔离、删除和 type 修改。
- 支持 atom properties 面板中修改 charge、mass、Name/ff-type、颜色等 type 级属性。
- 支持 type 统计表中双击选择指定 type，`Ctrl + 双击` 追加选择，`Shift + 双击` 范围选择。
- 支持添加 shell 原子、生成 core-shell bond、写入 `CS-Info`，并支持 core/shell 拓扑替换。
- 支持基于球坐标插入单个 atom，或复制已有 fragment 插入新的分子/片段。
- 支持调用 LAMMPS 自带 `msi2lmp.exe` 将 Materials Studio 导出的 `.car/.cor/.mdf` 转换为 data 后直接可视化。
- 支持同时打开多个模型窗口，平铺窗口，以及建立同步组执行部分 type 级批量操作。
- 另存为前检查总电荷，关闭程序前提示是否保存。

## 鼠标与快捷键

- 左键单击：选择 atom 或 bond。
- 左键拖动：框选原子。
- 右键拖动：旋转模型。
- 中键拖动：平移模型。
- 滚轮：缩放模型。
- `Alt + 左键双击 atom`：选择当前显示的同元素原子。
- `Alt + 左键双击 bond`：选择当前显示的同 type bond。
- `Ctrl + F`：打开查找窗口。
- `Delete`：删除当前选中的 atom/bond/angle/dihedral/improper。

## 插入原子与 Fragment

插入操作需要先点选一个原子作为局部球坐标原点，然后使用：

```text
选择 -> 插入原子...
选择 -> 插入 fragment...
```

球坐标定义：

- `r`：距离原点的长度。
- `theta`：相对 `+Z` 轴的极角，`theta=0` 表示沿 `+Z`。
- `phi`：XY 平面内从 `+X` 指向 `+Y` 的方位角。

插入原子时，可以从当前模型已有 atom type 中选择一个 type。新原子的 charge 默认取该 type 中已有原子的代表电荷；如果当前 atom style 有 molecule-ID，则自动分配新的 molecule-ID。

插入 fragment 时，程序从当前模型已有 fragment 中选择一个进行复制，把该 fragment 的质心移动到球坐标目标点，并复制 fragment 内部的 bond/angle/dihedral/improper 拓扑信息。

## Materials Studio 导入

可以通过以下入口导入 Materials Studio 导出的结构文件：

```text
文件 -> 导入 Materials Studio car/cor...
```

也可以直接拖拽 `.car/.cor/.mdf` 文件到界面。程序会调用 LAMMPS 自带的 `msi2lmp.exe` 转换出 LAMMPS data 文件，然后自动加载可视化。

首次使用时需要选择 `msi2lmp.exe` 路径。程序会记住该路径。转换时会自动把同名 `.car/.mdf/.cor` 文件复制到 `msi2lmp.exe` 所在目录，并在该目录下运行 `msi2lmp.exe`，避免环境未配置时读不到程序目录外的输入文件。拖拽同一模型的 `.car` 和 `.mdf/.cor` 时，程序会自动去重，只转换并打开一次。

如果需要修改 `msi2lmp.exe` 路径或命令模板，使用：

```text
文件 -> Materials Studio 转换设置...
```

转换参数使用模板形式，默认值为：

```text
{stem} -class I -frc cvff -i >data.dat
```

其中 `{stem}` 表示所选文件的同名主体。例如选择 `model.car` 时，`{stem}` 会替换为 `model`。不同 LAMMPS 版本和不同力场可能需要不同参数，运行前可以在弹窗中修改。`msi2lmp` 通常要求同一目录下存在同名 `.car` 和 `.mdf/.cor` 文件。

这里的 `-i` 对当前测试使用的 LAMMPS `msi2lmp` 很关键，缺少它时 stdout 可能只包含日志/错误信息，而不是可加载的 LAMMPS data 内容。`>data.dat` 表示把转换结果写入工作目录下的 `data.dat`。

如果参数模板里没有写 `>` 重定向，程序会单独询问输出 data 文件名模板，默认值为：

```text
data.dat
```

此时程序会自动把 `msi2lmp` 的 stdout 写入这个输出文件。例如参数写为 `{stem} -class I -frc cvff -i`，输出文件名写为 `data.dat`，等价于在命令行中执行：

```text
msi2lmp.exe {stem} -class I -frc cvff -i >data.dat
```

如果需要模拟命令行重定向，也可以在模板中写：

```text
{stem} -class I -frc cvff -i >data.dat
```

程序会把 `msi2lmp` 的 stdout 写入 `data.dat`。自动加载前会先检查候选文件是否包含 LAMMPS data 的基本结构；如果 `data.dat` 只是日志或不可读取，程序会继续扫描 `{stem}.data`、`{stem}.dat`、`data.data`、`data.dat` 以及本次转换新生成的其他 `.data/.dat` 文件。识别到可用 data 文件后，程序会把它复制回原始 `.car/.mdf/.cor` 所在目录，并加载复制后的文件。

当 `msi2lmp` 在工作目录中生成通用文件名 `data.dat` 或 `data.data` 时，程序复制回原目录时会改名为 `{stem}.dat` 或 `{stem}.data`，避免批量转换多个模型时互相覆盖。

如果 `msi2lmp` 返回非零错误码，但目录中已经生成了 `.data/.dat` 文件，程序会弹出警告并继续加载生成的 data 文件。常见输出名包括 `{stem}.data`、`{stem}.dat`、`data.data` 和 `data.dat`。

## 查找语法

查找窗口中的 type 字段支持多 type 表达式：

- `1,2,5`：选择 type 1、2、5。
- `1， 2， 5`：中文逗号和空格会被自动忽略。
- `1~3`：选择 type 1 到 type 3。
- `1,3~5`：选择 type 1、3、4、5。

## 保存说明

另存为时会把当前编辑后的 atom、bond、angle、dihedral、improper、Masses、系数注释和 `CS-Info` 写回 data 文件。保存前会检查所有原子的 charge 总和，如果模型未保持电中性，会弹出提醒。

## 多模型与同步组

程序支持同时打开多个 data 文件。一次选择或拖入多个 data 文件时，当前空窗口会打开第一个模型，其余模型会在独立窗口中打开。也可以使用：

```text
文件 -> 在新窗口打开...
文件 -> 平铺已打开窗口
```

需要对多个相似模型进行批量操作时，先使用：

```text
文件 -> 同步组管理...
```

同步组会检查被勾选模型的 type 编号、atom type 元素和 Name、bond/angle/dihedral/improper type 的 Name 是否一致。检查通过后，以下 type 级操作会同步到组内其他模型：

- atom type 的 `Name`、`ff-type`、`mass`、`color`。
- bond/angle/dihedral/improper type 的 `Name`、`ff-type`。
- 按当前 `ff-type` 赋予同一个力场参数。
- 选择完整 type 后修改 type 编号。
- 选择完整 type 后删除 atom/bond/angle/dihedral/improper。
- 对完整 atom type 添加 shell 原子。

同步组不按具体 atom ID 同步。修改 type 或删除时，只有当前选择覆盖完整 type，才会同步到其他模型；如果只是点选或框选了局部 atom/record，则只处理当前模型，防止误操作。

## 力场模块设计约定

后续力场模块建议优先支持 Excel 参数表，而不是直接以 `.frc` 作为主格式。Excel 更容易由课题组维护，也便于人工检查和修改。程序内部可以把 Excel 读取后转换为统一的数据结构，再应用到 LAMMPS data 文件。

### 推荐 Excel 工作表

#### `L-J`

用于定义 atom type 的非键参数和电荷，推荐列：

| 列名 | 含义 |
| --- | --- |
| `Species` | 参数说明，例如 `Bridging oxygen` |
| `Symbol` | 力场 symbol，例如 `ob`、`obts`、`h*` |
| `Charge` | 原子电荷 |
| `Do` | Lennard-Jones 势阱深度 |
| `Ro` | Lennard-Jones 距离参数 |
| `σ` | sigma 参数 |

程序中建议将 `Symbol` 对应到 atom type 的 `ff-type`，将用户自定义标签单独保存为 `Name`。

### 力场导入与力场库

程序支持通过菜单或拖拽导入力场文件：

- `.yaml/.yml`：标准力场库，直接读取。
- `.json`：标准力场库，直接读取。
- `.xlsx`：Excel 力场表，读取后转换为程序内部力场结构。

拖拽文件时，`.yaml/.yml/.json/.xlsx` 会被识别为力场文件；其他文件仍按 LAMMPS data 文件打开。

导入力场后，程序会询问是否保存到程序力场库。用户可以自定义一个力场库根目录，程序会自动在该目录下创建：

```text
forcefields/
```

所有需要长期保留的力场都会统一保存为 YAML 文件。这样 Excel 可以继续作为易编辑格式，而 YAML 作为稳定的标准力场库格式，方便下次直接从程序中加载。

菜单路径：

```text
力场 -> 导入力场文件...
力场 -> 设置力场库文件夹...
力场 -> 已保存力场
```

### 候选参数、映射和赋予力场

导入力场后，可以通过以下入口使用力场参数：

```text
力场 -> 候选参数表...
力场 -> 导入 ff-type 映射...
力场 -> 赋予力场参数...
```

也可以在右侧 `type 统计` 表中双击 `ff-type` 单元格，打开当前对象类型的候选参数表，并将选中的力场 `symbol` 写入该 type 的 `ff-type`。

`导入 ff-type 映射...` 支持简单的文本映射文件，格式为：

```ini
Hw*=h*
Ow*=o*
Si_surface=st
```

左侧是当前模型中的 `ff-type/Name` 模式，右侧是力场文件中的 `symbol`。`*` 表示任意文本。

`赋予力场参数...` 会按当前 type 的 `ff-type` 到已加载力场中查找同名 `symbol`，并写入：

- atom：`charge`、`mass` 和 `Pair Coeffs`。
- bond：`Bond Coeffs`。
- angle：`Angle Coeffs`。

当前第一版稳定支持 ClayFF 这类 `lj/cut/coul/long` atom 参数，以及 harmonic bond/angle 参数。未匹配或暂不支持的参数会在完成后弹窗列出，原有模型数据会保留。

#### `Bond+Angle`

用于定义 bond 和 angle 参数。当前推荐使用分段格式：

- `Bond stretch` 段：`Species i`、`Species j`、`K1`、`r0`。
- `Angle bend` 段：`Species i`、`Species j`、`Species k`、`K2`、`theta0`。

读取时应跳过空行、说明行和文献来源行，只解析明确的参数行。

### Name 与 ff-type

- `Name`：用户自定义的模型标签，来自 data 文件中 type 后面的 `#` 注释，例如 `Ow1`、`Hw2`、`Si_surface`。
- `ff-type`：力场 symbol，例如 ClayFF 中的 `ob`、`obts`、`st`。
- `element`：由 mass 推断，用于筛选候选力场参数。

读取 data 文件时，程序默认令 `Name = ff-type = # 注释内容`。之后用户可以保留 `Name` 不变，只修改 `ff-type` 来匹配具体力场 symbol。保存 data 时，`#` 注释写回的是 `Name`，不是 `ff-type`，避免力场映射覆盖原始模型命名。

### 映射文件建议

后续可以支持一个简单的 `.mod` 或 `.txt` 映射文件，用于把用户自定义 `Name` 映射到力场 `ff-type`：

```ini
[mapping]
Hw*=h*
Ow*=o*
Si_surface=st

[custom_atom_types]
Na_custom.element=Na
Na_custom.charge=1.0
Na_custom.mass=22.98977
Na_custom.pair_style=lj/cut/coul/long
Na_custom.pair_coeff=0.1301, 2.35
```

其中 `*` 表示任意文本。`custom_atom_types` 用于补充当前力场文件中没有的参数。

## 开发与检查

修改 Python 代码后至少运行：

```powershell
python -m py_compile .\lammps_data_viewer_v2.py
```

如果新增或修改功能，需要同步更新本 README 中对应的功能说明、操作方式和注意事项。

### 本项目全部代码（含Python脚本编译为可执行程序）均由本人使用个人ChatGPT账号通过AI生成，本人仅负责提出功能需求与实现思路。本程序开源免费，欢迎学习交流，但严禁用于商业用途。
### All code in this project (including compiling Python scripts into executable programs) was generated by AI via my personal ChatGPT account. I was solely responsible for providing the functional requirements and implementation ideas. This program is open-source and free for non-commercial use; learning and exchange are welcome, but commercial use is strictly prohibited.
    
