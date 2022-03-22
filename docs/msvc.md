## 第一次试验   探究残差块的作用

实验名称：`msvc_caps_dataset_task1`

使用机器：l0 for modelnet10,l4 for other

hash:c327667d8880e700340c7352f79cf37df4474685

参数：单层隐藏层，输入带宽32 16 8，**有残差块** ，分类器`nclasses * 3-->nclasses`

| 数据集     | 显卡占用            | acc    |      |
| ---------- | ------------------- | ------ | ---- |
| modelnet10 | 并行，bs=16，9800+M | 0.9118 |      |
| modelnet40 | 并行，bs=8，9800+M  |        |      |
| Shrec15    | 并行，bs=4，8200+M  |        |      |
| Shrec17    | 并行，bs=12，8779+M |        |      |



实验名称：`msvc_caps_dataset_task2`

使用机器：l1

hash:61a2e064812c44c757df9fb1f17dcd5d28c8a038

参数：单层隐藏层，输入带宽32 16 8，**无残差块**，分类器`nclasses * 3-->nclasses`



| 数据集     | 显卡占用            | Acc   |      |
| ---------- | ------------------- | ----- | ---- |
| modelnet10 | 并行，bs=16，9800+M | 0.894 |      |
| modelnet40 | 并行，bs=8，9800+M  |       |      |
| Shrec15    | 并行，bs=8，8200+M  |       |      |
| Shrec17    | 并行，bs=16，8779+M | 0.475 |      |



观察与结论：

## 第二次试验 探究分类器的参数

实验名称：`msvc_caps_dataset_cls2`

使用机器：l0

hash:9c7f3b3d295ef0c074f067788d1b3e759cff44c7

参数：单层隐藏层，输入带宽32 16 8，无残差块 ，**分类器** `nclasses * 3 --> 256 --> 512 --> 256 --> nclasses`

| 数据集     | 显卡占用            | Acc    | Time  |
| ---------- | ------------------- | ------ | ----- |
| modelnet10 | 并行，bs=16，9800+M | 0.8607 | 1h 32 |
| Shrec15    | 并行，bs=8，8200+M  | 0.4045 | 39m   |
|            |                     |        |       |
|            |                     |        |       |



实验名称：`msvc_caps_dataset_cls3`

使用机器：l5

hash:

参数：单层隐藏层，输入带宽32 16 8，无残差块 ，**分类器** `nclasses * 3 --> 256 --> nclasses`

| 数据集     | 显卡占用            | Acc    | Time |
| ---------- | ------------------- | ------ | ---- |
| modelnet10 | 并行，bs=16，9800+M | 0.8839 | 1h28 |
| Shrec15    | 并行，bs=8，8200+M  | 0.5365 | 44m  |
|            |                     |        |      |
|            |                     |        |      |



结论：多层的可能需要更长时间的训练，而浅层（2层）收敛较快



## 第三次实验 使用更深的层数

不使用残差块

hash:63e51cfda9891406affa09c018b32f5d2520f106

机器: l0 数据集：0 1 实验名：`msvc_caps_${dataset}_deep_no_res`

| 数据集     | acc    |
| ---------- | ------ |
| modelnet10 | 0.8848 |
| modelnet40 | 0.7748 |
| Shrec15    |        |
| Shrec17    |        |







使用残差块

Hash:e3ac37af9f78ebddfca7abb7545bd46decaeb6cb

机器: l4 数据集：0 1 实验名：`msvc_caps_${dataset}_deep_res`

| 数据集     | acc    |
| ---------- | ------ |
| modelnet10 | 0.9074 |
| modelnet40 | 0.8259 |
| Shrec15    |        |
| Shrec17    |        |

