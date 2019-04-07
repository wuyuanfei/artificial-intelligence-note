# 机器学习

[TOC]

## 1. 机器学习基础

### 定义

> 机器学习是一种能够赋予机器学习的能力以此让它完成直接编程无法完成的功能的方法。但从实践的意义上来说，机器学习是一种通过**利用数据，训练出模型，然后使用模型预测的一种方法**。
>

### 主要任务

- **分类：**实例数据划分，通常为**标称型数据**，**分类是对离散变量的预测**

- **回归：**预测**数值型数据**，通常为连续型数据，**回归是对连续变量的预测**

### 开发步骤

- 问题建模
- 获取数据
- 特征工程
- 模型训练与验证
- 模型诊断与调优
- 线上运行

### 机器学习相关库
- [**Numpy**](https://docs.scipy.org/doc/numpy/user/quickstart.html)：数值编程工具、矩阵数据类型、矢量处理。NumPy的全名为Numeric Python，是一个开源的Python科学计算库

- [**Pandas**](http://pandas.pydata.org/pandas-docs/stable/)：基于NumPy和Matplotlib的数据处理库。包括数据分析和数据可视化，特有的数据结构： Series	DataFrame
- [**Matplotlib**](https://matplotlib.org/tutorials/index.html)：可视化绘图库
- [**Scikit-learn**](https://scikit-learn.org/stable/)：机器学习库，基于NumPy、SciPy、Matplotlib，开源，涵盖分类、回归和聚类算法，代码和文档完备；具备分类、回归、聚类、梯度衰减、模型选择、预处理等机器学习基础功能

### 机器学习算法
- **监督学习  (supervised learning)**

  - K近邻算法
  - 线性回归
  - 逻辑回归
  - 支持向量机（ SVM）
  - 决策树和随机森林
  - 神经网络

  > 训练集的所有结果都是已知的，根据已知类别的样本进行学习

- **无监督学习 (unsupervised learning)**

  - **聚类**
    - K 均值
    - 层次聚类分析（ Hierarchical Cluster Analysis， HCA）
    - 期望最大值
  - **可视化和降维**
    - 主成分分析（ Principal Component Analysis， PCA）
    - 核主成分分析
    - 局部线性嵌入（ Locally-Linear Embedding， LLE）
    - t-分布邻域嵌入算法（ t-distributed Stochastic Neighbor Embedding， t-SNE）
  - **关联性规则学习**
    - Apriori 算法
    - Eclat 算法

  > 所有的结果都是未知的；缺乏足够的先验知识；难以人工标注类别或进行人工类别标注的成本太高；根据类别未知(没有被标记)的样本来学习 
  >

- **半监督学习(semisupervised learning)**

  > **少量的标注样本和大量的未标注样本**；尽量少的人工，带来尽量大的价值；平滑假设(Smoothness Assumption)；聚类假设(Cluster Assumption)；流形假设(Manifold Assumption) 

- **强化学习(reinforcement learning)**

  > 是否越来越接近目标（回报函数，reward function）；输入数据直接反馈到模型，模型必须对此立刻作出调整；常见的应用场景包括动态系统以及机器人控制

### 机器学习方式

- **批量学习（离线学习）**

  > 在批量学习中， 系统不能进行持续学习： 必须用所有可用数据进行训练。 这通常会占用**大量时间和计算资源**， 所以一般是**线下**做的。 首先是进行训练， 然后部署在生产环境且停止学习， 它只是使用已经学到的策略。 这称为**离线学习**。

- **在线学习**

  > 在在线学习中， 是用**数据实例持续地进行训练**， 可以一次一个或一次几个实例（ 称为小批量） 。 每个学习步骤都很快且廉价， 所以系统可以动态地学习到达的新数据（

### 主要挑战

- **错误的数据**

  - 训练数据不足

  - 没有代表性的数据

  - 低质量数据：训练集中的错误、 异常值和噪声（ 错误测量引入的） 太多

    > 1. 如果一些实例是**明显的异常值**， 最好**删掉**它们或尝试**手工修改**错误；
    > 2. 如果一些实例缺少特征，是否忽略这个属性、 忽略这些实例、 填入缺失值（ 平均数或中位数） ， 或者训练一个含有这个特征的模型和一个不含有这个特征的模型

  - 不相关的特征

    > 1. 特征选择： 在所有存在的特征中选取最有用的特征进行训练。
    > 2. 特征提取： 组合存在的特征， 生成一个更有用的特征
    > 3. 收集新数据创建新特征

- **错误的算法**

  - 过拟合训练数据：模型过于复杂

    > 1. 简化模型， 可以通过选择一个参数更少的模型（ 比如使用线性模型， 而不是高阶多项式模型） 、 减少训练数据的属性数、 或限制一下模型（正则化）
    > 2. 收集更多的训练数据
    > 3. 减小训练数据的噪声（ 比如， 修改数据错误和去除异常值）

  - 欠拟合训练数据：模型过于简单

    > 1. 选择一个更强大的模型， 带有更多参数
    > 2. 用更好的特征训练学习算法（ 特征工程）
    > 3. 减小对模型的限制（ 比如， 减小正则化超参数）


### 开源数据集

- 流行的开源数据仓库：
  - UC Irvine Machine Learning Repository
  - **Kaggle datasets**
  - Amazon’s AWS datasets
- 准入口（ 提供开源数据列表）
  - http://dataportals.org/
  - http://opendatamonitor.eu/
  - http://quandl.com/
- 其它列出流行开源数据仓库的网页：
  - Wikipedia’s list of Machine Learning datasets
  - Quora.com question
  - Datasets subreddit

## 2. 模型评估

### 损失函数
- **0~1损失函数（0~1 loss function）：分类问题**

$$
L(Y,f(X))=\begin{cases}
			1, & Y \neq f(X) \cr
			0, & Y = f(X)
           \end{cases}
$$

- **平方损失函数（quadratic loss function）：回归问题**

$$
L(Y,f(X))=(Y-f(X))^2
$$

- **绝对损失函数（absolute loss function）：**

$$
L(Y,P(Y|X))=|Y-f(x)|
$$

- **对数损失函数（logarithmic loss function）：**

$$
L(Y,P(Y|X))=-logP(Y|X)
$$

- **交叉熵（cross entry）**

> 说明：损失函数值越小表明模型越好**。

###  风险系数

- **损失函数期望**（风险函数*risk function*或期望损失*expected loss*）：模型关于联合分布的期望损失![img] 
  $$
  R_{exp}(f)=E_p[L(Y,f(X))]=\int_{x\bf{x}y}L(y,f(x))P(x,y)dxdy
  $$

- **经验风险或经验损失**（*empirical risk or empirical loss*）：模型关于训练样本集的**平均损失**
  $$
  R_{emp}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))
  $$

> **说明：根据大数定律，当样本容量N趋于无穷时，经验风险趋于期望风险，可用经验风险估计期望风险。**

  - **经验风险最小化**： 
    $$
    \min_{f \in F} \frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))
    $$


  > **说明：F为假设空间，经验风险最小的模型就是最优的模型。比如极大似然估计（maximum likelihood estimation），模型为条件概率分布，损失函数为对数损失函数，经验风险最小化就等价于极大似然估计**

- **结构风险**：
  $$
  R_{srm}(f)=\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f)
  $$


  > **说明：第一项为经验项， 第二项为正则项，**J(f)**为模型的复杂度，模型**f**越复杂，复杂度**j(f)**就越大，**λ**为惩罚因子。**

- **结构风险最小化**：防止过拟合 
  $$
  \min_{f \in F}\frac{1}{N}\sum_{i=1}^{N}L(y_i,f(x_i))+\lambda J(f)
  $$


  > **说明：贝叶斯估计中的最大后验概率估计**

### 模型选择

- 模型选择常用方法为：**正则化与交叉验证**


- **正则化的作用：**选择经验风险与模型复杂度同时较小的模型
  - **L1**范数正则化： 
    $$
    L(w)=\frac{1}{N}\sum_{i=1}^{N}(f(x_i;w)-y_i)^2+\lambda||w||_{1}
    $$

  - **L2**范数正则化：回归问题  
    $$
    L(w)=\frac{1}{N}\sum_{i=1}^{N}(f(x_i;w)-y_i)^2+\frac{\lambda}{2}||w||^{2}
    $$

- **交叉验证的作用：**拆分数据集反复进行训练、测试以及模型选择

  - **简单交叉验证**

    > 训练集：测试集 = 7 : 3 （其他比例亦可）

  - **K折交叉验证**

    > 将已给数据切分为K个互不相交的子集，K-1个子集用作训练集，余下的1个子集用于测试模型

  - **留一交叉验证**

    > K折交叉验证的特殊情形为K=N 

### 分类问题评价指标（混淆矩阵）

- **精准率/查准率 （Precision）**
  $$
  P=\frac{TP}{TP+FP}
  $$

- **召回率/查全率 （Recall）**
  $$
  R=\frac{TP}{TP+FN}
  $$

  > 等于真正率TPR

- **ROC曲线 （receiver operating characteristic curve）**

>  1. 二分类模型返回一个概率值，通过调整阈值，即大于该阈值为正类，反之负类，可以得到多个（FPR，TPR）点，描点画图得到的曲线即为ROC曲线。
>  2. ROC曲线是根据一系列不同的二分类方式（分界值或决定阈），以真阳性率（灵敏度）为纵坐标，假阳性率（1-特异度）为横坐标绘制的曲线。
>  3. 传统的诊断试验评价方法有一个共同的特点，必须将试验结果分为两类，再进行统计分析。ROC曲线的评价方法与传统的评价方法不同，无须此限制，而是根据实际情况，允许有中间状态，可以把试验结果划分为多个有序分类，如正常、大致正常、可疑、大致异常和异常五个等级再进行统计分析。越靠近（0,1）效果越好。

- **AUC面积（Area Under Curve）**

> 1. AUC（Area Under Curve）被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。
> 2. 使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC越大的分类器效果更好

-   **F1调和平均数（F1-Socre）**
  $$
  \frac{1}{F_1}=\frac{1}{2}(\frac{1}{P}+\frac{1}{R})
  $$

  $$
  F_1=\frac{2TP}{2TP+FP+FN}
  $$

  > 精准率和召回率都高时，F1值也会高
  >

-   **准确率/正确率（Accuracy）**
  $$
  Accuracy=\frac{TP+TN}{P+N}
  $$

>
> 对于给定的测试数据集，分类器正确分类的样本数与总样本数之比，从某种意义上得到一个分类器是否有效，但它并不总是能有效的评价一个分类器的工作
>

- **错误率（Error rate）**
  $$
  ErrorRate=\frac{FP+FN}{P+N}
  $$


> ​    对于给定的测试数据集，分类器错误分类的样本数与总样本数之比，分对与分错是互斥事件，所以accuracy =1 - error rate
>

 

API参考

 

 分类算法对比

 

  

 

算法选择

真正率（ True Positive Rate, TPR），也称灵敏度（sensitivity）：

真负率 (True Negative Rate, TNR），特指度（specificity）：

假正率 （False Positive Rate, FPR）：

假负率（ False Negative Rate , FNR）：

### 回归问题评价指标

- **RMSE（均方根误差）**
  $$
  RMSE(X,h)=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(h(x^{(i)}-y^{(i)})^2}
  $$

- **MAE（平均绝对误差）**
  $$
  MAE(X,h)=\frac{1}{m}\sum_{i=1}^{m}|h(x^{(i)}-y^{(i)}|
  $$
  

### 聚类问题评价指标





## 3. 特征工程

### 获取与存储

### 缺失值处理

### 特征处理

- **数据清洗**
- **数据预处理**

  - 标准化（StandardScaler）

  - 区间缩放（MinMaxScaler）

    

  - 归一化（Normalizer）

    > 需要：基于**参数**的模型或基于**距离**的模型，都是要进行特征的归一化
    >
    > 不需要：基于**树的方法**是不需要进行特征的归一化，例如随机森林，bagging 和 boosting等

  - 二值化（Binarizer）

  - 哑编码（OneHotEncoder）

    > 

  - 缺失值计算（Imputer）

  - 多项式数据转换（PolynomialFeatures）

- **特征选择理论**

  - 过滤法（Filter）

  - 包装法（Wrapper）

  - 集成法（Embedded）

- **特征选择**

  - 方差选择法（VarianceThreshold）
  - 相关系数法（SelectKBest）
  - 卡方检验（SelectKBest）
  - 互信息法（SelectKBest）
  - 递归特征消除法（RFE）
  - 基于惩罚项的特征选择法（SelectFromModel）
  - 基于树模型的特征选择法（SelectFromModel）
  - 嵌入法特征选择
  - 正则项特征选择
  - 树模型特征选择

- **特征扩展**

- **更新特征**



### 模型选择

![img](/home/yeapht/ArtificialIntelligenceDisk/aura/8-学习笔记/assets/image2017-7-10 20_4_57.png)



### 模型微调

- 网格搜索 GridSearchCV

  > 适用于参数组合较少的情形下

- 随机搜索 RandomizedSearchCV

  > 适用于参数组合较多的情形下

- 集成方法 

  >  将表现最好的模型组合起来



## 4. K近邻法（KNN）

### 定义

​	K最近邻 (KNN，K-Nearest Neighbor)数据挖掘分类技术中最简单的方法之一，所谓K最近邻，就是K个最近的邻居，每个样本都可以用它最接近的K个邻居来代表

![img](file:////tmp/wps-yeapht/ksohtml/wps5yQwTa.jpg) 

### 算法步骤

- 计算已知类别数据集中的点与当前点之间的距离

- 按照距离**递增次序**排序

- 选取与当前点距离最小的k个点

- 确定前k个点所在的类别的出现频率

- 返回前k个点出现频率最高的类别作为当前点的预测分类 


### 三要素

- **K值的选择**：反映了对近似误差与估计误差间的权衡

> 当K值较小，邻近点若为噪声，则准确率大幅降低。**K值的减小意味着整体模型变得复杂**，容易发生过拟合。当K值较大，可以减小估计误差，但是近似误差增大，**K值的增大意味着模型变得简单**，起不到预测作用。 
>

- **距离度量**

  -  **一般Lp距离**：这里p ≥ 1

  $$
  L_p(x_i,x_j)=(\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|^p)^\frac{1}{p})
  $$
  - **L1距离**：曼哈顿距离 Manhattan distance（p = 1）

  $$
  L_p(x_i,x_j)=\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|
  $$

  - **L2距离**：欧式距离 Euclidean distance（p = 2）

  $$
  L_p(x_i,x_j)=\sqrt[]{\sum_{l=1}^n|x_i^{(l)}-x_j^{(l)}|^2}
  $$

> ​	L1和L2比较。比较这两个度量方式是挺有意思的。在面对两个向量之间的差异时，L2比L1更加不能容忍这些差异。也就是说，相对于1个巨大的差异，L2距离更倾向于接受多个中等程度的差异

- **分类规则选择** 

  **多数表决，经验风险最小化**

### 特点

- 优点：**精度高，对异常值不敏感**，无数据输入假定


- 缺点：**计算复杂度高，空间复杂度高** 


### kd树

> kd*树（二叉树）是一种对*k*维空间中的实例点进行存储以便对其进行**快速检索**的树形数据结构。

### API参考

**Sklearn.neighbors.KNeighborsClassifier**

 

 

 

 

## 5. 决策树（Decision tree）

 







## 6. 朴素贝叶斯法（Nave Bayes）

 







## 7. 逻辑斯谛回归（Logistic Regression）

 







## 8. 支持向量机（SVM）

### 核函数

### 最小序列优化（SMO）

 





## 9. 集成学习

### **Bagging**

### **Boosting**

### **Stacking**



## 降维

- 降维方法

  - 投影（projection）
  - 流形学习（Maniflod Learning）

- 降维技术

  - 主成分分析（PCA）

  - 核成分分析（Kernel PCA）

  - 局部线性嵌入（LLE）

    

维数灾难

> 训练集的维数越高，过拟合的风险就越大
>
> 解决方案：增加训练集大小从而达到拥有足够密度的训练集





## Kaggle实战

kaggle wiki