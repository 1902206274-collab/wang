import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib

# 修复Mac字体扫描异常
try:
    matplotlib.font_manager._get_macos_fonts = lambda: []
except:
    pass

# ---------------------- 1. 加载与清洗数据集 ----------------------
# 原始英文列名（绘图使用，避免中文方框）
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "label"
]

# 读取训练集
df_train = pd.read_csv("adult.data.txt", header=None, skipinitialspace=True, names=columns)
# 读取测试集，跳过第一行注释
df_test = pd.read_csv("adult.test.txt", header=None, skipinitialspace=True, names=columns, skiprows=1)

# 清除测试集标签末尾的小数点
df_test["label"] = df_test["label"].str.replace(".", "")

# 合并数据集统一编码
df_all = pd.concat([df_train, df_test])
# 需要编码的离散特征
cat_cols = ["workclass", "education", "marital_status", "occupation",
            "relationship", "race", "sex", "native_country"]

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])
    label_encoders[col] = le

# 重新分割训练、测试集
X_train = df_all.iloc[:len(df_train), :-1]
y_train = df_all.iloc[:len(df_train)]["label"]
X_test = df_all.iloc[len(df_train):, :-1]
y_test = df_all.iloc[len(df_train):]["label"]

# ---------------------- 2. 训练三种决策树模型 ----------------------
criteria = [
    ("gini", "Gini"),
    ("entropy", "Information Gain"),
    ("log_loss", "Gain Ratio")
]

result_list = []

for criterion, name in criteria:
    print(f"\n========== Training Decision Tree: {name} ==========")
    # 构建决策树，限制最大深度为4，固定随机种子保证结果复现
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=42)
    dt.fit(X_train, y_train)

    # 预测并计算准确率
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    result_list.append([name, acc])
    print(f"Accuracy: {acc:.4f}")

    # 绘制决策树，指定DejaVu Sans字体，彻底解决方块乱码
    plt.figure(figsize=(14, 8), dpi=300)
    plt.rcParams["font.family"] = "DejaVu Sans"

    plot_tree(
        dt,
        feature_names=columns[:-1],
        class_names=["<=50K", ">50K"],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title(f"Decision Tree — {name}", fontsize=12)
    plt.savefig(f"tree_{criterion}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved image: tree_{criterion}.png")

# ---------------------- 输出实验结果 ----------------------
print("\n==== Final Results ====")
for name, acc in result_list:
    print(f"{name} : Accuracy = {acc:.4f}")

# 绘制准确率对比柱状图
plt.figure(dpi=300)
plt.rcParams["font.family"] = "DejaVu Sans"

names = [item[0] for item in result_list]
accuracies = [item[1] for item in result_list]

plt.bar(names, accuracies)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Different Split Criteria")
plt.savefig("accuracy_compare.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved comparison chart: accuracy_compare.png")
