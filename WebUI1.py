import pandas as pd
import gradio as gr
import torch
import torch.nn as nn
import pickle
param_mapping = {
    "一层住宅": 20, "两层住宅": 60,
    "农村": 'A', "城市": 'C',
    "所有设施": 'AllPub', "电、气、水（化粪池）": 'NoSewr', "只有电和气": 'NoSeWa', "只有电": 'ELO',
    "乙烯基壁板": 'VinylSd', "金属外墙": 'MetalSd', "木外墙": 'Wd Sdng',
    "面砖": 'BrkFace', "无": 'None', "石材": 'Stone',
    "优秀": 'Ex', "好": 'Gd', "普通": 'TA',
    "浇筑混凝土": 'PConc', "混凝土块": 'CBlock', "砖与瓦": 'BrkTil',
    "有": 'Y', "无": 'N',
    "连接到住宅的车库": 'Attchd', "与住宅分开的车库": 'Detchd',
    "无泳池": 'NA', "隐私良好": 'GdPrv', "几乎无隐私": 'MnPrv',
    "电梯": 'Elev', "棚屋": 'Shed'
}
class HousePricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HousePricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def predict_house_price(MSSubClass, MSZoning, LotArea, Utilities, OverallQual, YearBuilt, Exterior1st, MasVnrType, ExterQual, Foundation, BsmtQual, HeatingQC, CentralAir, BedroomAbvGr, KitchenQual, GarageType, PoolQC, Fence, MiscFeature):
    MSSubClass = param_mapping.get(MSSubClass, MSSubClass)
    MSZoning = param_mapping.get(MSZoning, MSZoning)
    Utilities = param_mapping.get(Utilities, Utilities)
    Exterior1st = param_mapping.get(Exterior1st, Exterior1st)
    MasVnrType = param_mapping.get(MasVnrType, MasVnrType)
    ExterQual = param_mapping.get(ExterQual, ExterQual)
    Foundation = param_mapping.get(Foundation, Foundation)
    BsmtQual = param_mapping.get(BsmtQual, BsmtQual)
    HeatingQC = param_mapping.get(HeatingQC, HeatingQC)
    CentralAir = param_mapping.get(CentralAir, CentralAir)
    KitchenQual = param_mapping.get(KitchenQual, KitchenQual)
    GarageType = param_mapping.get(GarageType, GarageType)
    PoolQC = param_mapping.get(PoolQC, PoolQC)
    Fence = param_mapping.get(Fence, Fence)
    MiscFeature = param_mapping.get(MiscFeature, MiscFeature)
    LotArea*=10.7639
    data = {
        "MSSubClass": [MSSubClass],
        "MSZoning": [MSZoning],
        "LotFrontage": [70.0],  
        "LotArea": [LotArea],
        "Street": ['Pave'],  
        "Alley": ['NA'],  
        "LotShape": ['Reg'],  
        "LandContour": ['Lvl'],  
        "Utilities": [Utilities],
        "LotConfig": ['Inside'],  
        "LandSlope": ['Gtl'],  
        "Neighborhood": ['NAmes'],  
        "Condition1": ['Norm'],  
        "Condition2": ['Norm'],  
        "BldgType": ['1Fam'],  
        "HouseStyle": ['1Story'],  
        "OverallQual": [OverallQual],
        "OverallCond": [5],  
        "YearBuilt": [YearBuilt],
        "YearRemodAdd": [1990],  
        "RoofStyle": ['Gable'],  
        "RoofMatl": ['CompShg'],  
        "Exterior1st": [Exterior1st],
        "Exterior2nd": ['VinylSd'],  
        "MasVnrType": [MasVnrType],
        "MasVnrArea": [LotArea],  
        "ExterQual": [ExterQual],
        "ExterCond": ['TA'],  
        "Foundation": [Foundation],
        "BsmtQual": [BsmtQual],
        "BsmtCond": ['TA'],  
        "BsmtExposure": ['No'],  
        "BsmtFinType1": ['Unf'],  
        "BsmtFinSF1": [0.8*LotArea],  
        "BsmtFinType2": ['Unf'],  
        "BsmtFinSF2": [0.8*LotArea],  
        "BsmtUnfSF": [0.1*LotArea],  
        "TotalBsmtSF": [0.9*LotArea],  
        "Heating": ['GasA'],  
        "HeatingQC": [HeatingQC],
        "CentralAir": [CentralAir],
        "Electrical": ['SBrkr'],  
        "1stFlrSF": [0.9*LotArea],  
        "2ndFlrSF": [0.9*LotArea],  
        "LowQualFinSF": [5],  
        "GrLivArea": [1515],  
        "BsmtFullBath": [0.5],  
        "BsmtHalfBath": [0.0],  
        "FullBath": [1.5],  
        "HalfBath": [0.5],  
        "BedroomAbvGr": [BedroomAbvGr],
        "KitchenAbvGr": [1],  
        "KitchenQual": [KitchenQual],
        "TotRmsAbvGrd": [6],  
        "Functional": ['Typ'],  
        "Fireplaces": [0],  
        "FireplaceQu": ['NA'],  
        "GarageType": [GarageType],
        "GarageYrBlt": [1980],  
        "GarageFinish": ['Unf'],  
        "GarageCars": [2],  
        "GarageArea": [0.3*LotArea],  
        "GarageQual": ['TA'],  
        "GarageCond": ['TA'],  
        "PavedDrive": ['Y'],  
        "WoodDeckSF": [94],  
        "OpenPorchSF": [47],  
        "EnclosedPorch": [22],  
        "3SsnPorch": [3],  
        "ScreenPorch": [15],  
        "PoolArea": [0.3*LotArea],  
        "PoolQC": [PoolQC],
        "Fence": [Fence],
        "MiscFeature": [MiscFeature],
        "MiscVal": [0],  
        "MoSold": [6],  
        "YrSold": [2007],  
        "SaleType": ['WD'],  
        "SaleCondition": ['Normal']  
    }   
    test_data = pd.DataFrame(data)
    with open('pickle/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # 只转换测试数据
    X_test = preprocessor.transform(test_data)

    # 将稀疏矩阵转换为密集矩阵
    X_test = X_test.toarray()

    # 转换为 PyTorch 张量
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 加载模型
    input_dim = X_test.shape[1]
    model = HousePricePredictor(input_dim)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    # 进行预测
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()

    # 加载目标变量的 StandardScaler
    with open('pickle/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    # 反标准化预测值
    predictions = scaler_y.inverse_transform(predictions)

    # 输出预测结果
    predictions = predictions.flatten()*7.2558
    return  f'您的梦中情房仅需{predictions[0]:.2f}￥！'
        

with gr.Blocks() as app:
    gr.Markdown("# 你的梦中情房多少钱？")  # 顶端标题
    with gr.Row():
        mssubclass = gr.Dropdown(label="房屋类型", choices=["一层住宅", "两层住宅"], value="一层住宅")
        mszoning = gr.Dropdown(label="居住地区", choices=["农村", "城市"], value="农村")
        lotarea = gr.Number(label="地块面积（平方米）", value=100)
        utilities = gr.Dropdown(label="可用设施", choices=["所有设施", "电、气、水（化粪池）", "只有电和气", "只有电"], value="所有设施")
    with gr.Row():
        overallqual = gr.Slider(label="总体材料和完成质量", minimum=1, maximum=10, value=6,step=1)
        yearbuilt = gr.Number(label="建造年份", minimum=1900, maximum=2025,value=2024,step=1)
        exterior1st = gr.Dropdown(label="外墙材料", choices=["乙烯基壁板", "金属外墙", "木外墙"], value="乙烯基壁板")
        masvnrtype = gr.Dropdown(label="石材饰面类型", choices=["面砖", "无", "石材"], value="无")
    with gr.Row():
        exterqual = gr.Dropdown(label="外墙质量", choices=["优秀", "好", "普通"], value="普通")
        foundation = gr.Dropdown(label="地基类型", choices=["浇筑混凝土", "混凝土块", "砖与瓦"], value="浇筑混凝土")
        bsmtqual = gr.Dropdown(label="地下室质量", choices=["优秀", "好", "普通"], value="优秀")
        heatingqc = gr.Dropdown(label="供暖质量和状况", choices=["优秀", "好"], value="优秀")
    with gr.Row():
        centralair = gr.Radio(label="中央空调", choices=["有", "无"], value="有")
        bedroomabvgr = gr.Slider(label="卧室数量", minimum=0, maximum=10, value=3, step=1)
        kitchenqual = gr.Dropdown(label="厨房质量", choices=["优秀", "好", "普通"], value="普通")
        garagetype = gr.Dropdown(label="车库类型", choices=["连接到住宅的车库", "与住宅分开的车库"], value="连接到住宅的车库")
    with gr.Row():
        poolqc = gr.Dropdown(label="泳池质量", choices=["优秀", "无泳池"], value="无泳池")
        fence = gr.Dropdown(label="围栏质量", choices=["隐私良好", "几乎无隐私"], value="隐私良好")
        miscfeature = gr.Dropdown(label="其他特征", choices=["电梯", "棚屋", "无"], value="无")
    btn = gr.Button("预测")
    output = gr.Label()
    btn.click(predict_house_price, inputs=[mssubclass, mszoning, lotarea, utilities, overallqual, yearbuilt, exterior1st, masvnrtype, exterqual, foundation, bsmtqual, heatingqc, centralair, bedroomabvgr, kitchenqual, garagetype, poolqc, fence, miscfeature], outputs=output)
    
app.launch(server_name="0.0.0.0", server_port=7860)
