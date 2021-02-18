from sklearn.externals import joblib

# 保存模型
def save_model(model, name):
    joblib.dump(model, name)

# 加载模型
def load_model(predict_model_name):
    return joblib.load(predict_model_name)