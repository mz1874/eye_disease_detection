FROM python:3.11-slim

WORKDIR /app

# 拷贝依赖文件并安装
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 只拷贝模型文件和 app2.py
COPY eye_disease_model.keras best_amd_resnet50_finetune.keras app2.py /app/

# 创建上传目录（容器运行时存放上传文件）
RUN mkdir -p /app/uploads

# 暴露 Flask 端口
EXPOSE 5000

# 启动 Flask 服务
CMD ["python", "app2.py"]
