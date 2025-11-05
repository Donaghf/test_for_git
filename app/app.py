import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from flask import Flask, request, jsonify
app=Flask(__name__)
model_path = "./models"
port = os.environ.get("PORT", 5000)

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("模型加载完成！开始对话吧！(输入'退出'结束)")
@app.route("/predict", methods=["POST"])
def predict():
    # 获取请求数据
    data = request.get_json()
    input_text = data.get("input_text", "")

    # 处理输入文本并生成输出
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)

    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 返回预测结果
    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    # 使用环境变量中设置的端口运行 Flask 应用
    app.run(host="0.0.0.0", port=port)