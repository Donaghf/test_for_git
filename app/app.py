import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from flask import Flask, request, jsonify
app=Flask(__name__)
model_path = "../models"
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        inputs = tokenizer(user_message, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not user_message.strip():
            return jsonify({'error': '消息不能为空'}), 400

        logger.info(f"收到用户消息: {user_message}")

        # 调用DeepSeek模型
        # bot_response = call_deepseek_model(user_message)

        # 记录交互
        logger.info(f"模型回复: {generated_text}")

        return jsonify({
            'response': generated_text,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return jsonify({'error': '服务器内部错误'}), 500


if __name__ == "__main__":
    # 使用环境变量中设置的端口运行 Flask 应用
    app.run(host="0.0.0.0", port=port)