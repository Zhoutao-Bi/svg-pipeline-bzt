# 这个脚本的作用是将原始的 JSON 文件进行压缩，去掉所有不必要的空格和换行，从而减小文件大小。
import json

# 1. 加载原始 JSON 文件
input_file = "features_raw.json"
output_file = "features_minified.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 使用 json.dumps 进行压缩
# separators=(',', ':') 的作用是移除逗号和冒号后面的空格
minified_content = json.dumps(data, ensure_ascii=False, separators=(',', ':'))

# 3. 将压缩后的内容写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(minified_content)

# 打印结果对比
original_content = json.dumps(data, ensure_ascii=False, indent=4)
saved_percent = (1 - len(minified_content) / len(original_content)) * 100 if original_content else 0
print(f"原始文件大小 (字符数): {len(original_content)}")
print(f"压缩后文件大小 (字符数): {len(minified_content)}")
print(f"节省了约 {saved_percent:.1f}% 的空间")
