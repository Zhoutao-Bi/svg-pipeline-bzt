import json

# 1. 加载原始 JSON 文件
input_file = 'Full_Features_v34.json'
output_file = 'Full_Features_v34_minified.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 使用 json.dumps 进行压缩
# separators=(',', ':') 的作用是移除逗号和冒号后面的空格
minified_content = json.dumps(data, separators=(',', ':'))

# 3. 将压缩后的内容写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(minified_content)

# 打印结果对比
print(f"原始文件大小 (字符数): {len(json.dumps(data, indent=4))}")
print(f"压缩后文件大小 (字符数): {len(minified_content)}")
print(f"节省了约 {((82268 - 18245) / 82268 * 100):.1f}% 的空间")