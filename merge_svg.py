import os
import re
import xml.etree.ElementTree as ET

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def final_nested_svg(input_folder, output_file):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.svg')]
    files.sort(key=natural_key)

    if not files:
        print("未找到 SVG 文件")
        return

    print(f"找到 {len(files)} 个文件，正在执行 [修复版] 嵌套合并...")

    # --- 1. 注册命名空间 ---
    # 告诉 Python：遇到这个网址，不要加 ns0 前缀，直接当作默认命名空间
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    
    # 防止 trimesh 相关的报错，我们给它一个合法的自定义前缀
    try:
        ET.register_namespace('trimesh', "https://github.com/mikedh/trimesh")
    except ValueError:
        pass

    # --- 2. 创建总容器 (修复核心) ---
    # 关键修改：不要在第二个参数字典里写 'xmlns'！
    # 直接在标签名里带上 URI，Python 会自动处理剩下的事情。
    master_root = ET.Element('{http://www.w3.org/2000/svg}svg')
    
    # 单独设置宽高，避免字典冲突
    master_root.set('width', '100%')
    master_root.set('height', '100%')
    # master_root.set('viewBox', '0 0 100 100') # 可选，如果不加，浏览器会自适应

    master_tree = ET.ElementTree(master_root)

    # --- 3. 循环处理 ---
    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            original_viewbox = root.get('viewBox')
            
            # 创建嵌套层 (也要带上 URI，确保它是标准的 svg 标签)
            layer_svg = ET.Element('{http://www.w3.org/2000/svg}svg')
            
            # 设置属性
            layer_svg.set('width', '100%')
            layer_svg.set('height', '100%')
            layer_svg.set('overflow', 'visible')
            layer_svg.set('id', os.path.splitext(filename)[0])
            
            if original_viewbox:
                layer_svg.set('viewBox', original_viewbox)
            else:
                # 如果原图没有viewBox，这可能是个问题，给一个默认值
                layer_svg.set('viewBox', '0 0 100 100')

            # 搬运内容
            for child in list(root):
                layer_svg.append(child)

            master_root.append(layer_svg)
            print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理 {filename} 失败: {e}")

    # --- 4. 保存 ---
    try:
        master_tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"--------------------------------")
        print(f"成功！修复版文件已保存至: {output_file}")
        print("现在用浏览器打开应该不会报错了。")
    except Exception as e:
        print(f"保存失败: {e}")

if __name__ == "__main__":
    final_nested_svg("optimized_slices_x", "merged_slices_x.svg")
    final_nested_svg("optimized_slices_y", "merged_slices_y.svg")
    final_nested_svg("optimized_slices_z", "merged_slices_z.svg")


