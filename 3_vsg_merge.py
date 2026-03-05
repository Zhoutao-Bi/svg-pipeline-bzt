import os
import re
import sys
import xml.etree.ElementTree as ET

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def final_nested_svg(input_folder, output_file):
    print(f"    [Debug] 准备从 {input_folder} 合并至 {os.path.basename(output_file)}")
    if not os.path.exists(input_folder):
        print(f"    [警告] 目录不存在，无法合并: {input_folder}")
        return
        
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.svg')]
    files.sort(key=natural_key)

    if not files:
        print(f"    [提示] 目录中无 SVG 文件: {input_folder}")
        return

    ET.register_namespace('', "http://www.w3.org/2000/svg")
    ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
    
    try:
        ET.register_namespace('trimesh', "https://github.com/mikedh/trimesh")
    except ValueError:
        pass

    master_root = ET.Element('{http://www.w3.org/2000/svg}svg')
    master_root.set('width', '100%')
    master_root.set('height', '100%')

    master_tree = ET.ElementTree(master_root)

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            original_viewbox = root.get('viewBox')
            layer_svg = ET.Element('{http://www.w3.org/2000/svg}svg')
            
            layer_svg.set('width', '100%')
            layer_svg.set('height', '100%')
            layer_svg.set('overflow', 'visible')
            layer_svg.set('id', os.path.splitext(filename)[0])
            
            if original_viewbox:
                layer_svg.set('viewBox', original_viewbox)
            else:
                layer_svg.set('viewBox', '0 0 100 100')

            for child in list(root):
                layer_svg.append(child)

            master_root.append(layer_svg)

        except Exception as e:
            print(f"    [错误] 解析图层 {filename} 失败: {e}")

    try:
        master_tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"    [Info] 合并成功！输出文件: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"    [错误] 文件保存失败: {e}")

if __name__ == "__main__":
    base_temp = "data/02_temp"
    base_out = "data/03_output"
    print(f"========== [3] 开始运行 SVG 图层合并脚本 ==========")
    
    if os.path.exists(base_temp):
        projects = [d for d in os.listdir(base_temp) if os.path.isdir(os.path.join(base_temp, d))]
        
        if len(sys.argv) > 1:
            projects = [p for p in projects if p == sys.argv[1]]
            
        for project_name in projects:
            print(f"\n>>> 正在合并项目 [{project_name}] 的切片...")
            fit_dir = f"{base_temp}/{project_name}/fit_slices"
            merged_dir = f"{base_out}/{project_name}/merged_svg"
            
            os.makedirs(merged_dir, exist_ok=True)
            
            final_nested_svg(f"{fit_dir}/X", os.path.join(merged_dir, "view_X.svg"))
            final_nested_svg(f"{fit_dir}/Y", os.path.join(merged_dir, "view_Y.svg"))
            final_nested_svg(f"{fit_dir}/Z", os.path.join(merged_dir, "view_Z.svg"))
    else:
        print(f"⚠️ [错误] 未找到目录: {base_temp}")
    print(f"========== [3] 脚本运行结束 ==========")