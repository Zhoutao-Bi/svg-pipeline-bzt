import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import os
import sys

def optimize_svg_circles(input_file, output_file, tolerance=0.01):
    print(f"    [Debug] 准备优化文件: {os.path.basename(input_file)}")
    if not os.path.exists(input_file):
        print(f"    [警告] 找不到输入文件，跳过: {input_file}")
        return False
        
    ET.register_namespace('', "http://www.w3.org/2000/svg")
    
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError:
        print(f"    [错误] XML解析失败: {input_file}")
        return False

    ns = {'svg': 'http://www.w3.org/2000/svg'}
    circle_map = defaultdict(list)

    layers = []
    for elem in root.iter():
        if 'id' in elem.attrib and str(elem.attrib['id']).startswith('layer_'):
            layers.append(elem)

    for layer in layers:
        layer_id_str = layer.attrib['id']
        try:
            layer_num = int(re.search(r'\d+', layer_id_str).group())
        except:
            continue

        circles = layer.findall('svg:circle', ns) + layer.findall('circle')
        
        for circle in circles:
            try:
                cx = float(circle.attrib.get('cx', 0))
                cy = float(circle.attrib.get('cy', 0))
                r = float(circle.attrib.get('r', 0))
                
                fingerprint = (round(cx, 2), round(cy, 2), round(r, 2))
                
                circle_map[fingerprint].append({
                    'layer_num': layer_num,
                    'element': circle,
                    'parent': layer
                })
            except ValueError:
                continue

    processed_count = 0
    removed_count = 0

    for fingerprint, occurrences in circle_map.items():
        if len(occurrences) <= 1:
            continue  

        occurrences.sort(key=lambda x: x['layer_num'])
        layer_nums = [x['layer_num'] for x in occurrences]
        
        ranges = []
        if not layer_nums: continue
        
        start = layer_nums[0]
        prev = layer_nums[0]
        
        for num in layer_nums[1:]:
            if num != prev + 1:
                if start == prev: ranges.append(f"{start}")
                else: ranges.append(f"{start}-{prev}")
                start = num
            prev = num
        
        if start == prev: ranges.append(f"{start}")
        else: ranges.append(f"{start}-{prev}")
            
        range_str = ", ".join(ranges)

        keeper_info = occurrences[-1]
        keeper_elem = keeper_info['element']
        keeper_elem.set('data-layer-range', range_str)

        for item in occurrences[:-1]:
            parent = item['parent']
            child = item['element']
            try:
                parent.remove(child)
                removed_count += 1
            except ValueError:
                pass 
        
        processed_count += 1

    try:
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"    [Info] 优化完成！输出: {os.path.basename(output_file)}")
    except Exception as e:
        print(f"    [错误] 写入文件失败: {e}")
    return True

if __name__ == "__main__":
    base_out = "data/03_output"
    print("========== [4] 开始运行 SVG 跨层特征优化脚本 ==========")
    if os.path.exists(base_out):
        projects = [d for d in os.listdir(base_out) if os.path.isdir(os.path.join(base_out, d))]
        
        if len(sys.argv) > 1:
            projects = [p for p in projects if p == sys.argv[1]]
            
        for project_name in projects:
            print(f"\n>>> 正在优化项目 [{project_name}] ...")
            in_dir = f"{base_out}/{project_name}/merged_svg"
            out_dir = f"{base_out}/{project_name}/optimized_svg"
            
            if not os.path.exists(in_dir):
                print(f"  [警告] 找不到输入目录: {in_dir}")
                continue
                
            os.makedirs(out_dir, exist_ok=True)
            
            optimize_svg_circles(os.path.join(in_dir, 'view_X.svg'), os.path.join(out_dir, 'view_X_Optimized.svg'))
            optimize_svg_circles(os.path.join(in_dir, 'view_Y.svg'), os.path.join(out_dir, 'view_Y_Optimized.svg'))
            optimize_svg_circles(os.path.join(in_dir, 'view_Z.svg'), os.path.join(out_dir, 'view_Z_Optimized.svg'))
    else:
        print(f"⚠️ [错误] 找不到目录: {base_out}")
    print("========== [4] 脚本运行结束 ==========")