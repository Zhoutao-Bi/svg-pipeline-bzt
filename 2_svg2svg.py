import os
import glob
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import parse_path
from shapely.geometry import Polygon

ET.register_namespace('', "http://www.w3.org/2000/svg")
NS = {'svg': 'http://www.w3.org/2000/svg'}

class SmartFitter:
    def __init__(self, precision=2):
        self.precision = precision
    def _fmt(self, num):
        return f"{float(num):.{self.precision}f}"
    def optimize_path_string(self, d_string):
        pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
        def repl(match): return self._fmt(match.group())
        return pattern.sub(repl, d_string)
    def fit(self, d_string):
        try:
            path_obj = parse_path(d_string)
            if not path_obj.isclosed(): return None
            coords = []
            for segment in path_obj:
                p = segment.start
                coords.append([p.real, p.imag])
            coords.append([path_obj[-1].end.real, path_obj[-1].end.imag])
            pts = np.array(coords)
            if len(pts) > 1:
                diff = pts[1:] - pts[:-1]
                mask = np.sum(diff**2, axis=1) > 1e-10 
                pts = pts[:-1][mask]
                pts = np.vstack([pts, pts[0]])
            if len(pts) < 3: return None
            poly = Polygon(pts)
            if not poly.is_valid: poly = poly.buffer(0)
            min_x, min_y, max_x, max_y = poly.bounds
            width, height = max_x - min_x, max_y - min_y
            if height == 0: return None 
            aspect_ratio = width / height
            if 0.95 < aspect_ratio < 1.05:
                center = np.mean(pts[:-1], axis=0)
                cx, cy = center[0], center[1]
                distances = np.linalg.norm(pts[:-1] - center, axis=1)
                r_mean = np.mean(distances)
                r_std = np.std(distances)
                r_min = np.min(distances)
                r_max = np.max(distances)
                if r_mean > 1e-6: 
                    cv = r_std / r_mean 
                    range_ratio = (r_max - r_min) / r_mean
                    if cv < 0.025 and range_ratio < 0.06:
                        return ('circle', {'cx': self._fmt(cx), 'cy': self._fmt(cy), 'r': self._fmt(r_mean)})
            size = max(width, height)
            tolerance = max(0.02, size * 0.002) 
            simplified_poly = poly.simplify(tolerance, preserve_topology=True)
            simple_coords = list(simplified_poly.exterior.coords)
            if len(simple_coords) > 0 and simple_coords[0] == simple_coords[-1]:
                simple_coords.pop() 
            points_str = " ".join([f"{self._fmt(x)},{self._fmt(y)}" for x, y in simple_coords])
            return ('polygon', {'points': points_str})
        except Exception as e:
            return None

def process_folder(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"    [警告] 输入目录不存在，跳过: {input_folder}")
        return
        
    os.makedirs(output_folder, exist_ok=True)
    fitter = SmartFitter(precision=2)
    svg_files = glob.glob(os.path.join(input_folder, "*.svg"))
    
    if not svg_files:
        return
        
    print(f"    [Info] 开始拟合处理 {len(svg_files)} 个 SVG 文件...")

    for file_path in svg_files:
        filename = os.path.basename(file_path)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            paths = root.findall('.//svg:path', NS)
            if not paths: paths = root.findall('.//path')
            
            for elem in paths:
                d_orig = elem.get('d')
                if not d_orig: continue

                fit_result = fitter.fit(d_orig)

                if fit_result:
                    tag_name, attrs = fit_result
                    if '}' in elem.tag:
                        ns_prefix = elem.tag.split('}')[0] + '}'
                        elem.tag = ns_prefix + tag_name
                    else:
                        elem.tag = tag_name
                    
                    elem.attrib.clear()
                    for k, v in attrs.items(): elem.set(k, v)
                    elem.set('fill', 'none')
                    elem.set('stroke', 'black')
                    elem.set('stroke-width', '0.5')
                else:
                    new_d = fitter.optimize_path_string(d_orig)
                    elem.set('d', new_d)
                    elem.set('fill', 'none') 
                    if 'stroke-width' not in elem.attrib: elem.set('stroke-width', '0.2')
                    if 'stroke' not in elem.attrib: elem.set('stroke', 'black')

            output_path = os.path.join(output_folder, filename)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
        except Exception as e:
            print(f"    [错误] 解析文件 {filename} 失败: {e}")

if __name__ == "__main__":
    base_temp = "data/02_temp"
    print(f"========== [2] 开始运行 SVG 几何拟合脚本 ==========")
    if os.path.exists(base_temp):
        projects = [d for d in os.listdir(base_temp) if os.path.isdir(os.path.join(base_temp, d))]
        
        if len(sys.argv) > 1:
            projects = [p for p in projects if p == sys.argv[1]]
            
        for project_name in projects:
            print(f"\n>>> 正在拟合项目 [{project_name}] 的切片...")
            raw_dir = f"{base_temp}/{project_name}/raw_slices"
            fit_dir = f"{base_temp}/{project_name}/fit_slices"
            
            print("  --- 处理 X 轴 ---")
            process_folder(f"{raw_dir}/X", f"{fit_dir}/X")
            print("  --- 处理 Y 轴 ---")
            process_folder(f"{raw_dir}/Y", f"{fit_dir}/Y")
            print("  --- 处理 Z 轴 ---")
            process_folder(f"{raw_dir}/Z", f"{fit_dir}/Z")
    else:
        print(f"⚠️ [错误] 未找到目录: {base_temp}")
    print(f"========== [2] 脚本运行结束 ==========")