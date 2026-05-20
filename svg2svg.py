import os
import glob
import re
import xml.etree.ElementTree as ET
import numpy as np
from svgpathtools import parse_path
from shapely.geometry import Polygon

# 注册命名空间
ET.register_namespace('', "http://www.w3.org/2000/svg")
NS = {'svg': 'http://www.w3.org/2000/svg'}

class SmartFitter:
    def __init__(self, precision=2):
        self.precision = precision

    def _fmt(self, num):
        """格式化数字"""
        return f"{float(num):.{self.precision}f}"

    def optimize_path_string(self, d_string):
        """正则优化小数位数"""
        pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
        def repl(match):
            return self._fmt(match.group())
        return pattern.sub(repl, d_string)

    def fit(self, d_string):
        try:
            path_obj = parse_path(d_string)
            if not path_obj.isclosed(): 
                return None
            
            # 1. 提取所有点坐标
            coords = []
            for segment in path_obj:
                p = segment.start
                coords.append([p.real, p.imag])
            # 确保闭合点
            coords.append([path_obj[-1].end.real, path_obj[-1].end.imag])
            
            pts = np.array(coords)
            # 去除相邻重复点（Shapely 甚至 numpy 计算时可能会有问题）
            # 简单的去重逻辑，防止除零错误
            if len(pts) > 1:
                diff = pts[1:] - pts[:-1]
                mask = np.sum(diff**2, axis=1) > 1e-10 # 过滤极小距离
                pts = pts[:-1][mask]
                # 重新闭合
                pts = np.vstack([pts, pts[0]])

            if len(pts) < 3: return None

            poly = Polygon(pts)
            if not poly.is_valid: poly = poly.buffer(0)
            
            min_x, min_y, max_x, max_y = poly.bounds
            width, height = max_x - min_x, max_y - min_y
            
            # 避免除零
            if height == 0: return None 
            aspect_ratio = width / height

            # ==========================================================
            # 改进算法 1: 更严格的圆形检测
            # ==========================================================
            # 只有长宽比非常接近 1:1 时才可能是圆 (0.95 - 1.05)
            if 0.95 < aspect_ratio < 1.05:
                # 计算几何中心
                center = np.mean(pts[:-1], axis=0)
                cx, cy = center[0], center[1]
                
                # 计算所有点到中心的距离
                distances = np.linalg.norm(pts[:-1] - center, axis=1)
                r_mean = np.mean(distances)
                r_std = np.std(distances)
                r_min = np.min(distances)
                r_max = np.max(distances)
                
                if r_mean > 1e-6: # 防止极小图形除零
                    # 判据 A: 变异系数 (原有逻辑，调低阈值到 2%)
                    cv = r_std / r_mean 
                    
                    # 判据 B: 极差率 (Max - Min) / Mean
                    # 圆形的极差极小；圆角正方形的直边和圆角距离差很大
                    # 正方形的直边距离圆心是 r，角距离圆心是 1.414r，差距 41%
                    # 即使是大圆角，差距通常也在 10% 以上。
                    # 我们设置阈值为 6% (0.06)，这足以容忍 3D 打印切片的抖动，但能排除圆角矩形
                    range_ratio = (r_max - r_min) / r_mean
                    
                    # 同时满足才认为是圆
                    if cv < 0.025 and range_ratio < 0.06:
                        return ('circle', {
                            'cx': self._fmt(cx),
                            'cy': self._fmt(cy),
                            'r': self._fmt(r_mean)
                        })

            # ==========================================================
            # 改进算法 2: 智能多边形简化
            # ==========================================================
            size = max(width, height)
            # 动态公差：根据模型尺寸调整，避免小细节丢失或大模型过度保留
            tolerance = max(0.02, size * 0.002) 
            simplified_poly = poly.simplify(tolerance, preserve_topology=True)
            
            simple_coords = list(simplified_poly.exterior.coords)
            
            # 转为字符串，注意去除末尾多余的闭合点以节省空间(SVG polygon会自动闭合)
            if len(simple_coords) > 0 and simple_coords[0] == simple_coords[-1]:
                simple_coords.pop() 
            
            points_str = " ".join([f"{self._fmt(x)},{self._fmt(y)}" for x, y in simple_coords])
            return ('polygon', {'points': points_str})

        except Exception as e:
            # print(f"Fit error: {e}") # 调试用
            return None

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 可以在这里控制精度，比如 precision=2
    fitter = SmartFitter(precision=2)
    svg_files = glob.glob(os.path.join(input_folder, "*.svg"))
    
    print(f"开始处理 {len(svg_files)} 个 SVG 文件...")

    for file_path in svg_files:
        filename = os.path.basename(file_path)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            paths = root.findall('.//svg:path', NS)
            if not paths: paths = root.findall('.//path')
            
            converted_count = 0
            
            for elem in paths:
                d_orig = elem.get('d')
                if not d_orig: continue

                # 尝试拟合
                fit_result = fitter.fit(d_orig)

                if fit_result:
                    # --- 情况 A: 成功拟合为 Circle 或 Polygon ---
                    tag_name, attrs = fit_result
                    
                    if '}' in elem.tag:
                        ns_prefix = elem.tag.split('}')[0] + '}'
                        elem.tag = ns_prefix + tag_name
                    else:
                        elem.tag = tag_name
                    
                    elem.attrib.clear()
                    for k, v in attrs.items():
                        elem.set(k, v)
                    
                    elem.set('fill', 'none')
                    elem.set('stroke', 'black')
                    elem.set('stroke-width', '0.5')
                    converted_count += 1
                else:
                    # --- 情况 B: 保持为 Path，但精简坐标数字 ---
                    
                    # 1. 使用正则替换 d 字符串中的长小数
                    new_d = fitter.optimize_path_string(d_orig)
                    elem.set('d', new_d)

                    # 2. 统一描边
                    elem.set('fill', 'none') # 如果需要去除填充，可加上这句
                    if 'stroke-width' not in elem.attrib:
                        elem.set('stroke-width', '0.2')
                    if 'stroke' not in elem.attrib:
                         elem.set('stroke', 'black')

            output_path = os.path.join(output_folder, filename)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)

        except Exception as e:
            print(f"  [Error] {filename}: {e}")

    print(f"全部完成！已输出至: {output_folder}")

if __name__ == "__main__":
    process_folder("./Out_X", "./Out_X_new")

    process_folder("./Out_Y", "./Out_Y_new")

    process_folder("./Out_Z", "./Out_Z_new")
