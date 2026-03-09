import re
import os
import json
import math
from bs4 import BeautifulSoup
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ==========================================
# 模型读取 3.0 - 本地几何计算引擎 v31.0
# 终极空间对齐版：修复三视图重合问题，引入全局自动居中(Auto-Centering)
# ==========================================

class ModelExtractorV31:
    def __init__(self):
        self.raw_features = []
        self.raw_lines = {'Z': [], 'X': [], 'Y': []} 
        
        # 对齐后的标准数据
        self.features_aligned = []
        self.lines_3d = {'Z': [], 'X': [], 'Y': []} 
        self.all_coords = {'x': [], 'y': [], 'z': []}

    def extract_coords(self, text):
        nums = [float(n) for n in re.findall(r'[-+]?\d*\.\d+|\d+', text)]
        return [(round(nums[i], 2), round(nums[i+1], 2)) for i in range(0, len(nums)-1, 2)]

    def extract_true_depth(self, layer_tag):
        desc = layer_tag.find('desc')
        if not desc: return 0.0
        match = re.search(r'(?:Length_Dir|H):\s*([-+]?\d*\.?\d+)', str(desc))
        return float(match.group(1)) if match else 0.0

    def is_inside(self, point, poly):
        x, y = point; inside = False; n = len(poly)
        if n < 3: return False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def parse_all(self):
        """解析原始数据，应用正确的投影轴映射"""
        for axis in ['Z', 'X', 'Y']:
            path = f"Out_{axis}.txt"
            if not os.path.exists(path): continue
            print(f"[*] 正在解析 {axis} 轴图层...")
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'xml')
            
            for layer in soup.find_all('svg', id=lambda x: x and x.startswith('layer_')):
                depth = self.extract_true_depth(layer)
                polygons = []
                
                for shape in layer.find_all(['path', 'polygon']):
                    pts = self.extract_coords(shape.get('d', '') if shape.name == 'path' else shape.get('points', ''))
                    if pts:
                        polygons.append(pts)
                        current_line = []
                        for sx, sy in pts:
                            # 修正 SVG Y轴向下造成的镜像倒置，统一映射到绝对 3D 空间
                            if axis == 'Z': gx, gy, gz = sx, -sy, depth
                            elif axis == 'X': gx, gy, gz = depth, sx, -sy
                            else: gx, gy, gz = sx, depth, -sy
                            current_line.append([gx, gy, gz])
                        self.raw_lines[axis].append(current_line)

                for c in layer.find_all('circle'):
                    cx, cy, r = float(c['cx']), float(c['cy']), float(c['r'])
                    f_type = "Hole" if any(self.is_inside((cx, cy), poly) for poly in polygons) else "Pillar"
                    
                    if axis == 'Z': cx3, cy3, cz3 = cx, -cy, depth
                    elif axis == 'X': cx3, cy3, cz3 = depth, cx, -cy
                    else: cx3, cy3, cz3 = cx, depth, -cy

                    self.raw_features.append({
                        "Type": f_type, "Axis": axis, "Center3D": [cx3, cy3, cz3], "R": round(r, 2)
                    })

                    circle_line = []
                    for i in range(37):
                        theta = (2.0 * math.pi * i) / 36
                        sx, sy = cx + r * math.cos(theta), cy + r * math.sin(theta)
                        if axis == 'Z': gx, gy, gz = sx, -sy, depth
                        elif axis == 'X': gx, gy, gz = depth, sx, -sy
                        else: gx, gy, gz = sx, depth, -sy
                        circle_line.append([gx, gy, gz])
                    self.raw_lines[axis].append(circle_line)

    def align_coordinates(self):
        """【核心修复】三视图独立包围盒居中，确保 100% 空间重合"""
        print("[*] 正在执行全局 3D 坐标居中与重合对齐...")
        for axis in ['Z', 'X', 'Y']:
            lines = self.raw_lines[axis]
            if not lines: continue
            
            # 计算该视图的绝对包围盒
            xs, ys, zs = [], [], []
            for line in lines:
                for px, py, pz in line:
                    xs.append(px); ys.append(py); zs.append(pz)
                    
            off_x = (max(xs) + min(xs)) / 2
            off_y = (max(ys) + min(ys)) / 2
            off_z = (max(zs) + min(zs)) / 2
            
            # 平移所有线段至重心 (0,0,0)
            for line in lines:
                aligned_line = [[round(px-off_x, 2), round(py-off_y, 2), round(pz-off_z, 2)] for px, py, pz in line]
                self.lines_3d[axis].append(aligned_line)
                for pt in aligned_line:
                    self.all_coords['x'].append(pt[0]); self.all_coords['y'].append(pt[1]); self.all_coords['z'].append(pt[2])
                    
            # 平移该轴的所有孔柱特征
            for f in self.raw_features:
                if f['Axis'] == axis:
                    c3d = f["Center3D"]
                    f["Center3D_Aligned"] = [round(c3d[0]-off_x, 2), round(c3d[1]-off_y, 2), round(c3d[2]-off_z, 2)]
                    self.features_aligned.append(f)

    def refine_depth_by_cross_views(self, feat_axis, c_x, c_y, c_z, r, approx_min, approx_max):
        """基于统一对齐坐标系的交叉测深"""
        candidates = []
        tolerance = 2.0
        search_axes = ['X', 'Y', 'Z']
        search_axes.remove(feat_axis)

        for s_axis in search_axes:
            for line in self.lines_3d[s_axis]:
                for px, py, pz in line:
                    if feat_axis == 'Z':
                        if abs(px - c_x) <= r + tolerance and abs(py - c_y) <= r + tolerance:
                            if approx_min - 5.0 <= pz <= approx_max + 5.0: candidates.append(pz)
                    elif feat_axis == 'X':
                        if abs(py - c_y) <= r + tolerance and abs(pz - c_z) <= r + tolerance:
                            if approx_min - 5.0 <= px <= approx_max + 5.0: candidates.append(px)
                    elif feat_axis == 'Y':
                        if abs(px - c_x) <= r + tolerance and abs(pz - c_z) <= r + tolerance:
                            if approx_min - 5.0 <= py <= approx_max + 5.0: candidates.append(py)

        if candidates: return round(min(candidates), 2), round(max(candidates), 2)
        return round(approx_min, 2), round(approx_max, 2)

    def export_json(self):
        print("[*] 正在导出结构化 JSON...")
        # 1. 动态生成 Z 轴台阶 (基于对齐后的 3D 坐标)
        z_layers = defaultdict(list)
        for line in self.lines_3d['Z']:
            z_layers[round(line[0][2], 2)].extend(line)
            
        sorted_z = sorted(z_layers.keys()); blocks, cur = [], None
        for d in sorted_z:
            pts = z_layers[d]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            bbox = {"X_Min": min(xs), "X_Max": max(xs), "Y_Min": min(ys), "Y_Max": max(ys)}
            
            if not cur: cur = {"Z": [d], "BBox": bbox}
            else:
                if any(abs(bbox[k]-cur["BBox"][k])>2.0 for k in bbox):
                    blocks.append(cur); cur = {"Z": [d], "BBox": bbox}
                else: cur["Z"].append(d)
        if cur: blocks.append(cur)

        # 2. 特征聚合与处理
        h_groups = defaultdict(list); p_groups = defaultdict(list)
        for f in self.features_aligned:
            # 统一使用平面的绝对坐标作为 Hash Key 聚合
            c3d = f["Center3D_Aligned"]
            if f['Axis'] == 'Z': key = f"Z_{c3d[0]}_{c3d[1]}"
            elif f['Axis'] == 'X': key = f"X_{c3d[1]}_{c3d[2]}"
            else: key = f"Y_{c3d[0]}_{c3d[2]}"
            
            if f["Type"] == "Hole": h_groups[key].append(f)
            else: p_groups[key].append(f)

        def format_precise_steps(feature_groups):
            final_features = []
            for k, v in feature_groups.items():
                axis = v[0]['Axis']
                c3d_ref = v[0]["Center3D_Aligned"]
                
                # 获取 2D 投影中心
                if axis == 'Z': center_key = "Center_XY"; cx, cy = c3d_ref[0], c3d_ref[1]
                elif axis == 'X': center_key = "Center_YZ"; cx, cy = c3d_ref[1], c3d_ref[2]
                else: center_key = "Center_XZ"; cx, cy = c3d_ref[0], c3d_ref[2]
                
                # 获取深度轴数据
                depth_idx = {'Z':2, 'X':0, 'Y':1}[axis]
                steps = sorted(v, key=lambda x: x["Center3D_Aligned"][depth_idx])
                
                compact = []
                for s in steps:
                    d_val = s["Center3D_Aligned"][depth_idx]
                    dia = round(s["R"]*2, 2)
                    if not compact or compact[-1]["Diameter"] != dia:
                        compact.append({"Start": d_val, "End": d_val, "Diameter": dia, "_r": s["R"], "_c3d": s["Center3D_Aligned"]})
                    else: compact[-1]["End"] = d_val
                
                # 精确测深
                for c in compact:
                    ex_min, ex_max = self.refine_depth_by_cross_views(axis, c["_c3d"][0], c["_c3d"][1], c["_c3d"][2], c["_r"], c["Start"], c["End"])
                    if axis == 'Z': c["Z_Start"] = ex_min; c["Z_End"] = ex_max
                    elif axis == 'X': c["X_Start"] = ex_min; c["X_End"] = ex_max
                    else: c["Y_Start"] = ex_min; c["Y_End"] = ex_max
                    del c["Start"]; del c["End"]; del c["_r"]; del c["_c3d"]
                
                final_features.append({"Axis": axis, center_key: [cx, cy], "Steps": compact})
            return final_features

        final_data = {
            "Part_Overview": {
                "Bounding_Box_LWH": [round(max(self.all_coords[i])-min(self.all_coords[i]), 2) if self.all_coords[i] else 0.0 for i in 'xyz'],
                "Note": "Coordinates are aligned to geometric center (0,0,0)"
            },
            "Solid_Base_Layers": [{"ID": f"Solid_{i+1}", "Z_Range": [min(b["Z"]), max(b["Z"])], "Size_XY": [round(b["BBox"]["X_Max"]-b["BBox"]["X_Min"], 2), round(b["BBox"]["Y_Max"]-b["BBox"]["Y_Min"], 2)]} for i, b in enumerate(blocks)],
            "Positive_Pillars": format_precise_steps(p_groups),
            "Negative_Holes": format_precise_steps(h_groups)
        }
        with open("Full_Features_v31.json", 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print("[+] JSON 生成完毕！")

    def generate_2d_views(self):
        print("[*] 正在生成完美对齐的 XYZ 三视图...")
        for v_axis in ['X', 'Y', 'Z']:
            lines = self.lines_3d[v_axis]
            if not lines: continue
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            
            idx = {'X':0, 'Y':1, 'Z':2}[v_axis]
            all_d = [line[0][idx] for line in lines if line]
            if not all_d: continue
            
            vmin, vmax = min(all_d), max(all_d)
            if vmin == vmax: vmax += 0.1 
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.viridis
            
            for line in lines:
                if not line: continue
                xs, ys, zs = zip(*line)
                if v_axis == 'Z': px, py, d = xs, ys, zs[0]
                elif v_axis == 'X': px, py, d = ys, zs, xs[0]
                else: px, py, d = xs, zs, ys[0]
                ax.plot(px, py, color=cmap(norm(d)), linewidth=1, alpha=0.7)
                
            fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax).set_label(f'{v_axis} Depth/Height (mm)')
            ax.set_title(f'View {v_axis} (Auto-Centered)'); ax.axis('equal'); plt.savefig(f'{v_axis}.png'); plt.close()

    def show_3d_render(self):
        print("[*] 正在启动 3D 实时渲染窗口...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = {'Z': '#2ca02c', 'X': '#1f77b4', 'Y': '#d62728'}
        labels = {'Z': 'Z View Data (Top)', 'X': 'X View Data (Side)', 'Y': 'Y View Data (Front)'}
        
        has_lines = False
        for axis, lines in self.lines_3d.items():
            first = True
            for line in lines:
                if not line: continue
                has_lines = True
                xs, ys, zs = zip(*line)
                label = labels[axis] if first else ""
                ax.plot(xs, ys, zs, color=colors[axis], alpha=0.5, lw=1, label=label)
                first = False
                
        if has_lines:
            ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
            ax.set_title('Real-time 3D Wireframe Rendering (Perfectly Aligned)')
            ax.legend()
            try: ax.set_box_aspect([1, 1, 0.3]) 
            except Exception: pass 
            plt.show()

if __name__ == "__main__":
    engine = ModelExtractorV31()
    engine.parse_all()
    engine.align_coordinates() # <- 核心魔法发生在这里
    engine.export_json()
    engine.generate_2d_views()
    engine.show_3d_render()