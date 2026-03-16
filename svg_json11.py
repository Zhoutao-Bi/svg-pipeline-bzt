import re
import os
import json
import math
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 模型读取 3.0 - 本地几何计算引擎 v33.0 (终极瘦身无弹窗版)
# 核心优化：彻底删除 Outer_Contour，只保留核心尺寸，解决 Dify 输出崩溃
# ==========================================

class ModelExtractorV33:
    def __init__(self):
        self.raw_features = []
        self.raw_lines = {'Z': [], 'X': [], 'Y': []} 
        self.features_aligned = []
        self.lines_3d = {'Z': [], 'X': [], 'Y': []} 
        self.all_coords = {'x': [], 'y': [], 'z': []}

    def poly_area(self, pts):
        if len(pts) < 3: return 0.0
        area = 0.0; n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def get_centroid_and_dia(self, pts):
        if not pts: return [0, 0], 0
        arr = np.array(pts)
        centroid = np.mean(arr, axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)
        avg_r = np.mean(dists)
        return centroid.tolist(), round(avg_r * 2, 2)

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
                    if pts: polygons.append(pts)
                if not polygons: continue
                sorted_polys = sorted(polygons, key=lambda p: self.poly_area(p), reverse=True)
                outer = sorted_polys[0]
                for p in sorted_polys:
                    current_line = []
                    for sx, sy in p:
                        if axis == 'Z': gx, gy, gz = sx, -sy, depth
                        elif axis == 'X': gx, gy, gz = depth, sx, -sy
                        else: gx, gy, gz = sx, depth, -sy
                        current_line.append([gx, gy, gz])
                    self.raw_lines[axis].append(current_line)
                for p in sorted_polys[1:]:
                    if self.is_inside(p[0], outer):
                        center_xy, dia = self.get_centroid_and_dia(p)
                        if axis == 'Z': cx3, cy3, cz3 = center_xy[0], -center_xy[1], depth
                        elif axis == 'X': cx3, cy3, cz3 = depth, center_xy[0], -center_xy[1]
                        else: cx3, cy3, cz3 = center_xy[0], depth, -center_xy[1]
                        self.raw_features.append({"Type": "Hole", "Axis": axis, "Center3D": [cx3, cy3, cz3], "R": dia/2.0})
                for c in layer.find_all('circle'):
                    cx, cy, r = float(c['cx']), float(c['cy']), float(c['r'])
                    f_type = "Hole" if self.is_inside((cx, cy), outer) else "Pillar"
                    if axis == 'Z': cx3, cy3, cz3 = cx, -cy, depth
                    elif axis == 'X': cx3, cy3, cz3 = depth, cx, -cy
                    else: cx3, cy3, cz3 = cx, depth, -cy
                    self.raw_features.append({"Type": f_type, "Axis": axis, "Center3D": [cx3, cy3, cz3], "R": round(r, 2)})

    def align_coordinates(self):
        print("[*] 正在执行全局 3D 坐标居中...")
        for axis in ['Z', 'X', 'Y']:
            lines = self.raw_lines[axis]
            if not lines: continue
            xs, ys, zs = [], [], []
            for line in lines:
                for px, py, pz in line:
                    xs.append(px); ys.append(py); zs.append(pz)
            if not xs: continue
            off_x, off_y, off_z = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2, (max(zs)+min(zs))/2
            for line in lines:
                aligned_line = [[round(px-off_x, 2), round(py-off_y, 2), round(pz-off_z, 2)] for px, py, pz in line]
                self.lines_3d[axis].append(aligned_line)
                for pt in aligned_line:
                    self.all_coords['x'].append(pt[0]); self.all_coords['y'].append(pt[1]); self.all_coords['z'].append(pt[2])
            for f in self.raw_features:
                if f['Axis'] == axis:
                    c3d = f["Center3D"]
                    f["Center3D_Aligned"] = [round(c3d[0]-off_x, 2), round(c3d[1]-off_y, 2), round(c3d[2]-off_z, 2)]
                    self.features_aligned.append(f)

    def refine_depth_by_cross_views(self, feat_axis, c_x, c_y, c_z, r, approx_min, approx_max):
        candidates = []
        tolerance = 2.0
        search_axes = ['X', 'Y', 'Z']
        search_axes.remove(feat_axis)
        for s_axis in search_axes:
            for line in self.lines_3d[s_axis]:
                for px, py, pz in line:
                    if feat_axis == 'Z' and abs(px - c_x) <= r + tolerance and abs(py - c_y) <= r + tolerance:
                        if approx_min - 5.0 <= pz <= approx_max + 5.0: candidates.append(pz)
                    elif feat_axis == 'X' and abs(py - c_y) <= r + tolerance and abs(pz - c_z) <= r + tolerance:
                        if approx_min - 5.0 <= px <= approx_max + 5.0: candidates.append(px)
                    elif feat_axis == 'Y' and abs(px - c_x) <= r + tolerance and abs(pz - c_z) <= r + tolerance:
                        if approx_min - 5.0 <= py <= approx_max + 5.0: candidates.append(py)
        if candidates: return round(min(candidates), 2), round(max(candidates), 2)
        return round(approx_min, 2), round(approx_max, 2)

    def export_json(self):
        print("[*] 正在提取特征并导出极致精简版 JSON...")
        z_layers = defaultdict(list)
        for line in self.lines_3d['Z']:
            if line: z_layers[round(line[0][2], 2)].append(line)
        sorted_z = sorted(z_layers.keys()); blocks, cur = [], None
        for d in sorted_z:
            lines = z_layers[d]
            outer_line = max(lines, key=lambda l: self.poly_area(l)) if lines else []
            if not outer_line: continue
            xs, ys = [p[0] for p in outer_line], [p[1] for p in outer_line]
            bbox = {"X_Min": min(xs), "X_Max": max(xs), "Y_Min": min(ys), "Y_Max": max(ys)}
            contour = [[round(x, 2), round(y, 2)] for x, y in zip(xs, ys)]
            if not cur: cur = {"Z": [d], "BBox": bbox, "Contour": contour}
            else:
                if any(abs(bbox[k]-cur["BBox"][k])>2.0 for k in bbox):
                    blocks.append(cur); cur = {"Z": [d], "BBox": bbox, "Contour": contour}
                else: cur["Z"].append(d)
        if cur: blocks.append(cur)

        # 核心优化：彻底删除 Outer_Contour，只保留 Z 轴范围和 XY 尺寸
        final_solid_blocks = [{"ID": f"Solid_{i+1}", 
                               "Z_Range": [min(b["Z"]), max(b["Z"])], 
                               "Size_XY": [round(b["BBox"]["X_Max"]-b["BBox"]["X_Min"], 2), round(b["BBox"]["Y_Max"]-b["BBox"]["Y_Min"], 2)]
                              } for i, b in enumerate(blocks)]

        h_groups = defaultdict(list); p_groups = defaultdict(list)
        for f in self.features_aligned:
            c3d = f["Center3D_Aligned"]
            if f['Axis'] == 'Z': key = f"Z_{c3d[0]}_{c3d[1]}"
            elif f['Axis'] == 'X': key = f"X_{c3d[1]}_{c3d[2]}"
            else: key = f"Y_{c3d[0]}_{c3d[2]}"
            if f["Type"] == "Hole": h_groups[key].append(f)
            else: p_groups[key].append(f)

        def format_steps(groups):
            final_features = []
            for k, v in groups.items():
                axis = v[0]['Axis']; c3d_ref = v[0]["Center3D_Aligned"]
                cx, cy = (c3d_ref[0], c3d_ref[1]) if axis == 'Z' else ((c3d_ref[1], c3d_ref[2]) if axis == 'X' else (c3d_ref[0], c3d_ref[2]))
                center_key = "Center_XY" if axis == 'Z' else ("Center_YZ" if axis == 'X' else "Center_XZ")
                depth_idx = {'Z':2, 'X':0, 'Y':1}[axis]
                steps = sorted(v, key=lambda x: x["Center3D_Aligned"][depth_idx])
                compact = []
                for s in steps:
                    d_val = s["Center3D_Aligned"][depth_idx]
                    dia = round(s["R"]*2, 2)
                    if not compact or compact[-1]["Diameter"] != dia:
                        compact.append({"Start": d_val, "End": d_val, "Diameter": dia, "_r": s["R"], "_c3d": s["Center3D_Aligned"]})
                    else: compact[-1]["End"] = d_val
                main_step = max(compact, key=lambda x: x["End"] - x["Start"]) if compact else {"Diameter": 0}
                for c in compact:
                    ex_min, ex_max = self.refine_depth_by_cross_views(axis, c["_c3d"][0], c["_c3d"][1], c["_c3d"][2], c["_r"], c["Start"], c["End"])
                    if axis == 'Z': c["Z_Start"] = ex_min; c["Z_End"] = ex_max
                    elif axis == 'X': c["X_Start"] = ex_min; c["X_End"] = ex_max
                    else: c["Y_Start"] = ex_min; c["Y_End"] = ex_max
                    del c["Start"]; del c["End"]; del c["_r"]; del c["_c3d"]
                final_features.append({"Axis": axis, center_key: [cx, cy], "Main_Diameter": main_step["Diameter"], "Steps": compact})
            return final_features

        final_data = {
            "Part_Overview": {"Bounding_Box_LWH": [round(max(self.all_coords[i])-min(self.all_coords[i]), 2) if self.all_coords[i] else 0.0 for i in 'xyz']},
            "Solid_Base_Layers": final_solid_blocks, 
            "Positive_Pillars": format_steps(p_groups), 
            "Negative_Holes": format_steps(h_groups)
        }
        
        with open("Full_Features_v33_minified.json", 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, separators=(',', ':'))
        print("[+] JSON 特征提取完毕，已实现极致瘦身！")

    def render_3d_and_views(self):
        print("[*] 正在生成 3D 轴测图与标准视图图片...")
        all_lines = []
        for axis in ['X', 'Y', 'Z']: all_lines.extend(self.lines_3d[axis])
        if not all_lines: return
        plt.ioff()
        fig = plt.figure(figsize=(14, 10))
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        for line in all_lines:
            pts = np.array(line); ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='teal', alpha=0.6, linewidth=0.8)
        ax_top = fig.add_subplot(2, 2, 2)
        for line in all_lines:
            pts = np.array(line); ax_top.plot(pts[:, 0], pts[:, 1], color='coral', alpha=0.5, linewidth=0.8)
        ax_front = fig.add_subplot(2, 2, 3)
        for line in all_lines:
            pts = np.array(line); ax_front.plot(pts[:, 0], pts[:, 2], color='dodgerblue', alpha=0.5, linewidth=0.8)
        ax_left = fig.add_subplot(2, 2, 4)
        for line in all_lines:
            pts = np.array(line); ax_left.plot(pts[:, 1], pts[:, 2], color='mediumseagreen', alpha=0.5, linewidth=0.8)
        plt.tight_layout()
        plt.savefig('Full_Isometric_View.png', dpi=150)
        plt.close(fig)

    def export_depth_mapped_views(self):
        import matplotlib.cm as cm
        from matplotlib.collections import LineCollection
        print("[*] 正在保存 X/Y/Z 三视图图片...")
        all_lines = []
        for axis in ['X', 'Y', 'Z']: all_lines.extend(self.lines_3d[axis])
        if not all_lines: return
        def save_depth_view(x_idx, y_idx, depth_idx, filename):
            plt.ioff()
            fig, ax = plt.subplots(figsize=(10, 8))
            depths = [pt[depth_idx] for line in all_lines for pt in line]
            min_d, max_d = min(depths), max(depths)
            cmap, norm = cm.viridis, plt.Normalize(vmin=min_d, vmax=max_d)
            segments, colors = [], []
            for line in all_lines:
                pts = np.array(line); segments.append(pts[:, [x_idx, y_idx]])
                colors.append(cmap(norm(np.mean(pts[:, depth_idx]))))
            lc = LineCollection(segments, colors=colors, linewidths=1.0, alpha=0.8)
            ax.add_collection(lc)
            all_x = [pt[x_idx] for line in all_lines for pt in line]
            all_y = [pt[y_idx] for line in all_lines for pt in line]
            ax.set_xlim(min(all_x)*1.1, max(all_x)*1.1); ax.set_ylim(min(all_y)*1.1, max(all_y)*1.1)
            ax.set_aspect('equal')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        save_depth_view(0, 1, 2, 'View_Z_Depth.png')
        save_depth_view(0, 2, 1, 'View_Y_Depth.png')
        save_depth_view(1, 2, 0, 'View_X_Depth.png')

if __name__ == "__main__":
    engine = ModelExtractorV33()
    engine.parse_all()
    engine.align_coordinates()
    engine.export_json()
    engine.export_depth_mapped_views()
    engine.render_3d_and_views()
    print("[+] 全部自动化流程已完成，无弹窗卡顿。")