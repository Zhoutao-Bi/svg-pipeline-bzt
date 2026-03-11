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
# 模型读取 3.0 - 本地几何计算引擎 v33.0 (终极整合版)
# 1. 增强型特征识别：自动识别嵌套多边形（三通/内孔）
# 2. 自动圆心拟合：将多边形孔洞拟合为标准的圆孔特征
# 3. 终极空间重合对齐与 3D OpenSCAD 模型导出
# 4. 集成 Matplotlib：实时 3D 渲染与标准三视图展示
# ==========================================

class ModelExtractorV33:
    def __init__(self):
        self.raw_features = []
        self.raw_lines = {'Z': [], 'X': [], 'Y': []} 
        self.features_aligned = []
        self.lines_3d = {'Z': [], 'X': [], 'Y': []} 
        self.all_coords = {'x': [], 'y': [], 'z': []}

    def poly_area(self, pts):
        """鞋带公式计算多边形面积"""
        if len(pts) < 3: return 0.0
        area = 0.0; n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def get_centroid_and_dia(self, pts):
        """计算多边形的形心和等效直径（用于拟合多边形孔）"""
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
        """射线法判断点是否在多边形内部"""
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
                
                # 提取所有路径
                for shape in layer.find_all(['path', 'polygon']):
                    pts = self.extract_coords(shape.get('d', '') if shape.name == 'path' else shape.get('points', ''))
                    if pts: polygons.append(pts)

                if not polygons: continue

                # 按照面积从大到小排序，最大的是实体外轮廓，其余如果在其内部，则是孔
                sorted_polys = sorted(polygons, key=lambda p: self.poly_area(p), reverse=True)
                outer = sorted_polys[0]

                # 转换线框供渲染
                for p in sorted_polys:
                    current_line = []
                    for sx, sy in p:
                        if axis == 'Z': gx, gy, gz = sx, -sy, depth
                        elif axis == 'X': gx, gy, gz = depth, sx, -sy
                        else: gx, gy, gz = sx, depth, -sy
                        current_line.append([gx, gy, gz])
                    self.raw_lines[axis].append(current_line)

                # 识别内嵌多边形孔 (三通/非标准 Circle)
                for p in sorted_polys[1:]:
                    if self.is_inside(p[0], outer):
                        center_xy, dia = self.get_centroid_and_dia(p)
                        if axis == 'Z': cx3, cy3, cz3 = center_xy[0], -center_xy[1], depth
                        elif axis == 'X': cx3, cy3, cz3 = depth, center_xy[0], -center_xy[1]
                        else: cx3, cy3, cz3 = center_xy[0], depth, -center_xy[1]
                        self.raw_features.append({
                            "Type": "Hole", "Axis": axis, "Center3D": [cx3, cy3, cz3], "R": dia/2.0
                        })

                # 处理标准 Circle
                for c in layer.find_all('circle'):
                    cx, cy, r = float(c['cx']), float(c['cy']), float(c['r'])
                    f_type = "Hole" if self.is_inside((cx, cy), outer) else "Pillar"
                    if axis == 'Z': cx3, cy3, cz3 = cx, -cy, depth
                    elif axis == 'X': cx3, cy3, cz3 = depth, cx, -cy
                    else: cx3, cy3, cz3 = cx, depth, -cy
                    self.raw_features.append({
                        "Type": f_type, "Axis": axis, "Center3D": [cx3, cy3, cz3], "R": round(r, 2)
                    })

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
        print("[*] 正在提取特征并导出 JSON/OpenSCAD...")
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

        final_solid_blocks = [{"ID": f"Solid_{i+1}", "Z_Range": [min(b["Z"]), max(b["Z"])], 
                               "Size_XY": [round(b["BBox"]["X_Max"]-b["BBox"]["X_Min"], 2), round(b["BBox"]["Y_Max"]-b["BBox"]["Y_Min"], 2)],
                               "Outer_Contour": b["Contour"]} for i, b in enumerate(blocks)]

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
                
                final_features.append({
                    "Axis": axis, center_key: [cx, cy], "Main_Diameter": main_step["Diameter"], "Steps": compact
                })
            return final_features

        final_data = {
            "Part_Overview": {"Bounding_Box_LWH": [round(max(self.all_coords[i])-min(self.all_coords[i]), 2) if self.all_coords[i] else 0.0 for i in 'xyz']},
            "Solid_Base_Layers": final_solid_blocks, "Positive_Pillars": format_steps(p_groups), "Negative_Holes": format_steps(h_groups)
        }
        with open("Full_Features_v33.json", 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
            
        print("[+] ✨ JSON 生成完毕。")

    def render_3d_and_views(self):
        """实时渲染核心：生成 3D 轴测图与主、俯、左三视图"""
        print("[*] 正在启动 Matplotlib 实时渲染引擎...")
        
        # 将所有轴向的线汇总
        all_lines = []
        for axis in ['X', 'Y', 'Z']:
            all_lines.extend(self.lines_3d[axis])
            
        if not all_lines:
            print("[-] 无渲染数据")
            return

        fig = plt.figure(figsize=(14, 10))
        fig.canvas.manager.set_window_title('零件逆向渲染 - v33.0 终极版')

        # 1. 3D 轴测图 (左上)
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_3d.set_title("3D Isometric View (实时轴测图)")
        ax_3d.set_xlabel('X (mm)'); ax_3d.set_ylabel('Y (mm)'); ax_3d.set_zlabel('Z (mm)')
        for line in all_lines:
            pts = np.array(line)
            ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='teal', alpha=0.6, linewidth=0.8)

        # 2. 俯视图 Top View (X-Y 平面，右上)
        ax_top = fig.add_subplot(2, 2, 2)
        ax_top.set_title("Top View (俯视图 / Z轴投影)")
        ax_top.set_xlabel('X'); ax_top.set_ylabel('Y')
        ax_top.set_aspect('equal')
        ax_top.grid(True, linestyle=':', alpha=0.5)
        for line in all_lines:
            pts = np.array(line)
            ax_top.plot(pts[:, 0], pts[:, 1], color='coral', alpha=0.5, linewidth=0.8)

        # 3. 主视图 Front View (X-Z 平面，左下)
        ax_front = fig.add_subplot(2, 2, 3)
        ax_front.set_title("Front View (主视图 / Y轴投影)")
        ax_front.set_xlabel('X'); ax_front.set_ylabel('Z')
        ax_front.set_aspect('equal')
        ax_front.grid(True, linestyle=':', alpha=0.5)
        for line in all_lines:
            pts = np.array(line)
            ax_front.plot(pts[:, 0], pts[:, 2], color='dodgerblue', alpha=0.5, linewidth=0.8)

        # 4. 左视图 Left View (Y-Z 平面，右下)
        ax_left = fig.add_subplot(2, 2, 4)
        ax_left.set_title("Left/Right View (侧视图 / X轴投影)")
        ax_left.set_xlabel('Y'); ax_left.set_ylabel('Z')
        ax_left.set_aspect('equal')
        ax_left.grid(True, linestyle=':', alpha=0.5)
        for line in all_lines:
            pts = np.array(line)
            ax_left.plot(pts[:, 1], pts[:, 2], color='mediumseagreen', alpha=0.5, linewidth=0.8)

        plt.tight_layout()
        print("[+] 渲染完成！请在弹出的图形窗口中查看。")
        plt.show()
    def export_depth_mapped_views(self):
        """
        生成并保存带有深度信息（颜色映射）的 X, Y, Z 三视图
        类似于热力图的视觉效果，深度越深颜色越暗，越浅颜色越亮（基于 viridis colormap）
        """
        import matplotlib.cm as cm
        from matplotlib.collections import LineCollection
        print("[*] 正在生成带有深度颜色的 X/Y/Z 三视图...")

        # 汇总所有对齐后的线段
        all_lines = []
        for axis in ['X', 'Y', 'Z']:
            all_lines.extend(self.lines_3d[axis])

        if not all_lines:
            print("[-] 无渲染数据，跳过生成深度图。")
            return

        def save_depth_view(view_name, x_idx, y_idx, depth_idx, filename):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 获取所有深度的极值，用于规范化颜色映射
            depths = [pt[depth_idx] for line in all_lines for pt in line]
            if not depths:
                plt.close(fig)
                return
            
            min_depth, max_depth = min(depths), max(depths)
            cmap = cm.viridis  # 使用和原图一致的 viridis 翠绿色系映射
            norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
            
            segments = []
            colors = []
            
            # 整理线段和对应的颜色
            for line in all_lines:
                pts = np.array(line)
                # 提取平面 2D 坐标作为线段
                segment = pts[:, [x_idx, y_idx]]
                segments.append(segment)
                # 计算该线段的平均深度并赋予颜色
                avg_depth = np.mean(pts[:, depth_idx])
                colors.append(cmap(norm(avg_depth)))
                
            # 使用 LineCollection 批量高效绘制带颜色的线段
            lc = LineCollection(segments, colors=colors, linewidths=1.0, alpha=0.8)
            ax.add_collection(lc)
            
            # 自动调整画面边界
            all_x = [pt[x_idx] for line in all_lines for pt in line]
            all_y = [pt[y_idx] for line in all_lines for pt in line]
            margin_x = (max(all_x) - min(all_x)) * 0.1 if all_x else 10
            margin_y = (max(all_y) - min(all_y)) * 0.1 if all_y else 10
            ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
            ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
            
            ax.set_aspect('equal')
            ax.set_title(f'View {view_name} (Auto-Centered)')
            
            # 添加右侧的颜色比例尺 (Colorbar)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(f'{view_name} Depth/Height (mm)')
            
            # 保存图片并关闭画布释放内存
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"[+] 已导出深度图: {filename}")

        # 生成 Z 视图 (顶视) - 投影面为 XY，深度为 Z
        save_depth_view('Z', x_idx=0, y_idx=1, depth_idx=2, filename='View_Z_Depth.png')
        
        # 生成 Y 视图 (主视) - 投影面为 XZ，深度为 Y
        save_depth_view('Y', x_idx=0, y_idx=2, depth_idx=1, filename='View_Y_Depth.png')
        
        # 生成 X 视图 (侧视) - 投影面为 YZ，深度为 X
        save_depth_view('X', x_idx=1, y_idx=2, depth_idx=0, filename='View_X_Depth.png')

if __name__ == "__main__":
    engine = ModelExtractorV33()
    engine.parse_all()         # 1. 读取并识别所有几何(包括隐形三通孔)
    engine.align_coordinates() # 2. 坐标对齐居中
    engine.export_json()       # 3. 输出 JSON 与 OpenSCAD 实体
    engine.export_depth_mapped_views()
    engine.render_3d_and_views() # 4. 实时渲染与三视图展示