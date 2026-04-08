import re
import math
import numpy as np
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection


class ModelExtractorV33:
    def __init__(self):
        self.raw_features = []
        self.raw_lines = {"Z": [], "X": [], "Y": []}
        self.features_aligned = []
        self.lines_3d = {"Z": [], "X": [], "Y": []}
        self.all_coords = {"x": [], "y": [], "z": []}

    def poly_area(self, pts):
        if len(pts) < 3:
            return 0.0
        area = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def poly_perimeter(self, pts):
        """新增：计算多边形周长，用于过滤不规则的幽灵碎片"""
        if len(pts) < 3:
            return 0.0
        p = 0.0
        for i in range(len(pts)):
            p += math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        return p

    def get_centroid_and_dia(self, pts):
        if not pts:
            return [0, 0], 0
        arr = np.array(pts)
        centroid = np.mean(arr, axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)
        avg_r = np.mean(dists)
        return centroid.tolist(), round(avg_r * 2, 2)

    def extract_coords(self, text):
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", text or "")]
        return [(round(nums[i], 2), round(nums[i + 1], 2)) for i in range(0, len(nums) - 1, 2)]

    def parse_points_attr(self, text):
        pts = []
        if not text:
            return pts
        for token in text.strip().split():
            if "," not in token:
                continue
            sx, sy = token.split(",", 1)
            try:
                pts.append((round(float(sx), 2), round(float(sy), 2)))
            except ValueError:
                continue
        return pts

    def to_3d(self, axis, sx, sy, depth):
        if axis == "Z":
            return [sx, -sy, depth]
        if axis == "X":
            return [depth, sx, -sy]
        return [sx, depth, -sy]

    def extract_true_depth(self, layer_tag):
        desc = layer_tag.find("desc")
        if not desc:
            return 0.0
        match = re.search(r"(?:Length_Dir|H):\s*([-+]?\d*\.?\d+)", str(desc))
        return float(match.group(1)) if match else 0.0

    def is_inside(self, point, poly):
        x, y = point
        inside = False
        n = len(poly)
        if n < 3:
            return False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def generate_circle_pts(self, cx, cy, r, segments=36):
        pts = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            pts.append((round(cx + r * math.cos(angle), 2), round(cy + r * math.sin(angle), 2)))
        return pts

    # 提取指定轴的每层外轮廓（核心去网格逻辑）
    def get_axis_outer_contours(self, axis):
        """
        提取指定轴的每层外轮廓，只保留每层面积最大的外边缘，过滤内部线条
        axis: Z / X / Y
        return: 外轮廓3D线条列表
        """
        depth_idx_map = {"Z": 2, "X": 0, "Y": 1}
        depth_idx = depth_idx_map[axis]
        # 按深度值分层
        layer_dict = defaultdict(list)
        for line in self.lines_3d[axis]:
            if not line:
                continue
            depth_val = round(line[0][depth_idx], 2)
            layer_dict[depth_val].append(line)
        
        # 每层只保留面积最大的外轮廓
        outer_contours = []
        for depth, lines in layer_dict.items():
            if not lines:
                continue
            # 计算轮廓面积，取最大的作为外边缘
            outer_line = max(lines, key=lambda l: self.poly_area([(p[0], p[1]) for p in l]))
            outer_contours.append(outer_line)
        return outer_contours

    def parse_all(self):
        for axis in ["Z", "X", "Y"]:
            path = f"Out_{axis}.txt"
            if not os.path.exists(path):
                continue
            print(f"[*] Parsing {path} ...")
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "xml")
            for layer in soup.find_all("svg", id=lambda x: x and x.startswith("layer_")):
                depth = self.extract_true_depth(layer)
                unified_shapes = []
                # 解析多边形、矩形、路径
                for shape in layer.find_all(["polygon", "rect", "path"]):
                    pts = []
                    if shape.name == "polygon":
                        pts = self.parse_points_attr(shape.get("points", ""))
                    elif shape.name == "rect":
                        x, y = float(shape.get("x", 0.0)), float(shape.get("y", 0.0))
                        w, h = float(shape.get("width", 0.0)), float(shape.get("height", 0.0))
                        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    else:
                        pts = self.extract_coords(shape.get("d", ""))
                    if len(pts) >= 3:
                        unified_shapes.append({"type": "poly", "pts": pts})
                # 解析圆形
                for c in layer.find_all("circle"):
                    cx, cy, r = float(c.get("cx", 0.0)), float(c.get("cy", 0.0)), float(c.get("r", 0.0))
                    if r > 0:
                        pts = self.generate_circle_pts(cx, cy, r)
                        unified_shapes.append({"type": "circle", "pts": pts, "cx": cx, "cy": cy, "r": r})
                # 解析椭圆（修复原代码pts未定义的bug）
                for e in layer.find_all("ellipse"):
                    cx, cy = float(e.get("cx", 0.0)), float(e.get("cy", 0.0))
                    rx, ry = float(e.get("rx", 0.0)), float(e.get("ry", 0.0))
                    r = (rx + ry) / 2.0
                    if r > 0:
                        pts = self.generate_circle_pts(cx, cy, r)
                        unified_shapes.append({"type": "ellipse", "pts": pts, "cx": cx, "cy": cy, "r": round(r, 2)})
                
                # 第一阶段：保存渲染线条，并计算关键指标 (形心、面积与圆度)
                for shape_obj in unified_shapes:
                    pts = shape_obj["pts"]
                    self.raw_lines[axis].append([self.to_3d(axis, sx, sy, depth) for sx, sy in pts])
                    if shape_obj["type"] in ["circle", "ellipse"]:
                        shape_obj["centroid"] = (shape_obj["cx"], shape_obj["cy"])
                        shape_obj["dia"] = shape_obj["r"] * 2
                        shape_obj["circularity"] = 1.0  # 完美圆形
                        shape_obj["area"] = math.pi * (shape_obj["r"] ** 2) # [新增] 计算面积
                    else:
                        c_xy, dia = self.get_centroid_and_dia(pts)
                        shape_obj["centroid"] = tuple(c_xy)
                        shape_obj["dia"] = dia
                        area = self.poly_area(pts)
                        perim = self.poly_perimeter(pts)
                        shape_obj["area"] = area # [新增] 记录面积
                        # 计算圆度
                        shape_obj["circularity"] = (4 * math.pi * area) / (perim * perim) if perim > 0 else 0.0
                
                # [核心修复 1] 轮廓去重：消除 SVG 导出的重叠路径对奇偶校验的破坏
                unique_shapes = []
                for shape in unified_shapes:
                    if shape.get("area", 0) < 1e-3: continue # 过滤面积为0的幽灵形状
                    is_dup = False
                    for u in unique_shapes:
                        dist = math.hypot(shape["centroid"][0] - u["centroid"][0], shape["centroid"][1] - u["centroid"][1])
                        area_diff = abs(shape["area"] - u["area"])
                        # 如果圆心接近(容差0.5)且面积相差不到5%，判定为同一图层的重合轮廓
                        if dist < 0.5 and area_diff < max(shape["area"], u["area"]) * 0.05:
                            is_dup = True
                            break
                    if not is_dup:
                        unique_shapes.append(shape)
                
                # 第二阶段：形心奇偶校验与圆度过滤 (基于去重数据)
                for i, shape_obj in enumerate(unique_shapes):
                    # 斩杀幽灵孔：任何圆度低于 0.75 的非规则多边形直接跳过
                    if shape_obj["circularity"] < 0.75:
                        continue
                    inside_count = 0
                    for j, other_obj in enumerate(unique_shapes):
                        if i == j: continue
                        # [核心修复 2] 拓扑层级限制：包围我的外框，面积必须比我大至少 5%
                        if other_obj["area"] > shape_obj["area"] * 1.05:
                            if self.is_inside(shape_obj["centroid"], other_obj["pts"]):
                                inside_count += 1
                    c3d = self.to_3d(axis, shape_obj["centroid"][0], shape_obj["centroid"][1], depth)
                    r_val = round(shape_obj["dia"] / 2.0, 2)
                    # 奇数层级：在基座内部 -> 负向特征 (孔)
                    if inside_count % 2 != 0:
                        self.raw_features.append({"Type": "Hole", "Axis": axis, "Center3D": c3d, "R": r_val})
                    # 偶数层级：孤立存在(0) 或 孔内的凸起(2) -> 正向特征 (柱)
                    else:
                        # [核心修复 3] 移除对 circle/ellipse 的强制类型歧视
                        # 只要上方的 circularity >= 0.75 拦截通过，哪怕是用 polygon 画的圆柱也应该识别为 Pillar
                        self.raw_features.append({"Type": "Pillar", "Axis": axis, "Center3D": c3d, "R": r_val})

    def align_coordinates(self):
        print("[*] Aligning global 3D coordinates (Zero-based layout) ...")
        # 回滚至完美运行的按轴提取偏移量逻辑，彻底修复 refiner 切片落空的问题
        for axis in ["Z", "X", "Y"]:
            lines = self.raw_lines[axis]
            axis_features = [f for f in self.raw_features if f["Axis"] == axis]
            xs, ys, zs = [], [], []
            for line in lines:
                for px, py, pz in line:
                    xs.append(px)
                    ys.append(py)
                    zs.append(pz)
            for f in axis_features:
                cx, cy, cz = f["Center3D"]
                xs.append(cx)
                ys.append(cy)
                zs.append(cz)
            if not xs:
                continue
            off_x = min(xs)
            off_y = min(ys)
            off_z = min(zs)
            for line in lines:
                aligned_line = [[round(px - off_x, 2), round(py - off_y, 2), round(pz - off_z, 2)] for px, py, pz in line]
                self.lines_3d[axis].append(aligned_line)
                for pt in aligned_line:
                    self.all_coords["x"].append(pt[0])
                    self.all_coords["y"].append(pt[1])
                    self.all_coords["z"].append(pt[2])
            for f in axis_features:
                c3d = f["Center3D"]
                f["Center3D_Aligned"] = [round(c3d[0] - off_x, 2), round(c3d[1] - off_y, 2), round(c3d[2] - off_z, 2)]
                self.features_aligned.append(f)

    def refine_depth_by_cross_views(self, feat_axis, c_x, c_y, c_z, r, approx_min, approx_max):
        candidates = []
        tolerance = 2.0
        search_axes = ["X", "Y", "Z"]
        search_axes.remove(feat_axis)
        for s_axis in search_axes:
            for line in self.lines_3d[s_axis]:
                for px, py, pz in line:
                    if feat_axis == "Z" and abs(px - c_x) <= r + tolerance and abs(py - c_y) <= r + tolerance:
                        if approx_min - 5.0 <= pz <= approx_max + 5.0:
                            candidates.append(pz)
                    elif feat_axis == "X" and abs(py - c_y) <= r + tolerance and abs(pz - c_z) <= r + tolerance:
                        if approx_min - 5.0 <= px <= approx_max + 5.0:
                            candidates.append(px)
                    elif feat_axis == "Y" and abs(px - c_x) <= r + tolerance and abs(pz - c_z) <= r + tolerance:
                        if approx_min - 5.0 <= py <= approx_max + 5.0:
                            candidates.append(py)
        if candidates:
            return round(min(candidates), 2), round(max(candidates), 2)
        return round(approx_min, 2), round(approx_max, 2)

    # 空间容差聚类 (修复了之前的特征断裂现象)
    def cluster_features(self, features, tolerance=0.5):
        groups = []
        for f in features:
            placed = False
            c3d = f["Center3D_Aligned"]
            axis = f["Axis"]
            for group in groups:
                ref_feat = group[0]
                if ref_feat["Axis"] != axis: continue
                ref_c3d = ref_feat["Center3D_Aligned"]
                if axis == "Z": dist = math.hypot(c3d[0] - ref_c3d[0], c3d[1] - ref_c3d[1])
                elif axis == "X": dist = math.hypot(c3d[1] - ref_c3d[1], c3d[2] - ref_c3d[2])
                else: dist = math.hypot(c3d[0] - ref_c3d[0], c3d[2] - ref_c3d[2])
                if dist <= tolerance:
                    group.append(f)
                    placed = True
                    break
            if not placed:
                groups.append([f])
        result_dict = {}
        for i, group in enumerate(groups):
            result_dict[f"Group_{i}"] = group
        return result_dict

    def export_json(self):
        print("[*] Exporting Optimized JSON ...")
        z_layers = defaultdict(list)
        for line in self.lines_3d["Z"]:
            if line:
                z_layers[round(line[0][2], 2)].append(line)
        sorted_z = sorted(z_layers.keys())
        blocks, cur = [], None
        for d in sorted_z:
            lines = z_layers[d]
            outer_line = max(lines, key=lambda l: self.poly_area(l)) if lines else []
            if not outer_line:
                continue
            xs, ys = [p[0] for p in outer_line], [p[1] for p in outer_line]
            bbox = {"X_Min": min(xs), "X_Max": max(xs), "Y_Min": min(ys), "Y_Max": max(ys)}
            contour = [[round(x, 2), round(y, 2)] for x, y in zip(xs, ys)]
            if not cur:
                cur = {"Z": [d], "BBox": bbox, "Contour": contour}
            else:
                if any(abs(bbox[k] - cur["BBox"][k]) > 2.0 for k in bbox):
                    blocks.append(cur)
                    cur = {"Z": [d], "BBox": bbox, "Contour": contour}
                cur["Z"].append(d)
        if cur:
            blocks.append(cur)
        
        final_solid_blocks = [
            {
                "ID": f"Solid_{i + 1}",
                "Z_Range": [min(b["Z"]), max(b["Z"])],
                "Size_XY": [round(b["BBox"]["X_Max"] - b["BBox"]["X_Min"], 2), round(b["BBox"]["Y_Max"] - b["BBox"]["Y_Min"], 2)],
                "Outer_Contour": b["Contour"],
            }
            for i, b in enumerate(blocks)
        ]

        all_holes = [f for f in self.features_aligned if f["Type"] == "Hole"]
        all_pillars = [f for f in self.features_aligned if f["Type"] == "Pillar"]
        h_groups = self.cluster_features(all_holes, tolerance=0.5)
        p_groups = self.cluster_features(all_pillars, tolerance=0.5)

        def format_steps(groups):
            final_features = []
            for _, v in groups.items():
                axis = v[0]["Axis"]
                c3d_ref = v[0]["Center3D_Aligned"]
                cx, cy = (
                    (c3d_ref[0], c3d_ref[1])
                    if axis == "Z" else ((c3d_ref[1], c3d_ref[2]) if axis == "X" else (c3d_ref[0], c3d_ref[2]))
                )
                center_key = "Center_XY" if axis == "Z" else ("Center_YZ" if axis == "X" else "Center_XZ")
                depth_idx = {"Z": 2, "X": 0, "Y": 1}[axis]
                steps = sorted(v, key=lambda x: x["Center3D_Aligned"][depth_idx])
                compact = []
                for s in steps:
                    d_val = s["Center3D_Aligned"][depth_idx]
                    dia = round(s["R"] * 2, 2)
                    if not compact or compact[-1]["Diameter"] != dia:
                        compact.append({"Start": d_val, "End": d_val, "Diameter": dia, "_r": s["R"], "_c3d": s["Center3D_Aligned"]})
                    else:
                        compact[-1]["End"] = d_val
                main_step = max(compact, key=lambda x: x["End"] - x["Start"]) if compact else {"Diameter": 0}
                for c in compact:
                    ex_min, ex_max = self.refine_depth_by_cross_views(axis, c["_c3d"][0], c["_c3d"][1], c["_c3d"][2], c["_r"], c["Start"], c["End"])
                    if axis == "Z":
                        c["Z_Start"] = ex_min
                        c["Z_End"] = ex_max
                    elif axis == "X":
                        c["X_Start"] = ex_min
                        c["X_End"] = ex_max
                    else:
                        c["Y_Start"] = ex_min
                        c["Y_End"] = ex_max
                    del c["Start"], c["End"], c["_r"], c["_c3d"]
                final_features.append({"Axis": axis, center_key: [cx, cy], "Main_Diameter": main_step["Diameter"], "Steps": compact})
            return final_features

        final_data = {
            "Part_Overview": {"Bounding_Box_LWH": [round(max(self.all_coords[i]) - min(self.all_coords[i]), 2) if self.all_coords[i] else 0.0 for i in "xyz"]},
            "Solid_Base_Layers": final_solid_blocks,
            "Positive_Pillars": format_steps(p_groups),
            "Negative_Holes": format_steps(h_groups),
        }
        with open("Full_Features_v33.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print("[+] Optimized JSON generated.")

    # 重写：3D+三视图渲染，隔离当前轴，但保留该轴的内部特征（孔、柱）
    def render_3d_and_views(self):
        print("[*] Rendering 3D and orthographic views (Isolated axes, full features) ...")
        
        # 使用完整的当前轴线条（包含内外），而不是只取外轮廓
        lines_z = self.lines_3d["Z"]
        lines_x = self.lines_3d["X"]
        lines_y = self.lines_3d["Y"]
        all_lines = lines_z + lines_x + lines_y
        
        if not all_lines:
            print("[!] No contour data to render")
            return

        plt.ioff()
        fig = plt.figure(figsize=(14, 10))
        
        # 1. 3D视图 (所有轴的所有线条，整体骨架)
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        for line in all_lines:
            pts = np.array(line)
            if len(pts) > 0:
                ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="dodgerblue", alpha=0.5, linewidth=0.8)
        ax_3d.set_title("3D View (All Axes, Full Features)", fontsize=12)
        ax_3d.set_xlabel("X", fontsize=10)
        ax_3d.set_ylabel("Y", fontsize=10)
        ax_3d.set_zlabel("Z", fontsize=10)
        ax_3d.set_aspect('equalxy')
        ax_3d.grid(False)
        ax_3d.set_axisbelow(False)

        # 2. 俯视图 (XY) -> 只看 Z 轴的所有切片（包含外边缘和内部孔柱）
        ax_top = fig.add_subplot(2, 2, 2)
        for line in lines_z:
            pts = np.array(line)
            if len(pts) > 0:
                ax_top.plot(pts[:, 0], pts[:, 1], color="dodgerblue", alpha=0.8, linewidth=1.0)
        ax_top.set_title("Top View (XY) - Z Axis Only (Full Features)", fontsize=12)
        ax_top.set_xlabel("X", fontsize=10)
        ax_top.set_ylabel("Y", fontsize=10)
        ax_top.set_aspect('equal', adjustable='box')
        ax_top.grid(False)

        # 3. 主视图 (XZ) -> 只看 Y 轴的所有切片
        ax_front = fig.add_subplot(2, 2, 3)
        for line in lines_y:
            pts = np.array(line)
            if len(pts) > 0:
                ax_front.plot(pts[:, 0], pts[:, 2], color="dodgerblue", alpha=0.8, linewidth=1.0)
        ax_front.set_title("Front View (XZ) - Y Axis Only (Full Features)", fontsize=12)
        ax_front.set_xlabel("X", fontsize=10)
        ax_front.set_ylabel("Z", fontsize=10)
        ax_front.set_aspect('equal', adjustable='box')
        ax_front.grid(False)

        # 4. 左视图 (YZ) -> 只看 X 轴的所有切片
        ax_left = fig.add_subplot(2, 2, 4)
        for line in lines_x:
            pts = np.array(line)
            if len(pts) > 0:
                ax_left.plot(pts[:, 1], pts[:, 2], color="dodgerblue", alpha=0.8, linewidth=1.0)
        ax_left.set_title("Left View (YZ) - X Axis Only (Full Features)", fontsize=12)
        ax_left.set_xlabel("Y", fontsize=10)
        ax_left.set_ylabel("Z", fontsize=10)
        ax_left.set_aspect('equal', adjustable='box')
        ax_left.grid(False)

        plt.tight_layout()
        plt.savefig("3D_And_Views_FullFeatures.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("[+] 3D and isolated full feature views rendered.")

    # 重写：深度渐变视图导出
    def export_depth_mapped_views(self):
        print("[*] Exporting depth mapped views (Isolated axes, full features) ...")
        # 直接使用包含内外特征的当前轴数据
        lines_z = self.lines_3d["Z"]
        lines_x = self.lines_3d["X"]
        lines_y = self.lines_3d["Y"]

        def save_depth_view(lines_data, x_idx, y_idx, depth_idx, filename, title):
            if not lines_data:
                return
            fig, ax = plt.subplots(figsize=(10, 8))
            segments = []
            depth_values = []
            
            for line in lines_data:
                pts = np.array(line)
                if len(pts) > 0:
                    segments.append(pts[:, [x_idx, y_idx]])
                    depth_val = pts[0, depth_idx]
                    depth_values.append(depth_val)
            
            if not segments:
                return

            # 深度渐变配置
            norm = plt.Normalize(min(depth_values), max(depth_values))
            cmap = plt.get_cmap('viridis')
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.2, alpha=0.9)
            lc.set_array(np.array(depth_values))
            
            ax.add_collection(lc)
            ax.autoscale()
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(["X", "Y", "Z"][x_idx], fontsize=10)
            ax.set_ylabel(["X", "Y", "Z"][y_idx], fontsize=10)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(False)
            plt.colorbar(lc, ax=ax, label='Depth Value')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

        # 导出三视图时，只传入对应的单轴全量数据
        save_depth_view(lines_z, 0, 1, 2, "View_Z_Top_FullFeatures.png", "Top View (XY) - Z Axis Only, Depth Gradient")
        save_depth_view(lines_y, 0, 2, 1, "View_X_Front_FullFeatures.png", "Front View (XZ) - Y Axis Only, Depth Gradient")
        save_depth_view(lines_x, 1, 2, 0, "View_Y_Left_FullFeatures.png", "Left View (YZ) - X Axis Only, Depth Gradient")
        
        print("[+] Depth mapped isolated full contour views exported.")


# 执行主流程
if __name__ == "__main__":
    engine = ModelExtractorV33()
    engine.parse_all()
    engine.align_coordinates()
    engine.export_json()
    engine.export_depth_mapped_views()
    engine.render_3d_and_views()