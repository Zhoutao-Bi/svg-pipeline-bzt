import re
import os
import json
import math
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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

                for c in layer.find_all("circle"):
                    cx, cy, r = float(c.get("cx", 0.0)), float(c.get("cy", 0.0)), float(c.get("r", 0.0))
                    if r > 0:
                        pts = self.generate_circle_pts(cx, cy, r)
                        unified_shapes.append({"type": "circle", "pts": pts, "cx": cx, "cy": cy, "r": r})

                for e in layer.find_all("ellipse"):
                    cx, cy = float(e.get("cx", 0.0)), float(e.get("cy", 0.0))
                    rx, ry = float(e.get("rx", 0.0)), float(e.get("ry", 0.0))
                    r = (rx + ry) / 2.0
                    if r > 0:
                        pts = self.generate_circle_pts(cx, cy, r)
                        unified_shapes.append({"type": "ellipse", "pts": pts, "cx": cx, "cy": cy, "r": round(r, 2)})

                # 第一阶段：保存渲染线条，并计算关键指标 (形心与圆度)
                for shape_obj in unified_shapes:
                    pts = shape_obj["pts"]
                    self.raw_lines[axis].append([self.to_3d(axis, sx, sy, depth) for sx, sy in pts])
                    
                    if shape_obj["type"] in ["circle", "ellipse"]:
                        shape_obj["centroid"] = (shape_obj["cx"], shape_obj["cy"])
                        shape_obj["dia"] = shape_obj["r"] * 2
                        shape_obj["circularity"] = 1.0  # 完美圆形
                    else:
                        c_xy, dia = self.get_centroid_and_dia(pts)
                        shape_obj["centroid"] = tuple(c_xy)
                        shape_obj["dia"] = dia
                        area = self.poly_area(pts)
                        perim = self.poly_perimeter(pts)
                        # 计算圆度
                        shape_obj["circularity"] = (4 * math.pi * area) / (perim * perim) if perim > 0 else 0.0

                # 第二阶段：形心奇偶校验与圆度过滤
                for i, shape_obj in enumerate(unified_shapes):
                    # 斩杀幽灵孔：任何圆度低于 0.75 的边缘多边形 (倒角、槽口等) 都不可能成为特征孔
                    if shape_obj["circularity"] < 0.75:
                        continue
                        
                    inside_count = 0
                    for j, other_obj in enumerate(unified_shapes):
                        if i == j: continue
                        if self.is_inside(shape_obj["centroid"], other_obj["pts"]):
                            inside_count += 1
                            
                    if inside_count % 2 != 0:
                        self.raw_features.append({"Type": "Hole", "Axis": axis, "Center3D": self.to_3d(axis, shape_obj["centroid"][0], shape_obj["centroid"][1], depth), "R": round(shape_obj["dia"] / 2.0, 2)})
                    else:
                        if shape_obj["type"] in ["circle", "ellipse"]:
                            self.raw_features.append({"Type": "Pillar", "Axis": axis, "Center3D": self.to_3d(axis, shape_obj["centroid"][0], shape_obj["centroid"][1], depth), "R": round(shape_obj["dia"] / 2.0, 2)})

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
                else:
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

        # 空间容差聚类 (修复了之前的特征断裂现象)
        def cluster_features(features, tolerance=0.5):
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

        all_holes = [f for f in self.features_aligned if f["Type"] == "Hole"]
        all_pillars = [f for f in self.features_aligned if f["Type"] == "Pillar"]

        h_groups = cluster_features(all_holes, tolerance=0.5)
        p_groups = cluster_features(all_pillars, tolerance=0.5)

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

    def render_3d_and_views(self):
        print("[*] Rendering 3D and orthographic views ...")
        all_lines = []
        for axis in ["X", "Y", "Z"]:
            all_lines.extend(self.lines_3d[axis])
        if not all_lines:
            return
        plt.ioff()
        fig = plt.figure(figsize=(14, 10))
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        for line in all_lines:
            pts = np.array(line)
            ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="teal", alpha=0.6, linewidth=0.8)
        ax_top = fig.add_subplot(2, 2, 2)
        for line in all_lines:
            pts = np.array(line)
            ax_top.plot(pts[:, 0], pts[:, 1], color="coral", alpha=0.5, linewidth=0.8)
        ax_front = fig.add_subplot(2, 2, 3)
        for line in all_lines:
            pts = np.array(line)
            ax_front.plot(pts[:, 0], pts[:, 2], color="dodgerblue", alpha=0.5, linewidth=0.8)
        ax_left = fig.add_subplot(2, 2, 4)
        for line in all_lines:
            pts = np.array(line)
            ax_left.plot(pts[:, 1], pts[:, 2], color="mediumseagreen", alpha=0.5, linewidth=0.8)
        plt.tight_layout()
        plt.savefig("Full_Isometric_View.png", dpi=150)
        plt.close(fig)

    def export_depth_mapped_views(self):
        import matplotlib.cm as cm
        from matplotlib.collections import LineCollection

        print("[*] Exporting depth mapped views ...")
        all_lines = []
        for axis in ["X", "Y", "Z"]:
            all_lines.extend(self.lines_3d[axis])
        if not all_lines:
            return

        def save_depth_view(x_idx, y_idx, depth_idx, filename):
            plt.ioff()
            fig, ax = plt.subplots(figsize=(10, 8))
            depths = [pt[depth_idx] for line in all_lines for pt in line]
            min_d, max_d = min(depths), max(depths)
            cmap, norm = cm.viridis, plt.Normalize(vmin=min_d, vmax=max_d)
            segments, colors = [], []
            for line in all_lines:
                pts = np.array(line)
                segments.append(pts[:, [x_idx, y_idx]])
                colors.append(cmap(norm(np.mean(pts[:, depth_idx]))))
            lc = LineCollection(segments, colors=colors, linewidths=1.0, alpha=0.8)
            ax.add_collection(lc)
            all_x = [pt[x_idx] for line in all_lines for pt in line]
            all_y = [pt[y_idx] for line in all_lines for pt in line]
            ax.set_xlim(min(all_x) * 1.1, max(all_x) * 1.1)
            ax.set_ylim(min(all_y) * 1.1, max(all_y) * 1.1)
            ax.set_aspect("equal")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

        save_depth_view(0, 1, 2, "View_Z_Depth.png")
        save_depth_view(0, 2, 1, "View_Y_Depth.png")
        save_depth_view(1, 2, 0, "View_X_Depth.png")


if __name__ == "__main__":
    engine = ModelExtractorV33()
    engine.parse_all()
    engine.align_coordinates()
    engine.export_json()
    engine.export_depth_mapped_views()
    engine.render_3d_and_views()
    print("[+] Done.")