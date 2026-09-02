import re
import math
import numpy as np
import os
import cv2  # 引入强大的 OpenCV 用于顶点和外接矩形分析
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from feature_recognition import deduplicate_axis_features, enrich_feature_data

class ShapeFeatureAnalyzer:
    """
    【工业视觉增强版】独立的形状特征与孔/柱分析类
    结合 OpenCV 的多边形逼近与最小外接矩形，精准提取 Length, Width, Angle 等工业参数
    """
    def __init__(self):
        pass

    def poly_area(self, pts):
        if len(pts) < 3: return 0.0
        area = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def poly_perimeter(self, pts):
        if len(pts) < 3: return 0.0
        p = 0.0
        for i in range(len(pts)):
            p += math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        return p

    def get_centroid_and_dia(self, pts):
        if not pts: return [0, 0], 0
        arr = np.array(pts)
        centroid = np.mean(arr, axis=0)
        dists = np.linalg.norm(arr - centroid, axis=1)
        avg_r = np.max(dists)
        return centroid.tolist(), round(avg_r * 2, 2)

    def is_inside(self, point, poly):
        x, y = point
        inside = False
        n = len(poly)
        if n < 3: return False
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

    def to_3d(self, axis, sx, sy, depth):
        if axis == "Z": return [sx, -sy, depth]
        if axis == "X": return [depth, sx, -sy]
        return [sx, depth, -sy]

    def classify_shape(self, pts):
        area = self.poly_area(pts)
        perim = self.poly_perimeter(pts)
        if area < 1e-3 or perim == 0: return None
            
        circularity = (4 * math.pi * area) / (perim * perim)
        centroid, dia = self.get_centroid_and_dia(pts)
        
        contour = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
        
        rect = cv2.minAreaRect(contour)
        (cx_r, cy_r), (w, h), angle = rect
        if w == 0 or h == 0: return None
        aspect_ratio = max(w, h) / min(w, h)
        rectangularity = min(1.0, area / max(w * h, 1e-9))
        
        epsilon = 0.025 * perim
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = len(approx)

        detected_shape = "Unknown"
        if corners > 6:
            if aspect_ratio <= 1.15 and circularity > 0.82:
                detected_shape = "Circle"
            elif aspect_ratio > 1.15 and rectangularity < 0.86:
                detected_shape = "Ellipse"
            else:
                detected_shape = "Capsule"
        elif corners == 3: detected_shape = "Triangle"
        elif corners == 4:
            detected_shape = "Square" if aspect_ratio <= 1.08 else "Rectangle"
        elif corners == 6: detected_shape = "Hexagon"
        elif corners == 5: detected_shape = "Pentagon"
        else:
            if aspect_ratio <= 1.15 and circularity > 0.85: detected_shape = "Circle"
            elif aspect_ratio > 1.15 and circularity > 0.7: detected_shape = "Capsule"

        shape_params = {}
        if detected_shape == "Circle":
            shape_params = {"Diameter": round(dia, 2)}
        elif detected_shape == "Ellipse":
            shape_params = {
                "Major_Diameter": round(max(w, h), 2),
                "Minor_Diameter": round(min(w, h), 2),
                "Angle": round(angle, 2),
            }
        elif detected_shape == "Capsule":
            shape_params = {
                "Length": round(max(w, h), 2),
                "Width": round(min(w, h), 2),
                "Angle": round(angle, 2)
            }
        elif detected_shape in ["Rectangle", "Square"]:
            shape_params = {
                "Length": round(max(w, h), 2),
                "Width": round(min(w, h), 2),
                "Angle": round(angle, 2),
            }
        elif detected_shape in ["Triangle", "Pentagon", "Hexagon"]:
            shape_params = {
                "Circumcircle_Diameter": round(dia, 2),
                "Angle": round(angle, 2),
                "Sides": corners
            }

        return {
            "shape_type": detected_shape,
            "centroid": tuple(centroid),
            "dia": dia,
            "area": area,
            "shape_params": shape_params,
            "pts": pts
        }

    def extract_topological_features(self, unique_shapes, axis, depth):
        features = []
        for i, shape_obj in enumerate(unique_shapes):
            if shape_obj["shape_type"] == "Unknown": continue
                
            inside_count = 0
            for j, other_obj in enumerate(unique_shapes):
                if i == j: continue
                if other_obj["area"] > shape_obj["area"] * 1.05:
                    if self.is_inside(shape_obj["centroid"], other_obj["pts"]):
                        inside_count += 1
                        
            c3d = self.to_3d(axis, shape_obj["centroid"][0], shape_obj["centroid"][1], depth)
            r_val = math.sqrt(shape_obj["area"] / math.pi)
            feat_type = "Hole" if inside_count % 2 != 0 else "Pillar"
            
            features.append({
                "Type": feat_type,
                "Shape": shape_obj["shape_type"],
                "Shape_Params": shape_obj["shape_params"], 
                "Axis": axis,
                "Center3D": c3d,
                "R": round(r_val, 2)
            })
        return features


class FeatureExtractor:
    def __init__(self):
        self.raw_features = []
        self.raw_lines = {"Z": [], "X": [], "Y": []}
        self.features_aligned = []
        self.lines_3d = {"Z": [], "X": [], "Y": []}
        self.all_coords = {"x": [], "y": [], "z": []}
        self.shape_analyzer = ShapeFeatureAnalyzer()
        self.slice_metadata = {}
        if os.getenv("PRESERVE_GLOBAL_SLICE_COORDINATES") == "1" and os.path.exists("slice_metadata.json"):
            with open("slice_metadata.json", "r", encoding="utf-8") as metadata_file:
                self.slice_metadata = json.load(metadata_file)

    def extract_coords(self, text):
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.\d+|\d+", text or "")]
        return [(round(nums[i], 2), round(nums[i+1], 2)) for i in range(0, len(nums)-1, 2)]

    def parse_points_attr(self, text):
        pts = []
        if not text: return pts
        for token in text.strip().split():
            if "," not in token: continue
            sx, sy = token.split(",", 1)
            try: pts.append((round(float(sx), 2), round(float(sy), 2)))
            except ValueError: continue
        return pts

    def extract_true_depth(self, layer_tag):
        desc = layer_tag.find("desc")
        if not desc: return 0.0
        match = re.search(r"(?:Length_Dir|H):\s*([-+]?\d*\.?\d+)", str(desc))
        return float(match.group(1)) if match else 0.0

    def generate_circle_pts(self, cx, cy, r, segments=36):
        pts = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            pts.append((round(cx + r * math.cos(angle), 2), round(cy + r * math.sin(angle), 2)))
        return pts

    def generate_ellipse_pts(self, cx, cy, rx, ry, segments=48):
        pts = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            pts.append((round(cx + rx * math.cos(angle), 2), round(cy + ry * math.sin(angle), 2)))
        return pts

    def parse_all(self):
        merged_files = {
            "X": "merged_slices_x.svg",
            "Y": "merged_slices_y.svg",
            "Z": "merged_slices_z.svg",
        }
        for axis in ["Z", "X", "Y"]:
            path = merged_files[axis]
            if not os.path.exists(path): continue
            print(f"[*] Parsing {path} ...")
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "xml")
            for layer in soup.find_all("svg", id=lambda x: x and x.startswith("layer_")):
                depth = self.extract_true_depth(layer)
                parsed_raw_shapes = []
                
                for shape in layer.find_all(["polygon", "rect", "path"]):
                    pts = []
                    if shape.name == "polygon": pts = self.parse_points_attr(shape.get("points", ""))
                    elif shape.name == "rect":
                        x, y = float(shape.get("x", 0.0)), float(shape.get("y", 0.0))
                        w, h = float(shape.get("width", 0.0)), float(shape.get("height", 0.0))
                        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    else: pts = self.extract_coords(shape.get("d", ""))
                    if len(pts) >= 3: parsed_raw_shapes.append({"pts": pts})
                        
                for c in layer.find_all("circle"):
                    cx, cy, r = float(c.get("cx", 0.0)), float(c.get("cy", 0.0)), float(c.get("r", 0.0))
                    if r > 0:
                        pts = self.generate_circle_pts(cx, cy, r)
                        parsed_raw_shapes.append({"pts": pts})
                        
                for e in layer.find_all("ellipse"):
                    cx, cy = float(e.get("cx", 0.0)), float(e.get("cy", 0.0))
                    rx, ry = float(e.get("rx", 0.0)), float(e.get("ry", 0.0))
                    if rx > 0 and ry > 0:
                        pts = self.generate_ellipse_pts(cx, cy, rx, ry)
                        parsed_raw_shapes.append({"pts": pts})
                
                unified_shapes = []
                for shape_data in parsed_raw_shapes:
                    pts = shape_data["pts"]
                    shape_info = self.shape_analyzer.classify_shape(pts)
                    if shape_info is not None:
                        unified_shapes.append(shape_info)
                    self.raw_lines[axis].append([self.shape_analyzer.to_3d(axis, sx, sy, depth) for sx, sy in pts])
                
                unique_shapes = []
                for shape in unified_shapes:
                    if shape.get("area", 0) < 1e-3: continue 
                    is_dup = False
                    for u in unique_shapes:
                        dist = math.hypot(shape["centroid"][0] - u["centroid"][0], shape["centroid"][1] - u["centroid"][1])
                        area_diff = abs(shape["area"] - u["area"])
                        if dist < 0.5 and area_diff < max(shape["area"], u["area"]) * 0.05:
                            is_dup = True
                            break
                    if not is_dup:
                        unique_shapes.append(shape)
                
                layer_features = self.shape_analyzer.extract_topological_features(unique_shapes, axis, depth)
                self.raw_features.extend(layer_features)

    def align_coordinates(self):
        print("[*] Aligning global 3D coordinates (Zero-based layout) ...")
        for axis in ["Z", "X", "Y"]:
            lines = self.raw_lines[axis]
            axis_features = [f for f in self.raw_features if f["Axis"] == axis]
            xs, ys, zs = [], [], []
            for line in lines:
                for px, py, pz in line: xs.append(px); ys.append(py); zs.append(pz)
            for f in axis_features:
                cx, cy, cz = f["Center3D"]
                xs.append(cx); ys.append(cy); zs.append(cz)
            if not xs: continue
            
            off_x, off_y, off_z = min(xs), min(ys), min(zs)
            depth_origins = self.slice_metadata.get("depth_origins", {})
            if axis == "X" and "X" in depth_origins:
                off_x = float(depth_origins["X"])
            elif axis == "Y" and "Y" in depth_origins:
                off_y = float(depth_origins["Y"])
            elif axis == "Z" and "Z" in depth_origins:
                off_z = float(depth_origins["Z"])
            
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

    def filter_volatile_features(self, feature_groups):
        valid_groups = {}
        for k, group in feature_groups.items():
            if len(group) < 2: continue 
            axis = group[0]["Axis"]
            depth_idx = {"Z": 2, "X": 0, "Y": 1}[axis]
            sorted_g = sorted(group, key=lambda x: x["Center3D_Aligned"][depth_idx])
            volatile_count = 0
            for i in range(1, len(sorted_g)):
                r_prev = sorted_g[i-1]["R"]
                r_curr = sorted_g[i]["R"]
                if r_prev > 0:
                    change_rate = abs(r_curr - r_prev) / r_prev
                    if change_rate > 0.10: volatile_count += 1
            if volatile_count > len(sorted_g) * 0.4: continue
            valid_groups[k] = group
        return valid_groups

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
            if not placed: groups.append([f])
        return {f"Group_{i}": group for i, group in enumerate(groups)}

    def get_feature_bbox(self, feat):
        axis = feat["Axis"]
        r = feat["Main_Diameter"] / 2.0
        steps = feat["Steps"]
        if axis == "Z":
            cx, cy = feat["Center_XY"]
            z_min = min(s["Z_Start"] for s in steps); z_max = max(s["Z_End"] for s in steps)
            return [cx - r, cx + r, cy - r, cy + r, z_min, z_max]
        elif axis == "X":
            cy, cz = feat["Center_YZ"]
            x_min = min(s["X_Start"] for s in steps); x_max = max(s["X_End"] for s in steps)
            return [x_min, x_max, cy - r, cy + r, cz - r, cz + r]
        else:
            cx, cz = feat["Center_XZ"]
            y_min = min(s["Y_Start"] for s in steps); y_max = max(s["Y_End"] for s in steps)
            return [cx - r, cx + r, y_min, y_max, cz - r, cz + r]

    def get_intersection_volume(self, boxA, boxB):
        ix_min = max(boxA[0], boxB[0]); ix_max = min(boxA[1], boxB[1])
        iy_min = max(boxA[2], boxB[2]); iy_max = min(boxA[3], boxB[3])
        iz_min = max(boxA[4], boxB[4]); iz_max = min(boxA[5], boxB[5])
        if ix_max > ix_min and iy_max > iy_min and iz_max > iz_min:
            return (ix_max - ix_min) * (iy_max - iy_min) * (iz_max - iz_min)
        return 0.0

    def get_bbox_volume(self, box):
        return max(0, box[1] - box[0]) * max(0, box[3] - box[2]) * max(0, box[5] - box[4])

    def get_shape_priority(self, shape_name):
        priorities = {"Circle": 8, "Capsule": 7, "Hexagon": 6, "Pentagon": 5, "Triangle": 2, "Unknown": 0}
        return priorities.get(shape_name, 0)

    def remove_ghost_features(self, formatted_features):
        # AABB overlap across different slicing axes is often a real physical
        # intersection (for example two orthogonal bores).  Only collapse
        # numerically identical observations from the same axis; cross-axis
        # evidence is preserved and resolved into explicit topology relations
        # by feature_recognition.enrich_feature_data.
        return deduplicate_axis_features(formatted_features)

    def export_json(self):
        print("[*] Exporting Optimized JSON with Shape_Params ...")
        z_layers = defaultdict(list)
        for line in self.lines_3d["Z"]:
            if line: z_layers[round(line[0][2], 2)].append(line)
        sorted_z = sorted(z_layers.keys())
        blocks, cur = [], None
        
        for d in sorted_z:
            lines = z_layers[d]
            outer_line = max(lines, key=lambda l: self.shape_analyzer.poly_area(l)) if lines else []
            if not outer_line: continue
            xs, ys = [p[0] for p in outer_line], [p[1] for p in outer_line]
            bbox = {"X_Min": min(xs), "X_Max": max(xs), "Y_Min": min(ys), "Y_Max": max(ys)}
            contour = [[round(x, 2), round(y, 2)] for x, y in zip(xs, ys)]
            if not cur: cur = {"Z": [d], "BBox": bbox, "Contour": contour}
            else:
                if any(abs(bbox[k] - cur["BBox"][k]) > 2.0 for k in bbox):
                    blocks.append(cur)
                    cur = {"Z": [d], "BBox": bbox, "Contour": contour}
                cur["Z"].append(d)
        if cur: blocks.append(cur)
        
        final_solid_blocks = []
        for i, b in enumerate(blocks):
            layer_id = "Base_Foundation" if i == 0 else f"Solid_Tier_{i + 1}"
            final_solid_blocks.append({
                "ID": layer_id,
                "Z_Range": [min(b["Z"]), max(b["Z"])],
                "Size_XY": [round(b["BBox"]["X_Max"] - b["BBox"]["X_Min"], 2), round(b["BBox"]["Y_Max"] - b["BBox"]["Y_Min"], 2)],
                "Outer_Contour": b["Contour"]
            })

        all_holes = [f for f in self.features_aligned if f["Type"] == "Hole"]
        all_pillars = [f for f in self.features_aligned if f["Type"] == "Pillar"]
        h_groups_raw = self.cluster_features(all_holes, tolerance=0.5)
        p_groups_raw = self.cluster_features(all_pillars, tolerance=0.5)
        h_groups = self.filter_volatile_features(h_groups_raw)
        p_groups = self.filter_volatile_features(p_groups_raw)

        def format_steps(groups):
            final_features = []
            for _, v in groups.items():
                axis = v[0]["Axis"]
                c3d_ref = v[0]["Center3D_Aligned"]
                cx, cy = ((c3d_ref[0], c3d_ref[1]) if axis == "Z" else ((c3d_ref[1], c3d_ref[2]) if axis == "X" else (c3d_ref[0], c3d_ref[2])))
                center_key = "Center_XY" if axis == "Z" else ("Center_YZ" if axis == "X" else "Center_XZ")
                depth_idx = {"Z": 2, "X": 0, "Y": 1}[axis]
                steps = sorted(v, key=lambda x: x["Center3D_Aligned"][depth_idx])
                compact = []
                for s in steps:
                    d_val = s["Center3D_Aligned"][depth_idx]; dia = round(s["R"] * 2, 2)
                    shape_type, shape_params = s.get("Shape", "Circle"), s.get("Shape_Params", {})
                    if not compact or compact[-1]["Diameter"] != dia:
                        compact.append({"Start": d_val, "End": d_val, "Diameter": dia, "Shape": shape_type, "Shape_Params": shape_params, "_c3d": s["Center3D_Aligned"]})
                    else: compact[-1]["End"] = d_val
                main_step = max(compact, key=lambda x: x["End"] - x["Start"]) if compact else {"Diameter": 0, "Shape": "Circle", "Shape_Params": {}}
                for c in compact:
                    if axis == "Z": c["Z_Start"], c["Z_End"] = c["Start"], c["End"]
                    elif axis == "X": c["X_Start"], c["X_End"] = c["Start"], c["End"]
                    else: c["Y_Start"], c["Y_End"] = c["Start"], c["End"]
                    del c["Start"], c["End"], c["_c3d"]
                final_features.append({"Axis": axis, center_key: [round(cx, 2), round(cy, 2)], "Main_Diameter": main_step["Diameter"], "Shape": main_step["Shape"], "Shape_Params": main_step.get("Shape_Params", {}), "Steps": compact})
            return final_features

        raw_pos_features, raw_neg_features = format_steps(p_groups), format_steps(h_groups)
        clean_pos_features, clean_neg_features = self.remove_ghost_features(raw_pos_features), self.remove_ghost_features(raw_neg_features)

        measured_bbox = [round(max(self.all_coords[i]) - min(self.all_coords[i]), 2) if self.all_coords[i] else 0.0 for i in "xyz"]
        metadata_bbox = self.slice_metadata.get("bounding_box_lwh")
        bounding_box = [round(float(value), 2) for value in metadata_bbox] if metadata_bbox else measured_bbox
        axis_spacing = {}
        for axis, depth_idx in {"X": 0, "Y": 1, "Z": 2}.items():
            depths = sorted({round(line[0][depth_idx], 6) for line in self.lines_3d[axis] if line})
            gaps = [right - left for left, right in zip(depths, depths[1:]) if right - left > 1e-9]
            if gaps:
                axis_spacing[axis] = round(float(np.median(gaps)), 6)
        final_data = {
            "Part_Overview": {"Bounding_Box_LWH": bounding_box},
            "Slice_Metadata": {"Axis_Layer_Spacing": axis_spacing},
            "Solid_Base_Layers": final_solid_blocks,
            "Positive_Pillars": clean_pos_features,
            "Negative_Holes": clean_neg_features,
        }
        final_data = enrich_feature_data(final_data)
        with open("features_raw.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        print("[+] Optimized JSON generated.")

    def render_3d_and_views(self):
        """三维及线框图正交视图渲染（带轴向标注与比例平衡）"""
        print("[*] Rendering 3D and orthographic views with Axis Labels ...")
        lines_z, lines_x, lines_y = self.lines_3d["Z"], self.lines_3d["X"], self.lines_3d["Y"]
        all_lines = lines_z + lines_x + lines_y
        if not all_lines: return

        plt.ioff()
        fig = plt.figure(figsize=(14, 10))
        
        # 1. 3D 线框图
        ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
        for line in all_lines:
            pts = np.array(line)
            if len(pts) > 0: ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="dodgerblue", alpha=0.5, linewidth=0.8)
        ax_3d.set_xlabel('X Axis'); ax_3d.set_ylabel('Y Axis'); ax_3d.set_zlabel('Z Axis')
        ax_3d.set_title('3D Wireframe Reconstruction')

        # 2. 顶视图 (Z轴投影 -> 显示 X-Y 平面)
        ax_top = fig.add_subplot(2, 2, 2)
        for line in lines_z:
            pts = np.array(line)
            if len(pts) > 0: ax_top.plot(pts[:, 0], pts[:, 1], color="dodgerblue", alpha=0.8, linewidth=1.0)
        ax_top.set_xlabel('X Axis'); ax_top.set_ylabel('Y Axis'); ax_top.set_title('Top View (Z-Projection)')
        ax_top.set_aspect('equal', adjustable='datalim'); ax_top.grid(True, linestyle='--', alpha=0.3)
            
        # 3. 正视图 (Y轴投影 -> 显示 X-Z 平面)
        ax_front = fig.add_subplot(2, 2, 3)
        for line in lines_y:
            pts = np.array(line)
            if len(pts) > 0: ax_front.plot(pts[:, 0], pts[:, 2], color="dodgerblue", alpha=0.8, linewidth=1.0)
        ax_front.set_xlabel('X Axis'); ax_front.set_ylabel('Z Axis'); ax_front.set_title('Front View (Y-Projection)')
        ax_front.set_aspect('equal', adjustable='datalim'); ax_front.grid(True, linestyle='--', alpha=0.3)

        # 4. 左视图 (X轴投影 -> 显示 Y-Z 平面)
        ax_left = fig.add_subplot(2, 2, 4)
        for line in lines_x:
            pts = np.array(line)
            if len(pts) > 0: ax_left.plot(pts[:, 1], pts[:, 2], color="dodgerblue", alpha=0.8, linewidth=1.0)
        ax_left.set_xlabel('Y Axis'); ax_left.set_ylabel('Z Axis'); ax_left.set_title('Side View (X-Projection)')
        ax_left.set_aspect('equal', adjustable='datalim'); ax_left.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig("feature_overview.png", dpi=300)
        plt.close(fig)

    def export_depth_mapped_views(self):
        """带深度映射的实心渲染图（带自动挖孔逻辑与轴向标注）"""
        print("[*] Exporting depth mapped views (Solid Fill Mode) ...")
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        def save_depth_view(lines_data, x_idx, y_idx, depth_idx, filename, title, xlabel, ylabel):
            if not lines_data: return
            fig, ax = plt.subplots(figsize=(10, 8))
            
            depth_groups = defaultdict(list)
            for line in lines_data:
                pts = np.array(line)
                if len(pts) > 2:
                    depth = pts[0, depth_idx]
                    depth_groups[round(depth, 3)].append(pts[:, [x_idx, y_idx]].tolist())
            if not depth_groups: return
            
            sorted_depths = sorted(depth_groups.keys())
            norm = plt.Normalize(min(sorted_depths), max(sorted_depths))
            cmap = plt.get_cmap('viridis')

            def signed_area(pts):
                area = 0.0; n = len(pts)
                for i in range(n):
                    j = (i + 1) % n
                    area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
                return area / 2.0

            for depth in sorted_depths:
                polys = depth_groups[depth]; color = cmap(norm(depth))
                polys_info = []
                for poly in polys:
                    s_area = signed_area(poly)
                    polys_info.append({"pts": poly, "area": abs(s_area), "s_area": s_area})
                polys_info.sort(key=lambda item: -item["area"])
                
                for i, p in enumerate(polys_info):
                    level = 0; pt = p["pts"][0]
                    for j in range(i):
                        if self.shape_analyzer.is_inside(pt, polys_info[j]["pts"]): level += 1
                    p["level"] = level
                    is_ccw = p["s_area"] > 0; needs_ccw = (level % 2 == 0)
                    if is_ccw != needs_ccw: p["pts"] = p["pts"][::-1]
                        
                vertices, codes = [], []
                for p in polys_info:
                    poly_pts = p["pts"]
                    vertices.extend(poly_pts); vertices.append(poly_pts[0])
                    codes.extend([Path.MOVETO] + [Path.LINETO] * (len(poly_pts) - 1) + [Path.CLOSEPOLY])
                if vertices:
                    path = Path(vertices, codes)
                    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.95)
                    ax.add_patch(patch)
            
            all_pts = np.vstack([p for line in lines_data for p in line])
            ax.set_xlim(np.min(all_pts[:, x_idx]), np.max(all_pts[:, x_idx]))
            ax.set_ylim(np.min(all_pts[:, y_idx]), np.max(all_pts[:, y_idx]))
            ax.set_aspect('equal')
            
            # 设置标签与标题
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, pad=15)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=f'Depth (Axis {["X", "Y", "Z"][depth_idx]})')
            
            plt.savefig(filename, dpi=300); plt.close(fig)

        # 修正：准确映射对应的坐标轴名称
        save_depth_view(self.lines_3d["Z"], 0, 1, 2, "depth_view_z.png", "Top View Depth Mapping", "X Axis", "Y Axis")
        save_depth_view(self.lines_3d["Y"], 0, 2, 1, "depth_view_y.png", "Front View Depth Mapping", "X Axis", "Z Axis")
        save_depth_view(self.lines_3d["X"], 1, 2, 0, "depth_view_x.png", "Side View Depth Mapping", "Y Axis", "Z Axis")

if __name__ == "__main__":
    engine = FeatureExtractor()
    engine.parse_all()
    engine.align_coordinates()
    engine.export_json()
    engine.export_depth_mapped_views()
    engine.render_3d_and_views()
