import json
import trimesh
import numpy as np
import math
import os

def refine_holes_high_res(mesh_path, rough_holes, margin=1.0, high_res=0.01):
    print(f"\n{'='*40}")
    print(f"[*] 启动高精度修正引擎 (时空对齐版)，加载模型: {mesh_path}...")
    try:
        original_mesh = trimesh.load(mesh_path)
        
        # --- 彻底杜绝负数：将原始 STL 的最小边界对齐到 0,0,0 ---
        bounds = original_mesh.bounds
        translation_vector = [-bounds[0][0], -bounds[0][1], -bounds[0][2]]
        original_mesh.apply_translation(translation_vector)
        print(f"[*] 已将原始 STL 基准点对齐至 0,0,0。偏移量: {translation_vector}")
        
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        return rough_holes

    refined_holes = []

    for index, hole in enumerate(rough_holes):
        axis = hole.get("Axis")
        main_dia = hole.get("Main_Diameter", 0)
        r = main_dia / 2.0
        
        print(f"\n  -> 正在处理第 {index+1} 个孔 | 方向: {axis}轴 | 初步直径: {main_dia}")

        if axis == 'Y' and "Center_XZ" in hole:
            cx, cz = hole["Center_XZ"]
            starts = [step["Y_Start"] for step in hole["Steps"]]
            ends = [step["Y_End"] for step in hole["Steps"]]
            depth_min, depth_max = min(starts), max(ends)
            plane_normal = [0, 1, 0]
        elif axis == 'X' and "Center_YZ" in hole:
            cy, cz = hole["Center_YZ"]
            starts = [step["X_Start"] for step in hole["Steps"]]
            ends = [step["X_End"] for step in hole["Steps"]]
            depth_min, depth_max = min(starts), max(ends)
            plane_normal = [1, 0, 0]
        elif axis == 'Z' and "Center_XY" in hole:
            cx, cy = hole["Center_XY"]
            starts = [step["Z_Start"] for step in hole["Steps"]]
            ends = [step["Z_End"] for step in hole["Steps"]]
            depth_min, depth_max = min(starts), max(ends)
            plane_normal = [0, 0, 1]
        else:
            print("    [-] 无法识别的孔洞结构，保留原数据。")
            refined_holes.append(hole)
            continue

        # 因为坐标系现在是 100% 对齐的，我们只需要轻微的 margin 冗余即可
        scan_margin = max(margin, 2.0)
        slice_levels = np.arange(depth_min - scan_margin, depth_max + scan_margin, high_res)
        print(f"    [*] 生成 {len(slice_levels)} 张高精度截面进行 CT 扫描...")
        
        slices = original_mesh.section_multiplane(
            plane_origin=[0, 0, 0], 
            plane_normal=plane_normal, 
            heights=slice_levels
        )

        valid_depths = []
        exact_diameters = []
        expected_area = math.pi * (r ** 2)

        for i, slice_2d in enumerate(slices):
            if slice_2d is None: continue
            
            current_level = slice_levels[i]
            
            if axis == 'Z': expected_c3d = np.array([cx, cy, current_level])
            elif axis == 'X': expected_c3d = np.array([current_level, cy, cz])
            elif axis == 'Y': expected_c3d = np.array([cx, current_level, cz])

            best_poly = None
            min_dist = float('inf')
            to_3d_mat = slice_2d.metadata.get('to_3D')
            if to_3d_mat is None: continue

            for poly in slice_2d.polygons_closed:
                if expected_area * 0.1 < poly.area < expected_area * 10.0:
                    c2d = poly.centroid.coords[0]
                    c3d = trimesh.transformations.transform_points([[c2d[0], c2d[1], 0.0]], to_3d_mat)[0]
                    dist = np.linalg.norm(c3d - expected_c3d)
                    
                    if dist < r + margin + 1.0 and dist < min_dist:
                        min_dist = dist
                        best_poly = poly

            if best_poly is not None:
                valid_depths.append(current_level)
                exact_dia = 2 * math.sqrt(best_poly.area / math.pi)
                exact_diameters.append(exact_dia)

        if valid_depths:
            exact_start = round(min(valid_depths), 2)
            exact_end = round(max(valid_depths), 2)
            exact_main_dia = round(np.mean(exact_diameters), 2)
            
            print(f"    [√] 修正成功: 深度区间 [{exact_start}, {exact_end}], 精确直径 {exact_main_dia}")
            
            hole["Main_Diameter"] = exact_main_dia
            if axis == 'Y':
                hole["Steps"] = [{"Diameter": exact_main_dia, "Y_Start": exact_start, "Y_End": exact_end}]
            elif axis == 'X':
                hole["Steps"] = [{"Diameter": exact_main_dia, "X_Start": exact_start, "X_End": exact_end}]
            elif axis == 'Z':
                hole["Steps"] = [{"Diameter": exact_main_dia, "Z_Start": exact_start, "Z_End": exact_end}]
                
        else:
            print("    [!] 高精度切片未捕捉到有效孔结构，保留原数据。")
            
        refined_holes.append(hole)

    print(f"{'='*40}\n")
    return refined_holes

if __name__ == "__main__":
    stl_file = "current_task.stl"
    input_json_file = "Full_Features_v33.json"   
    output_json_file = "Full_Features_v34.json"  
    
    if os.path.exists(input_json_file) and os.path.exists(stl_file):
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "Negative_Holes" in data and data["Negative_Holes"]:
            print(f"[*] 检测到 {len(data['Negative_Holes'])} 个孔洞特征，准备进行高精度修正...")
            refined_holes = refine_holes_high_res(stl_file, data["Negative_Holes"], margin=1.0, high_res=0.01)
            data["Negative_Holes"] = refined_holes
            
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[+] 孔洞修正完毕！已保存至 {output_json_file}")
        else:
            print("[-] JSON 中未检测到孔洞，跳过修正。")
    else:
        print(f"[!] 找不到文件: {stl_file} 或 {input_json_file}")