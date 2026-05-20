import json
import trimesh
import numpy as np
import math
import os

def align_mesh_to_origin(mesh):
    """将 3D 模型的最小边界严格对齐到绝对正数坐标系 (0,0,0)"""
    bounds = mesh.bounds
    translation_vector = [ -bounds[ 0 ][ 0 ], -bounds[ 0 ][ 1 ], -bounds[ 0 ][ 2 ] ]
    mesh.apply_translation(translation_vector)
    return translation_vector

def refine_feature_high_res(original_mesh, features, feature_type="Pillar", margin=1.0, high_res=0.1):
    refined_features = [ ]
    
    for index, feat in enumerate(features):
        axis = feat.get("Axis")
        main_dia = feat.get("Main_Diameter", 0)
        
        shape_type = feat.get("Shape", "Circle") 
        shape_params = feat.get("Shape_Params", { }) 
        
        r = main_dia / 2.0
        
        params_str = f" | 参数: {shape_params}" if shape_params else ""
        print(f"\n  -> 正在处理第 {index + 1} 个{feature_type} | 形状: {shape_type} | 方向: {axis}轴 | 初步等效直径: {main_dia}{params_str}")
       
        # 1. 解析 JSON 数据方向
        if axis == 'Y' and "Center_XZ" in feat:
            cx, cz = feat[ "Center_XZ" ]
            starts = [ step[ "Y_Start" ] for step in feat[ "Steps" ] ]
            ends = [ step[ "Y_End" ] for step in feat[ "Steps" ] ]
            depth_min, depth_max = min(starts), max(ends)
            plane_normal = [ 0, 1, 0 ]
        elif axis == 'X' and "Center_YZ" in feat:
            cy, cz = feat[ "Center_YZ" ]
            starts = [ step[ "X_Start" ] for step in feat[ "Steps" ] ]
            ends = [ step[ "X_End" ] for step in feat[ "Steps" ] ]
            depth_min, depth_max = min(starts), max(ends)
            plane_normal = [ 1, 0, 0 ]
        elif axis == 'Z' and "Center_XY" in feat:
            cx, cy = feat[ "Center_XY" ]
            starts = [ step[ "Z_Start" ] for step in feat[ "Steps" ] ]
            ends = [ step[ "Z_End" ] for step in feat[ "Steps" ] ]
            depth_min, depth_max = min(starts), max(ends)
            plane_normal = [ 0, 0, 1 ]
        else:
            print(f"    [-] 无法识别的{feature_type}结构，保留原数据。")
            refined_features.append(feat)
            continue

        # 2. 设定冗余扫描区间
        scan_margin = max(margin, 2.0)
        slice_levels = np.arange(depth_min - scan_margin, depth_max + scan_margin, high_res)
        print(f"    [*] 生成 {len(slice_levels)} 张高精度截面进行 CT 扫描...")
        
        slices = original_mesh.section_multiplane(
            plane_origin=[ 0, 0, 0 ], 
            plane_normal=plane_normal, 
            heights=slice_levels
        )

        valid_depths = [ ]
        exact_diameters = [ ]
        expected_area = math.pi * (r ** 2)

        for i, slice_2d in enumerate(slices):
            if slice_2d is None: continue
            
            current_level = slice_levels[ i ]
            if axis == 'Z': expected_c3d = np.array([ cx, cy, current_level ])
            elif axis == 'X': expected_c3d = np.array([ current_level, cy, cz ])
            elif axis == 'Y': expected_c3d = np.array([ cx, current_level, cz ])

            best_poly = None
            min_area_diff = float('inf') 
            to_3d_mat = slice_2d.metadata.get('to_3D')
            if to_3d_mat is None: continue

            for poly in slice_2d.polygons_closed:
                poly_r = math.sqrt(poly.area / math.pi)
                if abs(poly_r - r) <= 3.0: 
                    
                    c2d = poly.centroid.coords[ 0 ]
                    c3d = trimesh.transformations.transform_points([ [ c2d[ 0 ], c2d[ 1 ], 0.0 ] ], to_3d_mat)[ 0 ]
                    dist = np.linalg.norm(c3d - expected_c3d)
                    
                    dist_tolerance = (r + margin + 1.5) if feature_type == "Pillar" else (r + margin + 1.0)
                    
                    if dist < dist_tolerance:
                        area_diff = abs(poly.area - expected_area)
                        if area_diff < min_area_diff:
                            min_area_diff = area_diff
                            best_poly = poly

            if best_poly is not None:
                valid_depths.append(current_level)
                exact_dia = 2 * math.sqrt(best_poly.area / math.pi)
                exact_diameters.append(exact_dia)

        if valid_depths:
            exact_start = round(min(valid_depths), 2)
            exact_end = round(max(valid_depths), 2)
            exact_main_dia = round(np.mean(exact_diameters), 2)
            
            print(f"    [√] {feature_type} 修正成功: 深度区间 [{exact_start}, {exact_end}], 精确等效直径 {exact_main_dia}")
            
            feat[ "Main_Diameter" ] = exact_main_dia
            feat[ "Shape_Params" ] = shape_params
            
            step_data = { "Diameter": exact_main_dia, "Shape": shape_type, "Shape_Params": shape_params }
            if axis == 'Y':
                step_data.update({ "Y_Start": exact_start, "Y_End": exact_end })
            elif axis == 'X':
                step_data.update({ "X_Start": exact_start, "X_End": exact_end })
            elif axis == 'Z':
                step_data.update({ "Z_Start": exact_start, "Z_End": exact_end })
            
            feat[ "Steps" ] = [ step_data ]
                
        else:
            print(f"    [!] 高精度切片未捕捉到有效 {feature_type} 结构，保留原数据。")
            
        refined_features.append(feat)

    return refined_features


if __name__ == "__main__":
    stl_file = "current_task.stl"
    input_json_file = "Full_Features_v33.json"   
    output_json_file = "Full_Features_v34.json"  
    
    print(f"\n{'=' * 50}")
    print("[*] 启动高精度空间特征修正引擎 (精简输出版)")
    print(f"{'=' * 50}")

    if os.path.exists(input_json_file) and os.path.exists(stl_file):
        try:
            mesh = trimesh.load(stl_file)
            offset = align_mesh_to_origin(mesh)
            print(f"[*] 3D 模型加载完毕。全局偏移量: {offset}")

            with open(input_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 修正孔洞
            if "Negative_Holes" in data and data[ "Negative_Holes" ]:
                print(f"\n[>>>] 开始处理 {len(data[ 'Negative_Holes' ])} 个孔洞特征...")
                data[ "Negative_Holes" ] = refine_feature_high_res(mesh, data[ "Negative_Holes" ], feature_type="Hole", margin=1.0, high_res=0.1)

            # 修正柱体
            if "Positive_Pillars" in data and data[ "Positive_Pillars" ]:
                print(f"\n[>>>] 开始处理 {len(data[ 'Positive_Pillars' ])} 个柱体特征...")
                data[ "Positive_Pillars" ] = refine_feature_high_res(mesh, data[ "Positive_Pillars" ], feature_type="Pillar", margin=1.5, high_res=0.1)

            # ==========================================
            # 核心修改：只打包这两项输出
            # ==========================================
            output_dict = {
                "Negative_Holes": data.get("Negative_Holes", [ ]),
                "Positive_Pillars": data.get("Positive_Pillars", [ ])
            }

            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(output_dict, f, indent=4, ensure_ascii=False)
            
            print(f"\n[+] 修正完毕！精简数据已保存至: {output_json_file}")
            print(f"{'=' * 50}\n")

        except Exception as e:
            print(f"\n[!] 引擎运行报错: {e}")
    else:
        print(f"[!] 错误: 找不到文件 {stl_file} 或 {input_json_file}")