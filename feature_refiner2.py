import json
import trimesh
import numpy as np
import math
import os

def align_mesh_to_origin(mesh):
    """将 3D 模型的最小边界严格对齐到绝对正数坐标系 (0,0,0)"""
    bounds = mesh.bounds
    translation_vector = [-bounds[0][0], -bounds[0][1], -bounds[0][2]]
    mesh.apply_translation(translation_vector)
    return translation_vector

def refine_features_globally(mesh, data, high_res=0.01):
    """全局优化版：按轴分组 -> 区间合并跳过空白区 -> 一次性切片"""
    
    # 1. 收集所有特征并注册元数据
    all_features_info = []
    
    for f_type, dict_key, margin in [("Hole", "Negative_Holes", 1.0), ("Pillar", "Positive_Pillars", 1.5)]:
        if dict_key in data and data[dict_key]:
            for index, feat in enumerate(data[dict_key]):
                all_features_info.append({
                    'type': f_type,
                    'dict_key': dict_key,
                    'index': index,
                    'feat': feat,
                    'margin': margin
                })
    
    # 2. 按 Axis 分类，并计算每个特征的局部深度区间
    axis_tasks = {'X': [], 'Y': [], 'Z': []}
    
    for info in all_features_info:
        feat = info['feat']
        axis = feat.get("Axis")
        if axis not in axis_tasks:
            info['valid'] = False
            continue
        
        # 解析中心坐标和深度
        if axis == 'Y' and "Center_XZ" in feat:
            cx, cz = feat["Center_XZ"]
            starts = [step["Y_Start"] for step in feat["Steps"]]
            ends = [step["Y_End"] for step in feat["Steps"]]
            expected_c3d_func = lambda lvl, cx=cx, cz=cz: np.array([cx, lvl, cz])
        elif axis == 'X' and "Center_YZ" in feat:
            cy, cz = feat["Center_YZ"]
            starts = [step["X_Start"] for step in feat["Steps"]]
            ends = [step["X_End"] for step in feat["Steps"]]
            expected_c3d_func = lambda lvl, cy=cy, cz=cz: np.array([lvl, cy, cz])
        elif axis == 'Z' and "Center_XY" in feat:
            cx, cy = feat["Center_XY"]
            starts = [step["Z_Start"] for step in feat["Steps"]]
            ends = [step["Z_End"] for step in feat["Steps"]]
            expected_c3d_func = lambda lvl, cx=cx, cy=cy: np.array([cx, cy, lvl])
        else:
            info['valid'] = False
            continue
            
        depth_min, depth_max = min(starts), max(ends)
        
        # 强制打穿扫描区间 (保留至少 2.0mm 冗余)
        scan_margin = max(info['margin'], 2.0) 
        
        # 记录该特征的独立扫描区间 (已包含冗余)
        info['local_min'] = depth_min - scan_margin
        info['local_max'] = depth_max + scan_margin
        info['expected_c3d_func'] = expected_c3d_func
        info['valid'] = True
        
        axis_tasks[axis].append(info)

    # 3. 按轴进行【区间合并与断点切片】
    axis_slice_data = {}
    
    for axis, infos in axis_tasks.items():
        if not infos:
            continue
            
        # --- 区间合并算法开始 ---
        # 收集所有局部区间并按起点排序
        intervals = [[info['local_min'], info['local_max']] for info in infos]
        intervals.sort(key=lambda x: x[0])
        
        merged_intervals = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一个区间不重叠 (加入 1e-5 容差防止浮点误判)
            if not merged_intervals or merged_intervals[-1][1] < interval[0] - 1e-5:
                merged_intervals.append(interval)
            else:
                # 发生重叠，合并最大边界
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
        # --- 区间合并算法结束 ---

        # 根据合并后的【有效区间】生成切片高度
        slice_levels_list = []
        for mi in merged_intervals:
            slice_levels_list.append(np.arange(mi[0], mi[1] + high_res, high_res))
        
        # 拼接所有有效高度，并使用 np.unique 去重且自动排序
        slice_levels = np.unique(np.concatenate(slice_levels_list))
        
        if axis == 'Y': plane_normal = [0, 1, 0]
        elif axis == 'X': plane_normal = [1, 0, 0]
        elif axis == 'Z': plane_normal = [0, 0, 1]
        
        print(f"\n[*] 正在对 {axis} 轴执行分段 CT 扫描 (已跳过无用空白区)...")
        ranges_str = " | ".join([f"[{mi[0]:.2f}, {mi[1]:.2f}]" for mi in merged_intervals])
        print(f"    -> 锁定 {len(merged_intervals)} 个有效扫描区: {ranges_str}")
        print(f"    -> 实际生成截面: {len(slice_levels)} 张")
        
        slices = mesh.section_multiplane(
            plane_origin=[0, 0, 0], 
            plane_normal=plane_normal, 
            heights=slice_levels
        )
        axis_slice_data[axis] = (slice_levels, slices)

    # 4. 逐个特征认领切片数据并匹配
    for info in all_features_info:
        if not info.get('valid'): continue
            
        feat = info['feat']
        f_type = info['type']
        index = info['index']
        
        shape_type = feat.get("Shape", "Circle") 
        shape_params = feat.get("Shape_Params", {}) 
        main_dia = feat.get("Main_Diameter", 0)
        r = main_dia / 2.0
        axis = feat.get("Axis")
        
        params_str = f" | 参数: {shape_params}" if shape_params else ""
        print(f"\n  -> 正在处理第 {index+1} 个{f_type} | 形状: {shape_type} | 方向: {axis}轴 | 初步等效直径: {main_dia}{params_str}")
        
        slice_levels, slices = axis_slice_data[axis]
        expected_c3d_func = info['expected_c3d_func']
        
        valid_depths = []
        exact_diameters = []
        expected_area = math.pi * (r ** 2)
        
        # 在离散的全局切片中，仅遍历当前特征 [local_min, local_max] 范围内的切片
        for i, current_level in enumerate(slice_levels):
            if not (info['local_min'] - 1e-5 <= current_level <= info['local_max'] + 1e-5):
                continue
                
            slice_2d = slices[i]
            if slice_2d is None: continue
            
            expected_c3d = expected_c3d_func(current_level)
            best_poly = None
            min_area_diff = float('inf') 
            to_3d_mat = slice_2d.metadata.get('to_3D')
            
            if to_3d_mat is None: continue
            
            for poly in slice_2d.polygons_closed:
                poly_r = math.sqrt(poly.area / math.pi)
                if abs(poly_r - r) <= 3.0: 
                    
                    c2d = poly.centroid.coords[0]
                    c3d = trimesh.transformations.transform_points([[c2d[0], c2d[1], 0.0]], to_3d_mat)[0]
                    dist = np.linalg.norm(c3d - expected_c3d)
                    
                    dist_tolerance = (r + info['margin'] + 1.5) if f_type == "Pillar" else (r + info['margin'] + 1.0)
                    
                    if dist < dist_tolerance:
                        area_diff = abs(poly.area - expected_area)
                        if area_diff < min_area_diff:
                            min_area_diff = area_diff
                            best_poly = poly
                            
            if best_poly is not None:
                valid_depths.append(current_level)
                exact_dia = 2 * math.sqrt(best_poly.area / math.pi)
                exact_diameters.append(exact_dia)
                
        # 5. 汇总并更新 JSON 数据
        if valid_depths:
            exact_start = round(min(valid_depths), 2)
            exact_end = round(max(valid_depths), 2)
            exact_main_dia = round(np.mean(exact_diameters), 2)
            
            print(f"    [√] {f_type} 修正成功: 深度区间 [{exact_start}, {exact_end}], 精确等效直径 {exact_main_dia}")
            
            feat["Main_Diameter"] = exact_main_dia
            feat["Shape_Params"] = shape_params
            
            if axis == 'Y':
                feat["Steps"] = [{"Diameter": exact_main_dia, "Shape": shape_type, "Shape_Params": shape_params, "Y_Start": exact_start, "Y_End": exact_end}]
            elif axis == 'X':
                feat["Steps"] = [{"Diameter": exact_main_dia, "Shape": shape_type, "Shape_Params": shape_params, "X_Start": exact_start, "X_End": exact_end}]
            elif axis == 'Z':
                feat["Steps"] = [{"Diameter": exact_main_dia, "Shape": shape_type, "Shape_Params": shape_params, "Z_Start": exact_start, "Z_End": exact_end}]
        else:
            print(f"    [!] 高精度切片未捕捉到有效 {f_type} 结构，保留原数据。")
            
    return data

if __name__ == "__main__":
    stl_file = "current_task.stl"
    input_json_file = "Full_Features_v33.json"   
    output_json_file = "Full_Features_v34.json"  
    
    print(f"\n{'='*50}")
    print("[*] 启动高精度空间特征修正引擎 (按轴分段全局扫描提速版)")
    print(f"{'='*50}")

    if os.path.exists(input_json_file) and os.path.exists(stl_file):
        try:
            # 加载并对齐模型
            mesh = trimesh.load(stl_file)
            offset = align_mesh_to_origin(mesh)
            print(f"[*] 3D 模型加载完毕，已将基准点对齐至 (0,0,0)。全局偏移量: {offset}")

            # 读取粗提取 JSON 数据
            with open(input_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 【核心修改】：将全部数据交给管线一次性统筹扫描
            data = refine_features_globally(mesh, data, high_res=0.01)

            # 保存最终高精度结果
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"\n[+] 全部特征修正完毕！高精度数据已保存至: {output_json_file}")
            print(f"{'='*50}\n")

        except Exception as e:
            print(f"\n[!] 引擎运行报错: {e}")
    else:
        print(f"[!] 错误: 找不到文件 {stl_file} 或 {input_json_file}，请检查当前目录。")