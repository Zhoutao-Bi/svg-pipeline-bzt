import trimesh
import numpy as np
import os

def calculate_iou_aligned(stl_path_1, stl_path_2, sample_count=100000):
    """
    计算两个 STL 模型的 3D IoU，并自动将它们对齐到几何中心。
    
    Args:
        stl_path_1 (str): 第一个 STL 文件路径 (通常是 Ground Truth)
        stl_path_2 (str): 第二个 STL 文件路径 (通常是生成的模型)
        sample_count (int): 采样点数量。
        
    Returns:
        float: IoU 值 (0.0 到 1.0)
    """
    # 1. 加载模型
    try:
        mesh1 = trimesh.load(stl_path_1)
        mesh2 = trimesh.load(stl_path_2)
    except Exception as e:
        print(f"Error loading meshes: {e}")
        return 0.0

    print(f"\n--- Loading Models ---")
    print(f"Mesh 1: {os.path.basename(stl_path_1)} | Watertight: {mesh1.is_watertight}")
    print(f"Mesh 2: {os.path.basename(stl_path_2)} | Watertight: {mesh2.is_watertight}")

    if not mesh1.is_watertight or not mesh2.is_watertight:
        print("Warning: One or both meshes are not watertight. Calculation might be inaccurate.")

    # 2. 自动对齐：将两个模型移动到原点 (0,0,0)
    # 计算几何中心 (Centroid)
    centroid1 = mesh1.centroid
    centroid2 = mesh2.centroid
    
    print(f"\n--- Pre-Alignment Centroids ---")
    print(f"Mesh 1: {centroid1}")
    print(f"Mesh 2: {centroid2}")

    # 移动模型
    mesh1.apply_translation(-centroid1)
    mesh2.apply_translation(-centroid2)
    print(">>> Applied translation to center both meshes at (0,0,0).")

    # 3. 确定采样空间（联合包围盒）
    bounds_min = np.minimum(mesh1.bounds[0], mesh2.bounds[0])
    bounds_max = np.maximum(mesh1.bounds[1], mesh2.bounds[1])
    
    # 4. 生成随机采样点
    points = np.random.uniform(low=bounds_min, high=bounds_max, size=(sample_count, 3))

    # 5. 检查点包含关系
    print(f"\n--- Sampling {sample_count} points ---")
    # mesh.contains 返回布尔数组
    inside_1 = mesh1.contains(points)
    inside_2 = mesh2.contains(points)

    # 6. 计算 IoU
    intersection = np.logical_and(inside_1, inside_2)
    union = np.logical_or(inside_1, inside_2)

    count_intersection = np.sum(intersection)
    count_union = np.sum(union)
    
    # 防止除以零
    if count_union == 0:
        print("Union volume is zero.")
        return 0.0

    iou = count_intersection / count_union
    
    print(f"Stats:")
    print(f"  Volume 1 (Approx points): {np.sum(inside_1)}")
    print(f"  Volume 2 (Approx points): {np.sum(inside_2)}")
    print(f"  Intersection Points:      {count_intersection}")
    print(f"  Union Points:             {count_union}")
    
    return iou

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你自己的 stl 文件路径
    # file1 = "20251031_Body.stl"
    # file2 = "Cruciform_Base_Model.stl"
    
    # 让你能够手动输入文件名，或者直接修改上面的变量
    file1 = input("请输入第一个 STL 文件路径 (直接回车默认 '20251031_Body.stl'): ").strip() or "20251031_Body.stl"
    file2 = input("请输入第二个 STL 文件路径 (直接回车默认 'Cruciform_Base_Model.stl'): ").strip() or "Cruciform_Base_Model.stl"

    if os.path.exists(file1) and os.path.exists(file2):
        iou_score = calculate_iou_aligned(file1, file2)
        print(f"\n====== Final Calculated IoU: {iou_score:.4f} ======")
    else:
        print(f"\n错误: 找不到文件。请确保 {file1} 和 {file2} 在当前目录下。")