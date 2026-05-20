import trimesh
import numpy as np

def orient_stl_model(input_file, output_file, mode='stable'):
    """
    读取并摆正STL模型
    
    参数:
        input_file: 输入的STL文件路径
        output_file: 导出的STL文件路径
        mode: 摆正模式
              'stable' - 计算物理稳定姿态（底面平放在地上）
              'rotate_x' - 绕X轴旋转90度（修复Y-up到Z-up的坐标系问题）
    """
    print(f"正在读取模型: {input_file} ...")
    # 1. 读取 STL 文件
    mesh = trimesh.load(input_file)
    
    if not isinstance(mesh, trimesh.Trimesh):
        print("错误：读取的文件不是一个有效的3D网格模型。")
        return

    print(f"读取成功！模型包含 {len(mesh.vertices)} 个顶点, {len(mesh.faces)} 个面。")

    if mode == 'stable':
        # 方法一：自动寻找最稳定的放置面
        print("正在计算模型的物理最稳定姿态...")
        # compute_stable_poses 会返回多个能稳定放置的变换矩阵以及对应的概率
        transforms, probabilities = trimesh.poses.compute_stable_poses(mesh)
        
        if len(transforms) > 0:
            # 找到概率最大的（也就是最稳定的）那个姿态
            best_transform = transforms[np.argmax(probabilities)]
            mesh.apply_transform(best_transform)
            print("已应用最稳定姿态。")
        else:
            print("未能找到稳定姿态。")
            
    elif mode == 'rotate_x':
        # 方法二：坐标系轴向修正（绕 X 轴旋转 90 度）
        print("正在将模型绕X轴旋转90度...")
        # 生成绕 X 轴 (1, 0, 0) 旋转 90 度 (π/2) 的 4x4 变换矩阵
        matrix = trimesh.transformations.rotation_matrix(np.pi / 2,)
        mesh.apply_transform(matrix)
        print("已应用旋转变换。")

    # --- 辅助对齐操作 ---
    # 通常摆正后，我们还希望模型居中，并且刚好踩在地面上 (Z=0)
    
    # 1. X轴和Y轴居中 (根据质心)
    mesh.apply_translation([-mesh.centroid[ 0 ], -mesh.centroid[ 1 ], 0])
    
    # 2. Z轴底部对齐到0 (根据包围盒的最低点 bounds 是 [min_x, min_y, min_z])
    mesh.apply_translation([0, 0, -mesh.bounds[ 0 ][ 2 ]])
    print("模型已水平居中，并对齐到底面 (Z=0)。")

    # 3. 导出模型
    mesh.export(output_file)
    print(f"摆正后的模型已成功保存至: {output_file}")


# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    input_stl_path = "00.stl"   # 替换为你的输入文件路径
    output_stl_path = "00.stl" # 替换为你想保存的输出路径

    # 如果模型形状不规则，想自动找个平坦的面立在地上，使用 'stable'
    orient_stl_model(input_stl_path, output_stl_path, mode='stable')

    # 如果模型只是因为导出的软件坐标系不对躺下了，使用 'rotate_x'
    # orient_stl_model(input_stl_path, output_stl_path, mode='rotate_x')