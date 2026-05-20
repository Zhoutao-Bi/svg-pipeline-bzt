import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET

def slice_stl_to_svg_fixed_orientation(stl_path, output_dir, layer_height=0.05, padding=2.0, slice_direction=[0, 0, 1]):
    # 1. 加载模型
    print(f"正在加载模型: {stl_path} ...")
    mesh = trimesh.load(stl_path)
    
    # ================== 【修正核心：强制视角矩阵】 ==================
    # 目标：无论切哪个轴，都把那个轴转到 Z，同时保持原本的 Z 轴尽可能竖直（对应 SVG 的 Y）
    
    target_dir = np.array(slice_direction)
    target_dir = target_dir / np.linalg.norm(target_dir) # 归一化
    
    # 初始化变换矩阵 (4x4 单位矩阵)
    transform = np.eye(4)
    
    # 判断切片方向，构建特定的“观察矩阵”
    # 注意：矩阵的行代表新坐标轴的来源
    
    if np.allclose(target_dir, [0, 0, 1]): 
        # Case A: 切 Z 轴 (默认) -> 俯视图
        # 新X = 旧X, 新Y = 旧Y, 新Z = 旧Z
        print("模式：Z轴切片 (俯视图 XY)")
        transform = np.eye(4) # 不变
        
    elif np.allclose(target_dir, [1, 0, 0]):
        # Case B: 切 X 轴 -> 侧视图 (YZ平面)
        # 我们希望：生成的 2D 图中，横向是 Y，纵向是 Z
        # 所以：新X <= 旧Y,  新Y <= 旧Z,  新Z <= 旧X (切片厚度方向)
        print("模式：X轴切片 (侧视图 YZ)")
        transform = np.array([
            [0, 1, 0, 0],  # New X axis gets Old Y
            [0, 0, 1, 0],  # New Y axis gets Old Z (保持高度竖直)
            [1, 0, 0, 0],  # New Z axis gets Old X (切片方向)
            [0, 0, 0, 1]
        ])
        
    elif np.allclose(target_dir, [0, 1, 0]):
        # Case C: 切 Y 轴 -> 正视图 (XZ平面)
        # 我们希望：生成的 2D 图中，横向是 X，纵向是 Z
        # 所以：新X <= 旧X,  新Y <= 旧Z,  新Z <= 旧Y (切片厚度方向)
        print("模式：Y轴切片 (正视图 XZ)")
        transform = np.array([
            [1, 0, 0, 0],  # New X axis gets Old X
            [0, 0, 1, 0],  # New Y axis gets Old Z (保持高度竖直)
            [0, 1, 0, 0],  # New Z axis gets Old Y (切片方向)
            [0, 0, 0, 1]
        ])
        
    else:
        # Case D: 任意斜向切片 (降级使用自动对齐)
        print(f"模式：任意方向 {slice_direction}，使用自动对齐")
        z_axis = np.array([0, 0, 1])
        transform = trimesh.geometry.align_vectors(target_dir, z_axis)

    # 应用变换
    mesh.apply_transform(transform)
    # ==========================================================

    # 2. 全局尺寸计算 (此时 mesh 已经是旋转后的标准姿态)
    bounds = mesh.bounds
    min_x, min_y = bounds[0, 0], bounds[0, 1]
    max_x, max_y = bounds[1, 0], bounds[1, 1]
    
    model_width = max_x - min_x
    model_height = max_y - min_y
    canvas_width = model_width + (padding * 2)
    canvas_height = model_height + (padding * 2)
    
    # 3. 准备切片高度
    z_min, z_max = bounds[:, 2]
    z_levels = np.arange(z_min + layer_height/2, z_max, layer_height)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"开始切片... 方向: {slice_direction}, 层数: {len(z_levels)}")

    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace('', SVG_NS)

    for index, z in enumerate(z_levels):
        # 始终切变换后的 Z 轴
        slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])

        if slice_3d is None:
            continue

        # 坐标变换 (3D -> 2D)
        vertices = slice_3d.vertices[:, :2] 
        vertices[:, 0] = vertices[:, 0] - min_x + padding
        vertices[:, 1] = max_y - vertices[:, 1] + padding 
        
        path_2d = trimesh.path.Path2D(entities=slice_3d.entities, vertices=vertices)
        svg_string = path_2d.export(file_type='svg')
        
        # XML 处理部分 (保持不变)
        try:
            root = ET.fromstring(svg_string)
            root.set('width', f'{canvas_width:.2f}mm')
            root.set('height', f'{canvas_height:.2f}mm')
            root.set('viewBox', f'0 0 {canvas_width:.2f} {canvas_height:.2f}')
            
            direction_str = str(slice_direction)
            layer_num = str(index + 1)
            thickness_str = str(layer_height)
            height_str = f"{z:.3f}"
            
            root.set('data-slice-direction', direction_str)
            root.set('data-layer-number', layer_num)
            root.set('data-layer-thickness', thickness_str)
            root.set('data-current-height', height_str)
            
            desc_text = f"Dir:{direction_str} | layer_num:{layer_num} |layer-thickness:{layer_height} | Length_Dir:{height_str}"
            desc_node = ET.Element(f"{{{SVG_NS}}}desc")
            desc_node.text = desc_text
            root.insert(0, desc_node)
            
            ns = {'svg': SVG_NS} 
            for path in root.findall('.//svg:path', ns):
                path.set('fill', 'none')
                path.set('stroke', 'black')
                path.set('stroke-width', '0.05') 
            
            final_svg = ET.tostring(root, encoding='unicode')
        except Exception:
            final_svg = svg_string

        svg_filename = os.path.join(output_dir, f"layer_{index:04d}.svg")
        with open(svg_filename, 'w') as f:
            f.write(final_svg)
            
    print("切片完成。")

if __name__ == "__main__":
    stl_file = 'current_task.stl'
    
    # 场景1：切 X 轴 (得到侧面轮廓，Z轴依然朝上)
    slice_stl_to_svg_fixed_orientation(stl_file, './Out_X', layer_height=1, slice_direction=[1, 0, 0])
    
    # 场景2：切 Y 轴 (得到正面轮廓，Z轴依然朝上)
    slice_stl_to_svg_fixed_orientation(stl_file, './Out_Y', layer_height=1, slice_direction=[0, 1, 0])

    slice_stl_to_svg_fixed_orientation(stl_file, './Out_Z', layer_height=1, slice_direction=[0, 0, 1])