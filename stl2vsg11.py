import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET

def slice_stl_to_svg_fixed_orientation(stl_path, output_dir, layer_height=1.0, padding=2.0, slice_direction=[0, 0, 1], max_slices=80):
    # 1. 加载模型
    print(f"\n[*] 正在加载模型: {stl_path} ...")
    mesh = trimesh.load(stl_path)
    
    # ================== 【视角矩阵变换】 ==================
    target_dir = np.array(slice_direction)
    target_dir = target_dir / np.linalg.norm(target_dir) # 归一化
    transform = np.eye(4)
    
    if np.allclose(target_dir, [0, 0, 1]): 
        print("[*] 模式：Z轴切片 (俯视图 XY)")
        transform = np.eye(4) 
    elif np.allclose(target_dir, [1, 0, 0]):
        print("[*] 模式：X轴切片 (侧视图 YZ)")
        transform = np.array([
            [0, 1, 0, 0],  
            [0, 0, 1, 0],  
            [1, 0, 0, 0],  
            [0, 0, 0, 1]
        ])
    elif np.allclose(target_dir, [0, 1, 0]):
        print("[*] 模式：Y轴切片 (正视图 XZ)")
        transform = np.array([
            [1, 0, 0, 0],  
            [0, 0, 1, 0],  
            [0, 1, 0, 0],  
            [0, 0, 0, 1]
        ])
    else:
        print(f"[*] 模式：任意方向 {slice_direction}，使用自动对齐")
        z_axis = np.array([0, 0, 1])
        transform = trimesh.geometry.align_vectors(target_dir, z_axis)

    # 应用变换
    mesh.apply_transform(transform)
    # ==========================================================

    # 2. 全局尺寸计算 
    bounds = mesh.bounds
    min_x, min_y = bounds[0, 0], bounds[0, 1]
    max_x, max_y = bounds[1, 0], bounds[1, 1]
    z_min, z_max = bounds[:, 2]
    
    model_width = max_x - min_x
    model_height = max_y - min_y
    canvas_width = model_width + (padding * 2)
    canvas_height = model_height + (padding * 2)
    
    # ================== 【核心优化：自适应层厚计算】 ==================
    total_depth = z_max - z_min
    expected_slices = total_depth / layer_height
    
    if expected_slices > max_slices:
        # 如果超过80张，动态重新计算层厚
        actual_layer_height = total_depth / max_slices
        print(f"[!] 触发自适应降维：当前方向总厚度 {total_depth:.2f}mm，1mm切片将达 {int(expected_slices)} 张。")
        print(f"[!] 已将切片层厚自动增加至: {actual_layer_height:.3f}mm，确保总切片数为 {max_slices} 张。")
        layer_height = actual_layer_height
    else:
        print(f"[+] 尺寸合规：当前方向总厚度 {total_depth:.2f}mm，预计生成 {int(expected_slices)} 张切片 (未超限)。")
    # ==================================================================

    # 3. 准备切片高度
    z_levels = np.arange(z_min + layer_height/2, z_max, layer_height)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[*] 开始生成 SVG... 实际层数: {len(z_levels)}")

    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace('', SVG_NS)

    for index, z in enumerate(z_levels):
        slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])

        if slice_3d is None:
            continue

        vertices = slice_3d.vertices[:, :2] 
        vertices[:, 0] = vertices[:, 0] - min_x + padding
        vertices[:, 1] = max_y - vertices[:, 1] + padding 
        
        path_2d = trimesh.path.Path2D(entities=slice_3d.entities, vertices=vertices)
        svg_string = path_2d.export(file_type='svg')
        
        try:
            root = ET.fromstring(svg_string)
            root.set('width', f'{canvas_width:.2f}mm')
            root.set('height', f'{canvas_height:.2f}mm')
            root.set('viewBox', f'0 0 {canvas_width:.2f} {canvas_height:.2f}')
            
            direction_str = str(slice_direction)
            layer_num = str(index + 1)
            thickness_str = f"{layer_height:.3f}"
            height_str = f"{z:.3f}"
            
            root.set('data-slice-direction', direction_str)
            root.set('data-layer-number', layer_num)
            root.set('data-layer-thickness', thickness_str)
            root.set('data-current-height', height_str)
            
            # 这里的 layer-thickness 已经变成了动态调整后的真实厚度
            desc_text = f"Dir:{direction_str} | layer_num:{layer_num} |layer-thickness:{thickness_str} | Length_Dir:{height_str}"
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
            
    print("[+] 当前方向切片全部导出完毕。\n" + "-"*40)

if __name__ == "__main__":
    stl_file = 'current_task.stl'
    
    # 通过 max_slices 参数控制最大张数，如果你的模型特别长，它会自动把 layer_height 从 1 变大
    slice_stl_to_svg_fixed_orientation(stl_file, './Out_X', layer_height=1, slice_direction=[1, 0, 0], max_slices=50)
    slice_stl_to_svg_fixed_orientation(stl_file, './Out_Y', layer_height=1, slice_direction=[0, 1, 0], max_slices=50)
    slice_stl_to_svg_fixed_orientation(stl_file, './Out_Z', layer_height=1, slice_direction=[0, 0, 1], max_slices=50)