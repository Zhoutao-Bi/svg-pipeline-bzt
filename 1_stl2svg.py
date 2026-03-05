import trimesh
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET

def slice_stl_to_svg_fixed_orientation(stl_path, output_dir, layer_height=0.05, padding=2.0, slice_direction=[0, 0, 1]):
    print(f"    [Info] 加载模型: {os.path.basename(stl_path)} | 方向: {slice_direction}")
    try:
        mesh = trimesh.load(stl_path)
        print(f"    [Debug] 模型加载成功，包含 {len(mesh.vertices)} 个顶点")
    except Exception as e:
        print(f"    [错误] 模型加载失败: {e}")
        return

    target_dir = np.array(slice_direction)
    target_dir = target_dir / np.linalg.norm(target_dir) 
    
    transform = np.eye(4)
    if np.allclose(target_dir, [0, 0, 1]): 
        print("    [Debug] 模式匹配：Z轴切片 (俯视图 XY)")
        transform = np.eye(4)
    elif np.allclose(target_dir, [1, 0, 0]):
        print("    [Debug] 模式匹配：X轴切片 (侧视图 YZ)")
        transform = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    elif np.allclose(target_dir, [0, 1, 0]):
        print("    [Debug] 模式匹配：Y轴切片 (正视图 XZ)")
        transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    else:
        z_axis = np.array([0, 0, 1])
        transform = trimesh.geometry.align_vectors(target_dir, z_axis)

    mesh.apply_transform(transform)
    os.makedirs(output_dir, exist_ok=True)

    bounds = mesh.bounds
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    
    model_width = max_x - min_x
    model_height = max_y - min_y
    canvas_width = model_width + (padding * 2)
    canvas_height = model_height + (padding * 2)
    
    try:
        print(f"    [Debug] 准备渲染正交投影图...")
        scale_px = 20 
        render_w = int(canvas_width * scale_px)
        render_h = int(canvas_height * scale_px)

        scene = trimesh.Scene(mesh)
        scene.camera.orthographic = True
        scene.camera.xmag = canvas_width / 2
        scene.camera.ymag = canvas_height / 2

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        camera_pos = [center_x, center_y, max_z + 100]
        
        camera_transform = np.eye(4)
        camera_transform[:3, 3] = camera_pos
        scene.camera_transform = camera_transform

        png_data = scene.save_image(resolution=[render_w, render_h])
        render_path = os.path.join(output_dir, "top_view_render.png")
        with open(render_path, 'wb') as f:
            f.write(png_data)
        print(f"    [Info] 渲染图已保存: {render_w}x{render_h} 像素")

    except Exception as e:
        print(f"    [警告] 渲染过程出错: {e}")

    z_min, z_max = bounds[:, 2]
    z_levels = np.arange(z_min + layer_height/2, z_max, layer_height)
    print(f"    [Debug] 开始生成 SVG 切片，层厚: {layer_height}mm，共计: {len(z_levels)} 层...")

    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace('', SVG_NS)
    
    valid_slice_count = 0
    for index, z in enumerate(z_levels):
        slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if slice_3d is None: continue

        valid_slice_count += 1
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
            height_str = f"{index * layer_height:.3f}"
            root.set('data-slice-direction', direction_str)
            root.set('data-layer-number', layer_num)
            root.set('data-current-height', height_str)
            
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
            
    print(f"    [Info] 切片处理完成！实际生成了 {valid_slice_count} 个非空切片文件。")

if __name__ == "__main__":
    base_temp = "data/02_temp"
    print(f"========== [1] 开始运行 STL 切片脚本 ==========")
    if not os.path.exists(base_temp):
        exit("⚠️ [错误] 未找到临时文件夹，请先运行 0_step2stl.py")
        
    projects = [d for d in os.listdir(base_temp) if os.path.isdir(os.path.join(base_temp, d))]
    
    if len(sys.argv) > 1:
        projects = [p for p in projects if p == sys.argv[1]]
    
    for project_name in projects:
        stl_file = os.path.join(base_temp, f"{project_name}/stl/{project_name}.stl")
        
        if os.path.exists(stl_file):
            print(f"\n>>> 正在为项目 [{project_name}] 进行全向切片...")
            raw_dir = os.path.join(base_temp, f"{project_name}/raw_slices")
            
            slice_stl_to_svg_fixed_orientation(stl_file, f"{raw_dir}/X", layer_height=2, slice_direction=[1, 0, 0])
            slice_stl_to_svg_fixed_orientation(stl_file, f"{raw_dir}/Y", layer_height=2, slice_direction=[0, 1, 0])
            slice_stl_to_svg_fixed_orientation(stl_file, f"{raw_dir}/Z", layer_height=2, slice_direction=[0, 0, 1])
        else:
            print(f"  [警告] 找不到项目 [{project_name}] 的 STL 文件，跳过。")
            
    print(f"========== [1] 脚本运行结束 ==========")