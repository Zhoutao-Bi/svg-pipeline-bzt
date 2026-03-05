import gmsh
import os
import glob
import shutil
import sys

def convert_step_to_stl(step_path, stl_path):
    print(f"    [Debug] 初始化 gmsh 引擎...")
    gmsh.initialize()
    gmsh.model.add("Model")
    
    print(f"    [Debug] 正在导入 STEP 形状数据: {step_path}")
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    
    print(f"    [Debug] 配置网格精度参数 (MeshSizeMax = 1.0)...")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
    
    print(f"    [Debug] 开始生成 3D 网格 (这可能需要几秒钟)...")
    gmsh.model.mesh.generate(3)
    
    print(f"    [Debug] 正在导出 STL 文件至: {stl_path}")
    gmsh.write(stl_path)
    gmsh.finalize()
    print(f"    [Info] 转换完成: {os.path.basename(stl_path)}")

if __name__ == "__main__":
    input_dir = "data/01_input"
    print(f"========== [0] 开始运行 模型准备 脚本 ==========")
    print(f"[Debug] 检查输入目录: {input_dir}")
    os.makedirs(input_dir, exist_ok=True)
    
    # 支持 .stp, .step 和 .stl
    input_files = glob.glob(os.path.join(input_dir, "*.stp")) + \
                  glob.glob(os.path.join(input_dir, "*.step")) + \
                  glob.glob(os.path.join(input_dir, "*.stl"))
                  
    print(f"[Debug] 扫描到 {len(input_files)} 个输入文件")
    
    # 接收外部指定项目名
    if len(sys.argv) > 1:
        target_proj = sys.argv[1]
        input_files = [f for f in input_files if os.path.splitext(os.path.basename(f))[0] == target_proj]
    
    if not input_files:
        print(f"⚠️ [警告] 在 {input_dir} 中没有找到待处理的文件。")
    
    for file_path in input_files:
        project_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()
        
        stl_dir = f"data/02_temp/{project_name}/stl"
        os.makedirs(stl_dir, exist_ok=True)
        stl_path = os.path.join(stl_dir, f"{project_name}.stl")
        
        print(f"\n>>> 正在处理项目: [{project_name}]")
        
        if ext == '.stl':
            print(f"    [Info] 检测到输入已经是 STL 文件，跳过转换，直接复制...")
            shutil.copy(file_path, stl_path)
            print(f"    [Info] 复制完成: {os.path.basename(stl_path)}")
        else:
            convert_step_to_stl(file_path, stl_path)
        
    print(f"========== [0] 脚本运行结束 ==========")