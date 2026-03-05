import cv2
import numpy as np
import torch
import os
from ultralytics.models.sam import SAM3SemanticPredictor

COLOR_MAIN = [200, 100, 0]    
COLORS_SUB = [
    [0, 255, 255], [0, 0, 255], [0, 255, 0], 
    [255, 0, 255], [255, 255, 0], [0, 128, 255]
]
DUPLICATE_DIST_THRESH = 15  

def get_instances(mask_data, target_shape):
    if mask_data is None: return []
    mask = np.any(mask_data.cpu().numpy(), axis=0).astype(np.uint8) if torch.is_tensor(mask_data) else mask_data
    if mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1: continue 
        inst_mask = np.zeros_like(mask)
        inst_mask[labels == i] = 1
        center = (centroids[i][0], centroids[i][1]) 
        instances.append({"mask": inst_mask, "area": area, "center": center})
    return instances

if __name__ == "__main__":
    print("========== [5] 开始运行 SAM3 语义分割脚本 ==========")
    print("[Debug] 正在初始化 SAM3 模型预测器 (Model: sam3.pt)...")
    try:
        overrides = dict(conf=0.15, task="segment", mode="predict", model="sam3.pt", half=True, save=False)
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print("[Info] SAM3 模型加载成功。")
    except Exception as e:
        exit(f"[致命错误] 模型加载失败，请检查依赖: {e}")
    
    base_temp = "data/02_temp"
    base_out = "data/03_output"
    
    if not os.path.exists(base_temp):
        exit("⚠️ [错误] 未找到临时文件夹，请确认是否运行过前面步骤。")
        
    projects = [d for d in os.listdir(base_temp) if os.path.isdir(os.path.join(base_temp, d))]
    
    for project_name in projects:
        print(f"\n==================================================")
        print(f">>> 开始对项目 [{project_name}] 进行智能切片识别")
        
        image_path = f"{base_temp}/{project_name}/raw_slices/Z/top_view_render.png"
        out_dir = f"{base_out}/{project_name}/sam3_results"
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"  [Debug] 寻找 Z 轴顶视图渲染图: {image_path}")
        if not os.path.exists(image_path):
            print(f"  [跳过] 未找到渲染图，可能是第一步渲染失败。")
            continue
            
        original_img = cv2.imread(image_path)
        if original_img is None: 
            print(f"  [错误] OpenCV 读取图片失败。")
            continue
            
        h, w = original_img.shape[:2]
        print(f"  [Info] 图片读取成功，分辨率: {w}x{h}")
        predictor.set_image(image_path)

        prompt_main = input(f"\n[交互] 1. 请输入 [{project_name}] 的 [大物体/主体] 提示词: ")
        print(f"  [Debug] 正在向 SAM3 发送推理请求 (Prompt: {prompt_main})...")
        res_main = predictor(text=[prompt_main])
        main_insts = get_instances(res_main[0].masks.data if res_main[0].masks else None, (h, w))
        
        if not main_insts: 
            print(f"  [警告] 未在图像中识别到 '{prompt_main}'，跳过当前项目。")
            continue

        main_obj = max(main_insts, key=lambda x: x['area'])
        mask_main, area_main = main_obj['mask'], main_obj['area']
        print(f"  [Info] ✅ 成功锁定主体特征，像素面积: {area_main} px")

        masked_img_input = np.zeros_like(original_img)
        masked_img_input[mask_main == 1] = original_img[mask_main == 1]
        step1_save_path = os.path.join(out_dir, "step1_cropped_object.png")
        cv2.imwrite(step1_save_path, masked_img_input)
        print(f"  [Debug] 裁剪出的主体图像已缓存: {step1_save_path}")

        predictor.set_image(masked_img_input)
        prompts_input = input(f"\n[交互] 2. 请输入主体内部的 [小零件] 提示词(英文逗号分隔): ").split(",")
        prompts_sub = [p.strip() for p in prompts_input if p.strip()]

        thresholds_dict = {}
        for p in prompts_sub:
            try:
                val = input(f"  ↳ 请输入 [{p}] 的面积占比过滤阈值 (如 0.1 代表 0.1%): ")
                thresholds_dict[p] = float(val) if val else 0.0
            except ValueError:
                thresholds_dict[p] = 0.0

        final_results = [] 
        print(f"\n  [Debug] 开始进行子特征逐一识别与过滤...")

        for i, p_sub in enumerate(prompts_sub):
            print(f"    -> 正在推理: {p_sub} ...", end=" ")
            res = predictor(text=[p_sub])
            if res[0].masks is None: 
                print("未发现实体。")
                continue
            
            current_instances = get_instances(res[0].masks.data, (h, w))
            valid_sub_count = 0
            
            for inst in current_instances:
                ratio = (inst['area'] / area_main) * 100
                if ratio < thresholds_dict[p_sub]:
                    continue
                    
                is_dup = False
                for exist in final_results:
                    dist = np.linalg.norm(np.array(inst['center']) - np.array(exist['pos_px']))
                    if dist < DUPLICATE_DIST_THRESH:
                        if p_sub not in exist['labels']: 
                            exist['labels'].append(p_sub)
                        is_dup = True
                        break
                
                if not is_dup:
                    final_results.append({
                        "pos_px": inst['center'],
                        "area": inst['area'],
                        "labels": [p_sub],
                        "mask": inst['mask']
                    })
                    valid_sub_count += 1
            print(f"发现并保留 {valid_sub_count} 个实体。")

        final_output = np.zeros((h, w, 4), dtype=np.uint8)
        final_output[mask_main == 1] = COLOR_MAIN + [255]

        log_lines = [
            f"Summary Report (Per-Label Threshold Filtering)\n",
            f"Project: {project_name}\n",
            f"Main Object: {prompt_main} | Total Area: {area_main} px\n",
            f"Image Resolution: {w}x{h}\n",
            f"Thresholds Config: {thresholds_dict}\n",
            "-"*60 + "\n"
        ]

        for i, res in enumerate(final_results):
            color = COLORS_SUB[i % len(COLORS_SUB)]
            final_output[res['mask'] == 1] = color + [255]
            
            norm_x = round(res['pos_px'][0] / w, 6)
            norm_y = round(res['pos_px'][1] / h, 6)
            
            ratio = (res['area'] / area_main) * 100
            label_name = f"{res['labels'][0]}_{i+1}"
            note = f" (Includes: {', '.join(res['labels'][1:])})" if len(res['labels']) > 1 else ""
            
            line = f"Label: {label_name}{note} | Ratio: {ratio:.4f}% | NormPos: ({norm_x}, {norm_y})\n"
            log_lines.append(line)

        out_base = os.path.join(out_dir, f"result_{prompt_main}")
        cv2.imwrite(f"{out_base}.png", final_output)
        with open(f"{out_base}_data.txt", "w", encoding="utf-8") as f:
            f.writelines(log_lines)

        print(f"\n  [Info] 🎉 项目 [{project_name}] 处理完毕！")
        print(f"  [Info] 最终分割图: {out_base}.png")
        print(f"  [Info] 坐标数据记录: {out_base}_data.txt")
    print(f"========== [5] 脚本运行结束 ==========")