import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["__root__.txt"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))
import numpy as np
np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_

import argparse
from general_utils.huggingface_model_utils import load_model_by_repo_id
from general_utils.img_utils import visualize # 可选，用于可视化
from general_utils.img_utils import prepare_text_img # 可选，用于可视化
from torchvision.transforms import Compose, ToTensor, Normalize, transforms
from PIL import Image
import pandas as pd
import torch
import inspect
import time # 导入 time 模块
import shutil # 导入 shutil 模块

# 从 config.py 导入所有常量和配置
import config

def pil_to_input(pil_image, device='cuda'):
    # input is a rgb image normalized.
    trans = Compose([
        transforms.Resize((112, 112)),
        ToTensor(),
        Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    input = trans(pil_image).unsqueeze(0).to(device)
    return input

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='持续人脸识别服务')
    parser.add_argument('--recognition_model_id', type=str, default=config.RECOGNITION_MODEL_ID,
                        help='用于人脸识别的模型ID。')
    parser.add_argument('--aligner_id', type=str, default=config.ALIGNER_ID,
                        help='用于人脸对齐的模型ID。')
    parser.add_argument('--input_image_dir', type=str, required=True,
                        help='包含待查询图片的目录路径。这些图片将在处理后被移动到dealed_dir。')
    parser.add_argument('--gallery_dir', type=str, required=True,
                        help='包含固定图库图片的目录路径。这些图片不会被移动。')
    parser.add_argument('--threshold', type=float, default=config.THRESHOLD,
                        help='判断是否匹配的相似度阈值。')
    parser.add_argument('--output_csv', type=str, default=config.OUTPUT_CSV_FILENAME,
                        help='识别结果输出的CSV文件路径。')
    parser.add_argument('--scan_interval', type=int, default=config.SCAN_INTERVAL_SECONDS,
                        help='扫描新图片的时间间隔（秒）。')
    args = parser.parse_args()

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    fr_model = load_model_by_repo_id(repo_id=args.recognition_model_id,
                                     save_path=os.path.expanduser(f'~/.cvlface_cache/{args.recognition_model_id}'),
                                     HF_TOKEN=os.environ.get('HF_TOKEN', ''),
                                     ).to(device)
    aligner = load_model_by_repo_id(repo_id=args.aligner_id,
                                    save_path=os.path.expanduser(f'~/.cvlface_cache/{args.aligner_id}'),
                                    HF_TOKEN=os.environ.get('HF_TOKEN', ''),
                                    ).to(device)

    # 提前处理图库图片（gallery_dir），提取所有图库特征，这些图片是固定不变的
    print(f"正在加载图库图片并提取特征从: {args.gallery_dir}")
    gallery_feats_data = [] # 存储 (特征, 相对路径)
    gallery_image_paths = [] # 收集所有图库图片的完整路径
    for root_dir, _, files in os.walk(args.gallery_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                gallery_image_paths.append(os.path.join(root_dir, file))
    
    if not gallery_image_paths:
        print(f"错误: 在图库目录 {args.gallery_dir} 中未找到任何图片。程序将退出。")
        sys.exit(1)

    print(f"在 {args.gallery_dir} 中找到 {len(gallery_image_paths)} 张图库图片。")

    batch_size = config.BATCH_SIZE # 从配置中获取批量大小
    
    # 循环遍历图库图片，分批处理并提取特征
    for i in range(0, len(gallery_image_paths), batch_size):
        batch_paths = gallery_image_paths[i:i + batch_size]
        print(f"  正在加载图库批次 {i//batch_size + 1}/{len(gallery_image_paths)//batch_size + (1 if len(gallery_image_paths)%batch_size else 0)}...")
        
        batch_images_for_inference = []
        batch_relative_paths = []

        for gallery_img_path in batch_paths:
            relative_gallery_path = os.path.relpath(gallery_img_path, args.gallery_dir)
            try:
                gallery_img = Image.open(gallery_img_path)
                if gallery_img.mode != 'RGB':
                    gallery_img = gallery_img.convert('RGB')
                batch_images_for_inference.append(pil_to_input(gallery_img, device))
                batch_relative_paths.append(relative_gallery_path)
            except Exception as e:
                print(f"加载图库图片 {gallery_img_path} 时出错: {e}。跳过此图片。")
        
        if not batch_images_for_inference:
            continue

        batch_input_tensors = torch.cat(batch_images_for_inference, dim=0)

        try:
            with torch.no_grad():
                aligned_gallery_x_batch, _, aligned_ldmks_gallery_batch, _, _, _ = aligner(batch_input_tensors)
                input_signature = inspect.signature(fr_model.model.net.forward) # 获取模型forward方法的签名
                if input_signature.parameters.get('keypoints') is not None:
                    gallery_feat_batch = fr_model(aligned_gallery_x_batch, aligned_ldmks_gallery_batch)
                else:
                    gallery_feat_batch = fr_model(aligned_gallery_x_batch)

            for j, feat in enumerate(gallery_feat_batch):
                gallery_feats_data.append((feat, batch_relative_paths[j]))

            del batch_input_tensors, aligned_gallery_x_batch, aligned_ldmks_gallery_batch, gallery_feat_batch
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"处理图库批次图片时模型推理出错: {e}。此批次图库图片特征提取失败。")
            if 'batch_input_tensors' in locals():
                del batch_input_tensors
            torch.cuda.empty_cache()
    
    if not gallery_feats_data:
        print("错误: 未能成功提取任何图库图片的特征。程序将退出。")
        sys.exit(1)
    
    print(f"成功提取 {len(gallery_feats_data)} 张图库图片的特征。")
    # 将所有图库特征堆叠成一个张量，并保存对应的相对路径
    all_gallery_feats = torch.stack([item[0] for item in gallery_feats_data]).to(device)
    all_gallery_relative_paths = [item[1] for item in gallery_feats_data]
    del gallery_feats_data # 释放内存，因为特征已堆叠

    # 定义并创建处理后的输入图片目录，在 input_image_dir 中
    dealed_dir_name = config.DEALED_DIR_NAME # 从配置中获取目录名
    dealed_dir = os.path.join(args.input_image_dir, dealed_dir_name)
    os.makedirs(dealed_dir, exist_ok=True)
    print(f"处理后的查询图片将移至: {dealed_dir}")

    print("程序初始化完毕，开始监控新图片...") # 新增的日志信息

    print(f"开始监控输入目录: {args.input_image_dir}，每隔 {args.scan_interval} 秒扫描一次新图片。")

    while True:
        # 获取 input_image_dir 中所有图片路径，排除 dealed_dir
        current_input_image_paths = []
        for root_dir, dirs, files in os.walk(args.input_image_dir):
            # 排除 'dealed_dir' 目录及其子目录
            if dealed_dir_name in dirs:
                dirs.remove(dealed_dir_name) # 这会阻止 os.walk 遍历 'dealed_dir'
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root_dir, file)
                    current_input_image_paths.append(full_path)

        if current_input_image_paths:
            print(f"检测到 {len(current_input_image_paths)} 张新的查询图片，开始处理。")
            
            new_recognition_results = [] # 收集当前扫描周期内的所有结果

            # 逐张处理每一张新的查询图片
            for query_img_path in current_input_image_paths:
                current_query_base_name = os.path.basename(query_img_path)
                print(f"\n正在处理查询图片: {query_img_path}")
                
                query_processed_successfully = False # 标记查询图片是否成功提取特征
                current_query_feat = None

                try:
                    query_img = Image.open(query_img_path)
                    if query_img.mode != 'RGB':
                        query_img = query_img.convert('RGB')

                    with torch.no_grad():
                        query_input = pil_to_input(query_img, device)
                        aligned_query_x, orig_pred_ldmks_query, aligned_ldmks_query, score_query, thetas_query, normalized_bbox_query = aligner(query_input)

                        # 再次获取fr_model.model.net.forward的签名以确保一致性
                        input_signature_for_query = inspect.signature(fr_model.model.net.forward) 
                        if input_signature_for_query.parameters.get('keypoints') is not None:
                            current_query_feat = fr_model(aligned_query_x, aligned_ldmks_query)
                        else:
                            current_query_feat = fr_model(aligned_query_x)
                        
                        query_processed_successfully = True

                    del query_input, aligned_query_x, orig_pred_ldmks_query, aligned_ldmks_query, score_query, thetas_query, normalized_bbox_query
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"处理查询图片 {query_img_path} 时出错: {e}。跳过此图片的比对。")
                    # 添加错误结果条目，表示查询图片本身处理失败
                    new_recognition_results.append({
                        'query_image': current_query_base_name,
                        'gallery_image': 'N/A', # 此时无法与任何图库图片关联
                        'is_match': 'Error',
                        'cossim': 'Error'
                    })
                    query_processed_successfully = False # 确保标志为False
                
                # 如果查询图片特征成功提取，则与图库进行比对
                if query_processed_successfully and current_query_feat is not None:
                    try:
                        # 扩展查询特征的维度以匹配所有图库特征，进行批量余弦相似度计算
                        query_feat_expanded = current_query_feat.expand(all_gallery_feats.shape[0], -1)
                        cossims = torch.nn.functional.cosine_similarity(query_feat_expanded, all_gallery_feats, dim=1).tolist()

                        for j, cossim in enumerate(cossims):
                            is_match = cossim > args.threshold
                            gallery_relative_path = all_gallery_relative_paths[j]
                            new_recognition_results.append({
                                'query_image': current_query_base_name,
                                'gallery_image': gallery_relative_path,
                                'is_match': is_match,
                                'cossim': cossim
                            })
                        
                        del current_query_feat, query_feat_expanded
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"比较查询图片 {query_img_path} 与图库时出错: {e}。")
                        # 添加错误结果条目，表示比对过程失败
                        new_recognition_results.append({
                            'query_image': current_query_base_name,
                            'gallery_image': 'N/A_Comparison_Error',
                            'is_match': 'Error',
                            'cossim': 'Error'
                        })
                
                # 无论处理结果如何（成功、失败、未匹配），都将查询图片移动到 dealed_dir
                relative_path_to_move = os.path.relpath(query_img_path, args.input_image_dir)
                try:
                    destination_path = os.path.join(dealed_dir, relative_path_to_move)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    shutil.move(query_img_path, destination_path)
                    print(f"已将 {current_query_base_name} 移动到 {os.path.basename(dealed_dir)}。")
                except Exception as move_e:
                    print(f"移动文件 {query_img_path} 时出错: {move_e}")

            # 处理当前扫描周期内收集到的所有结果
            if new_recognition_results:
                df_all_results = pd.DataFrame(new_recognition_results) # 包含所有结果的DataFrame

                # 筛选出匹配的结果写入CSV
                df_matched_results = df_all_results[df_all_results['is_match'] == True]

                if not df_matched_results.empty:
                    if os.path.exists(args.output_csv):
                        df_matched_results.to_csv(args.output_csv, mode='a', header=False, index=False)
                    else:
                        df_matched_results.to_csv(args.output_csv, index=False)
                    print(f"\n匹配结果已追加到 {args.output_csv}")
                else:
                    print("本轮没有匹配成功的结果需要写入CSV。")

                print("\n本轮所有识别结果:") # 打印所有结果到控制台
                print(df_all_results)
            else:
                print("本轮没有新的识别结果需要处理。")

        time.sleep(args.scan_interval)