"""
简单的测试脚本 - 不需要 Jupyter
逐步检查 pipeline 的各个组件
"""
import sys
import os
import traceback
from pathlib import Path

print("=" * 60)
print("PDF Pipeline 诊断测试")
print("=" * 60)

# 1. 检查基础环境
print("\n[1] 检查基础环境...")
print(f"  Python version: {sys.version}")
print(f"  Working directory: {os.getcwd()}")
print(f"  Platform: {sys.platform}")

# 2. 测试模块导入
print("\n[2] 测试模块导入...")
modules_to_test = [
    'config',
    'utils',
    'page_render',
    'layout_detect',
    'region_refiner',
    'image_extraction',
    'image_ocr',
    'native_text',
    'scanned_page',
    'caption_extract',
    'pipeline'
]

failed_imports = []
for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
    except Exception as e:
        print(f"  ✗ {module_name}: {e}")
        failed_imports.append(module_name)
        traceback.print_exc()

if failed_imports:
    print(f"\n  警告: {len(failed_imports)} 个模块导入失败")
else:
    print(f"\n  ✓ 所有模块导入成功")

# 3. 测试依赖包
print("\n[3] 测试依赖包...")
dependencies = {
    'PIL': 'Pillow',
    'layoutparser': 'layoutparser',
    'fitz': 'pymupdf',
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'shapely': 'shapely',
    'torch': 'torch'
}

missing_deps = []
for import_name, package_name in dependencies.items():
    try:
        __import__(import_name)
        print(f"  ✓ {package_name} ({import_name})")
    except ImportError as e:
        print(f"  ✗ {package_name} ({import_name}): {e}")
        missing_deps.append(package_name)

# 测试 detectron2
print("\n  测试 detectron2...")
try:
    import detectron2
    print(f"  ✓ detectron2")
except ImportError:
    print(f"  ⚠ detectron2 not found (layout detection may fail)")
    missing_deps.append('detectron2 (optional)')

if missing_deps:
    print(f"\n  警告: 缺少 {len(missing_deps)} 个依赖包")
    print(f"  请运行: pip install {' '.join([d for d in missing_deps if not d.startswith('detectron2')])}")
else:
    print(f"\n  ✓ 所有依赖包可用")

# 4. 检查配置文件
print("\n[4] 检查配置文件...")
try:
    import config
    print(f"  OUTPUT_DIR: {config.OUTPUT_DIR}")
    print(f"  IMAGE_DIR: {config.IMAGE_DIR}")
    print(f"  RENDER_DIR: {config.RENDER_DIR}")
    print(f"  OCR_LANG: {config.OCR_LANG}")
except Exception as e:
    print(f"  ✗ 配置文件错误: {e}")

# 5. 查找 PDF 文件
print("\n[5] 查找 PDF 文件...")
pdf_path = None

# 首先尝试命令行参数
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
    if os.path.exists(pdf_path):
        print(f"  ✓ 使用命令行参数: {pdf_path}")
    else:
        print(f"  ✗ 命令行指定的文件不存在: {pdf_path}")
        pdf_path = None

# 如果没有，尝试默认路径
if not pdf_path:
    default_path = r"C:\Users\Wangy\Downloads\Sonata\ocr\Copy of 241- 20 - DP ARCHITECTURE -  920 49TH AVE SW - APRIL 13, 2021 (1).pdf"
    if os.path.exists(default_path):
        pdf_path = default_path
        print(f"  ✓ 找到默认 PDF: {pdf_path}")
    else:
        # 查找当前目录下的 PDF
        pdf_files = list(Path('.').glob('*.pdf'))
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"  ✓ 找到当前目录的 PDF: {pdf_path}")
        else:
            print(f"  ✗ 未找到 PDF 文件")
            print(f"    请提供 PDF 路径作为参数: python simple_test.py <pdf_path>")

if pdf_path:
    print(f"  文件大小: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB")

# 6. 测试 Step 1: Page Render
if pdf_path and not failed_imports:
    print("\n[6] 测试 Step 1: PDF 渲染...")
    try:
        from page_render import render_pdf_pages
        import config
        
        page_info = render_pdf_pages(pdf_path, config.RENDER_DIR)
        print(f"  ✓ 成功渲染 {len(page_info)} 页")
        for idx, (img_path, width, height) in enumerate(page_info[:3]):
            print(f"    页 {idx+1}: {width}x{height}")
        if len(page_info) > 3:
            print(f"    ... 还有 {len(page_info)-3} 页")
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()
        page_info = None
else:
    page_info = None
    if not pdf_path:
        print("\n[6] 跳过: 未找到 PDF 文件")
    else:
        print("\n[6] 跳过: 模块导入失败")

# 7. 测试 Step 2: Layout Detection
if page_info:
    print("\n[7] 测试 Step 2: Layout 检测...")
    try:
        from layout_detect import detect_layout, filter_figure_blocks
        
        test_image_path = page_info[0][0]
        layout_blocks = detect_layout(test_image_path)
        print(f"  ✓ 找到 {len(layout_blocks)} 个 layout blocks")
        
        figure_blocks = filter_figure_blocks(layout_blocks)
        print(f"  ✓ 找到 {len(figure_blocks)} 个 figure 候选")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()

# 8. 测试 Native Text Extraction
if page_info:
    print("\n[8] 测试 Native Text 提取...")
    try:
        from native_text import extract_native_text
        
        test_image_path, width_px, height_px = page_info[0]
        native_text_blocks, has_native_text = extract_native_text(
            pdf_path, 0, width_px, height_px
        )
        print(f"  ✓ 提取了 {len(native_text_blocks)} 个文本块")
        print(f"  Has native text: {has_native_text}")
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        traceback.print_exc()

# 9. 测试完整 Pipeline
if pdf_path and not failed_imports:
    print("\n[9] 测试完整 Pipeline...")
    print("=" * 60)
    try:
        from pipeline import process_pdf, save_result
        
        result = process_pdf(pdf_path, output_dir="output")
        
        print("\n" + "=" * 60)
        print("Pipeline 完成!")
        print(f"  总页数: {result['meta']['page_count']}")
        print(f"  总图像: {sum(len(page['images']) for page in result['pages'])}")
        print(f"  总 Caption: {sum(len(page['captions']) for page in result['pages'])}")
        print(f"  文本长度: {len(result['text_content'])} 字符")
        
        save_result(result, "output/result.json")
        print(f"\n  结果已保存到: output/result.json")
        
    except Exception as e:
        print(f"\n  ✗ Pipeline 失败: {e}")
        traceback.print_exc()
else:
    if not pdf_path:
        print("\n[9] 跳过: 未找到 PDF 文件")
    else:
        print("\n[9] 跳过: 模块导入失败")

# 总结
print("\n" + "=" * 60)
print("诊断完成!")
print("=" * 60)

if failed_imports:
    print(f"\n需要修复的问题:")
    print(f"  - {len(failed_imports)} 个模块导入失败")
    
if missing_deps:
    print(f"  - {len(missing_deps)} 个依赖包缺失")

if not pdf_path:
    print(f"  - 未找到 PDF 文件")

if not failed_imports and pdf_path:
    print("\n所有检查通过! Pipeline 应该可以正常工作。")
