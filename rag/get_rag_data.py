import os
from PyPDF2 import PdfReader, PdfWriter


bookmarks = {
    "1": {
        "title": "第一章",
        "pages": [1, 56]
    },
    "2": {
        "title": "第二章",
        "pages": [57, 71]
    },
    "3": {
        "title": "第三章",
        "pages": [72, 99]
    },
    "4": {
        "title": "第四章",
        "pages": [100, 105]
    },
    "5": {
        "title": "第五章",
        "pages": [106, 116]
    },
    "6": {
        "title": "第六章",
        "pages": [117, 135]
    },
    "7": {
        "title": "第七章",
        "pages": [136, 158]
    },
    "8": {
        "title": "第八章",
        "pages": [159, 180]
    },
    "9": {
        "title": "第九章",
        "pages": [181, 205]
    },
    "10": {
        "title": "第十章",
        "pages": [206, 224]
    },
    "11": {
        "title": "第十一章",
        "pages": [225, 245]
    },
    "12": {
        "title": "第十二章",
        "pages": [246, 256]
    },
    "13": {
        "title": "第十三章",
        "pages": [257, 280]
    },
    "14": {
        "title": "第十四章",
        "pages": [281, 295]
    },
    "15": {
        "title": "第十五章",
        "pages": [296, 318]
    },
    "16": {
        "title": "第十六章",
        "pages": [319, 338]
    },
    "17": {
        "title": "第十七章",
        "pages": [339, 351]
    },
    "18": {
        "title": "第十八章",
        "pages": [352, 366]
    },
    "19": {
        "title": "第十九章",
        "pages": [367, 376]
    },
    "20": {
        "title": "第二十章",
        "pages": [377, 388]
    },
    "21": {
        "title": "第二十一章",
        "pages": [389, 401]
    },
    "22": {
        "title": "第二十二章",
        "pages": [402, 421]
    },
    "23": {
        "title": "第二十三章",
        "pages": [422, 433]
    },
    "24": {
        "title": "第二十四章",
        "pages": [434, 449]
    },
    "25": {
        "title": "第二十五章",
        "pages": [450, 461]
    },
    "26": {
        "title": "第二十六章",
        "pages": [462, 470]
    }
}

# 按给定的章节页码分割PDF文件
def split_pdf_by_pages(pdf_path, output_dir='output'):
    """
    按给定的章节页码分割PDF文件
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"文档总页数: {total_pages}")
    
    for chapter, info in bookmarks.items():
        # 从字典中获取页码列表
        pages = info["pages"]
        title = info["title"]
        
        # 页码是从1开始的，需要转换为从0开始的索引
        start_page = pages[0] - 1  # PDF索引从0开始
        end_page = pages[1]  # 结束页（不包含），所以不需要减1
        
        # 检查页面范围
        if start_page < 0 or end_page > total_pages:
            print(f"警告: 章节 '{title}' 的页面范围超出文档范围，跳过")
            continue
        
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        
        # 清理文件名
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        output_path = os.path.join(output_dir, f"{chapter.zfill(2)}_{clean_title}.pdf")
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"已保存: {output_path} (页面 {pages[0]}-{pages[1]}, 共 {end_page - start_page} 页)")

def split_pdf_by_bookmarks(pdf_path, output_dir='output'):
    """
    根据PDF书签（目录）分割PDF文件
    适合有明确书签结构的PDF
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取PDF
    reader = PdfReader(pdf_path)
    
    # 获取文档信息
    print(f"文档总页数: {len(reader.pages)}")
    
    if reader.outline:
        print("找到书签/目录结构")
        # 获取书签（目录）信息
        outlines = reader.outline


        # 将书签转换为扁平列表，只保留一级书签（主要章节）
        def flatten_outlines(outlines, level=0, max_level=1):
            flat = []
            for item in outlines:
                if isinstance(item, list):
                    # 如果是子书签列表，递归处理
                    if level < max_level:
                        flat.extend(flatten_outlines(item, level+1, max_level))
                else:
                    # 如果是书签对象
                    try:
                        page_num = reader.get_destination_page_number(item)
                        flat.append((item.title, page_num, level))
                    except Exception as e:
                        print(f"警告: 无法获取书签 '{item.title}' 的页面号: {e}")
            return flat
        
        # 先获取所有书签（包括层级信息）
        all_bookmarks = flatten_outlines(outlines, max_level=10)
        print(f"找到 {len(all_bookmarks)} 个书签")
        print("前10个书签示例:", all_bookmarks[:10])
        
        # 只使用一级书签（level=0）作为主要章节
        main_bookmarks = [(title, page_num) for title, page_num, level in all_bookmarks if level == 0]
        
        # 如果没有一级书签，使用所有书签，但过滤掉页面号相同或非常接近的书签
        if not main_bookmarks:
            print("未找到一级书签，使用所有书签，但会合并相邻的书签")
            # 过滤：如果两个书签的页面号相同或只差1页，只保留第一个
            filtered_bookmarks = []
            last_page = -1
            for title, page_num, level in all_bookmarks:
                if page_num > last_page + 1:  # 至少间隔2页才认为是新章节
                    filtered_bookmarks.append((title, page_num))
                    last_page = page_num
            main_bookmarks = filtered_bookmarks
        
        bookmarks = main_bookmarks
        print(f"将使用 {len(bookmarks)} 个主要章节进行分割")
        print("章节列表:", bookmarks[:20])  # 显示前20个
        
        # 按书签分割
        for i, (title, page_num) in enumerate(bookmarks):
            writer = PdfWriter()
            
            # 确定当前章节的结束页
            start_page = page_num
            if i < len(bookmarks) - 1:
                end_page = bookmarks[i+1][1] - 1
            else:
                end_page = len(reader.pages) - 1
            
            # 检查页面范围是否有效
            if start_page > end_page:
                print(f"警告: 章节 '{title}' 的起始页({start_page})大于结束页({end_page})，跳过")
                continue
            
            if start_page >= len(reader.pages) or end_page >= len(reader.pages):
                print(f"警告: 章节 '{title}' 的页面范围超出文档范围，跳过")
                continue
            
            print(f"处理章节 {i+1}: '{title}' - 页面 {start_page+1} 到 {end_page+1} (共 {end_page - start_page + 1} 页)")
            
            # 添加页面
            for page in range(start_page, end_page + 1):
                writer.add_page(reader.pages[page])
            
            # 清理文件名（移除非法字符）
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            if not clean_title:
                clean_title = f"chapter_{i+1}"
            
            output_path = os.path.join(output_dir, f"{i+1:02d}_{clean_title}.pdf")
            
            # 写入文件
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            #print(f"  ✓ 已保存: {output_path} ({end_page - start_page + 1}页)\n")
    else:
        print("未找到书签，将按固定页数分割")
        # 如果没有书签，按固定页数分割
        split_by_fixed_pages(pdf_path, output_dir, pages_per_chapter=20)

def split_by_fixed_pages(pdf_path, output_dir='output', pages_per_chapter=20):
    """
    按固定页数分割PDF
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    chapter_num = 1
    for start in range(0, total_pages, pages_per_chapter):
        end = min(start + pages_per_chapter, total_pages)
        
        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        
        output_path = os.path.join(output_dir, f"chapter_{chapter_num:02d}_pages_{start+1}-{end}.pdf")
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"已保存: {output_path} ({end-start}页)")
        chapter_num += 1

# 使用示例
if __name__ == "__main__":
    pdf_path = r"D:\Users\Documents\project\临床营养学_第3版.pdf"  # 你的PDF文件路径
    #split_pdf_by_bookmarks(pdf_path, "split_chapters")
    split_pdf_by_pages(pdf_path, "split_chapters")