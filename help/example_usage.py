"""
小红书爬虫使用示例
演示如何使用 get_rb_data.py 爬取"职场沟通"话题下的笔记数据
"""
from get_rb_data import XiaohongshuCrawler, crawl_topic_notes


def example_1_direct_function():
    """示例1：直接使用函数方式"""
    print("=" * 60)
    print("示例1：使用 crawl_topic_notes 函数")
    print("=" * 60)
    
    # 爬取"职场沟通"话题，10页数据，输出JSON格式
    crawl_topic_notes(
        topic_keyword="职场沟通",
        max_pages=10,
        output_format="json"
    )


def example_2_class_usage():
    """示例2：使用类的方式，更灵活的控制"""
    print("\n" + "=" * 60)
    print("示例2：使用 XiaohongshuCrawler 类")
    print("=" * 60)
    
    # 创建爬虫实例
    crawler = XiaohongshuCrawler()
    
    all_notes = []
    keyword = "职场沟通"
    max_pages = 10
    
    print(f"开始爬取话题: {keyword}")
    print(f"计划爬取 {max_pages} 页数据\n")
    
    for page in range(1, max_pages + 1):
        print(f"正在爬取第 {page}/{max_pages} 页...")
        
        # 搜索话题
        search_result = crawler.search_topic(keyword, page=page, page_size=20)
        
        if not search_result:
            print(f"  第 {page} 页获取失败，跳过")
            continue
        
        # 解析搜索结果（根据实际API响应结构调整）
        # 注意：这里需要根据实际API响应结构调整解析逻辑
        notes = search_result.get('data', {}).get('items', [])
        
        if not notes:
            print(f"  第 {page} 页没有更多数据，停止爬取")
            break
        
        # 解析每一条笔记
        page_count = 0
        for note in notes:
            try:
                note_info = crawler.parse_note_data(note)
                if note_info:
                    all_notes.append(note_info)
                    page_count += 1
                    title = note_info.get('title', '无标题')[:30]
                    print(f"  ✓ [{page_count}] {title}")
            except Exception as e:
                print(f"  ✗ 解析笔记失败: {str(e)}")
        
        print(f"  第 {page} 页完成，获取 {page_count} 条笔记\n")
        
        # 随机延迟，避免请求过快
        if page < max_pages:
            crawler._random_delay(2.0, 4.0)
    
    # 保存数据为JSON格式
    if all_notes:
        from datetime import datetime
        import re
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_keyword = re.sub(r'[^\w\s-]', '', keyword).strip()
        safe_keyword = re.sub(r'[-\s]+', '_', safe_keyword)
        filename = f"xiaohongshu_{safe_keyword}_{timestamp}.json"
        
        crawler.save_to_json(all_notes, filename)
        
        print(f"\n爬取完成！")
        print(f"共获取 {len(all_notes)} 条笔记数据")
        print(f"数据已保存到: {filename}")
        
        # 显示统计信息
        print("\n数据统计：")
        total_likes = sum(note.get('interact_info', {}).get('liked_count', 0) for note in all_notes)
        total_collects = sum(note.get('interact_info', {}).get('collected_count', 0) for note in all_notes)
        total_comments = sum(note.get('interact_info', {}).get('comment_count', 0) for note in all_notes)
        
        print(f"  总点赞数: {total_likes:,}")
        print(f"  总收藏数: {total_collects:,}")
        print(f"  总评论数: {total_comments:,}")
        print(f"  平均点赞数: {total_likes // len(all_notes) if all_notes else 0:,}")
    else:
        print("\n未获取到任何数据")


def example_3_with_filtering():
    """示例3：带过滤条件的爬取（例如只保存点赞数超过100的笔记）"""
    print("\n" + "=" * 60)
    print("示例3：带过滤条件的爬取")
    print("=" * 60)
    
    crawler = XiaohongshuCrawler()
    all_notes = []
    filtered_notes = []
    
    keyword = "职场沟通"
    max_pages = 10
    min_likes = 100  # 最小点赞数阈值
    
    print(f"爬取话题: {keyword}")
    print(f"过滤条件: 点赞数 >= {min_likes}")
    print(f"计划爬取 {max_pages} 页数据\n")
    
    for page in range(1, max_pages + 1):
        print(f"正在爬取第 {page}/{max_pages} 页...")
        
        search_result = crawler.search_topic(keyword, page=page)
        
        if not search_result:
            continue
        
        notes = search_result.get('data', {}).get('items', [])
        
        if not notes:
            break
        
        for note in notes:
            try:
                note_info = crawler.parse_note_data(note)
                if note_info:
                    all_notes.append(note_info)
                    
                    # 过滤：只保留点赞数超过阈值的笔记
                    likes = note_info.get('interact_info', {}).get('liked_count', 0)
                    if likes >= min_likes:
                        filtered_notes.append(note_info)
                        print(f"  ✓ 符合条件: {note_info.get('title', '')[:30]} (点赞: {likes})")
            except Exception as e:
                print(f"  ✗ 解析失败: {str(e)}")
        
        if page < max_pages:
            crawler._random_delay(2.0, 4.0)
    
    # 保存过滤后的数据
    if filtered_notes:
        from datetime import datetime
        import re
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_keyword = re.sub(r'[^\w\s-]', '', keyword).strip()
        safe_keyword = re.sub(r'[-\s]+', '_', safe_keyword)
        filename = f"xiaohongshu_{safe_keyword}_filtered_{timestamp}.json"
        
        crawler.save_to_json(filtered_notes, filename)
        
        print(f"\n过滤完成！")
        print(f"原始数据: {len(all_notes)} 条")
        print(f"过滤后: {len(filtered_notes)} 条 (点赞数 >= {min_likes})")
        print(f"数据已保存到: {filename}")
    else:
        print(f"\n未找到符合条件的笔记（点赞数 >= {min_likes}）")


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "=" * 60)
    print("小红书爬虫使用示例")
    print("=" * 60)
    print("\n本示例将演示三种不同的使用方式：")
    print("1. 直接使用函数（最简单）")
    print("2. 使用类的方式（更灵活）")
    print("3. 带过滤条件的爬取（高级用法）")
    print("\n" + "=" * 60)
    
    # 选择要运行的示例
    choice = input("\n请选择要运行的示例 (1/2/3，默认1): ").strip() or "1"
    
    if choice == "1":
        example_1_direct_function()
    elif choice == "2":
        example_2_class_usage()
    elif choice == "3":
        example_3_with_filtering()
    else:
        print("无效选择，运行示例1")
        example_1_direct_function()


if __name__ == "__main__":
    # 直接运行示例1（使用指定参数）
    print("直接运行示例：爬取'职场沟通'话题，10页，JSON格式")
    print("=" * 60)
    
    crawl_topic_notes(
        topic_keyword="职场沟通",
        max_pages=10,
        output_format="json"
    )
    
    # 如果需要运行其他示例，取消下面的注释
    # main()

