"""
小红书爬虫使用示例 - 爬取"职场沟通"话题
参数：keyword="职场沟通", max_pages=10, output_format="json"
"""
from get_rb_data import crawl_topic_notes

if __name__ == "__main__":
    # 爬取"职场沟通"话题下的笔记数据
    # 参数说明：
    # - topic_keyword: 要搜索的话题关键词
    # - max_pages: 要爬取的页数（每页约20条笔记）
    # - output_format: 输出格式，可选 'json' 或 'csv'
    # - cookie: 可选的Cookie（如果需要，从浏览器开发者工具中获取）
    
    # 如果需要使用Cookie，取消下面的注释并填入你的Cookie
    # cookie = "your_cookie_here"
    cookie = None
    
    crawl_topic_notes(
        topic_keyword="职场沟通",
        max_pages=10,
        output_format="json",
        cookie=cookie
    )

