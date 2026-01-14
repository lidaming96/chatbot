# 从小红书爬取指定话题下的笔记数据
"""
小红书话题笔记爬虫
注意：请遵守小红书的使用条款和robots.txt，仅用于学习和研究目的
建议：使用官方API或联系小红书获取数据授权
"""
import requests
import json
import time
import random
import re
import os
import sys
import urllib.parse
from datetime import datetime
from typing import List, Dict, Optional
import csv


class XiaohongshuCrawler:
    """小红书话题笔记爬虫类"""
    
    def __init__(self, cookie: str = None):
        """
        初始化爬虫
        
        Args:
            cookie: 可选的Cookie字符串（从浏览器开发者工具中获取）
        """
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.xiaohongshu.com/',
            'Origin': 'https://www.xiaohongshu.com',
        }
        
        # 如果提供了Cookie，添加到headers中
        if cookie:
            self.headers['Cookie'] = cookie
            print("✓ 已设置Cookie")
        
        self.session.headers.update(self.headers)
        # 尝试多个可能的API端点
        self.base_urls = [
            "https://edith.xiaohongshu.com",
            "https://www.xiaohongshu.com",
            "https://api.xiaohongshu.com",
            "https://t2.xiaohongshu.com",
        ]
        self.base_url = self.base_urls[0]  # 默认使用第一个
    
    def set_cookie(self, cookie: str):
        """设置Cookie"""
        self.headers['Cookie'] = cookie
        self.session.headers.update(self.headers)
        print("✓ Cookie已更新")
        
    def search_topic(self, keyword: str, page: int = 1, page_size: int = 20) -> Optional[Dict]:
        """
        搜索话题/关键词
        
        Args:
            keyword: 搜索关键词
            page: 页码
            page_size: 每页数量
            
        Returns:
            搜索结果字典
        """
        # 尝试多个可能的API端点
        # 注意：/api/v2/collect 可能不是搜索API，放在后面尝试
        possible_endpoints = [
            "/api/sns/web/v1/search/notes",
            "/web_api/sns/v3/search/notes",
            "/api/sns/v4/search/notes",
            "/api/sns/web/v1/feed",
            "/api/v2/collect",  # 这个可能不是搜索API
            "/api/data",
        ]
        
        for base_url in self.base_urls:
            for endpoint in possible_endpoints:
                try:
                    search_url = f"{base_url}{endpoint}"
                    
                    # 尝试不同的参数格式
                    params_variants = [
                        {
                            'keyword': keyword,
                            'page': page,
                            'page_size': page_size,
                            'search_id': self._generate_search_id(),
                        },
                        {
                            'keyword': keyword,
                            'page': page,
                            'page_size': page_size,
                        },
                        {
                            'key': keyword,
                            'page': page,
                            'page_size': page_size,
                        }
                    ]
                    
                    for params in params_variants:
                        try:
                            response = self.session.get(search_url, params=params, timeout=10)
                            
                            if response.status_code == 200:
                                try:
                                    data = response.json()
                                    # 检查响应是否有效
                                    if data and isinstance(data, dict):
                                        # 检查是否包含笔记数据（根据实际API响应结构调整）
                                        # 实际响应结构：{"code": 0, "success": true, "data": {"items": [...]}}
                                        has_data = False
                                        if 'data' in data and isinstance(data['data'], dict):
                                            # 检查data中是否包含items
                                            if 'items' in data['data']:
                                                has_data = True
                                        elif any(key in data for key in ['items', 'notes', 'result', 'feeds', 'list']):
                                            has_data = True
                                        if has_data:
                                            print(f"✓ 成功连接到API: {base_url}{endpoint}")
                                            # 打印部分响应用于调试
                                            print(f"  响应结构: {list(data.keys())[:5]}")
                                            if 'data' in data and 'items' in data.get('data', {}):
                                                print(f"  找到笔记数据: {len(data['data']['items'])} 条")
                                            return data
                                        else:
                                            # 打印响应结构用于调试
                                            print(f"  响应结构不匹配: {list(data.keys())[:5] if isinstance(data, dict) else type(data)}")
                                            # 继续尝试下一个端点
                                except json.JSONDecodeError:
                                    # 响应不是JSON格式
                                    print(f"  响应不是JSON格式，内容: {response.text[:100]}")
                                    continue
                            elif response.status_code == 404:
                                continue  # 尝试下一个端点
                            else:
                                print(f"  尝试 {base_url}{endpoint}: 状态码 {response.status_code}")
                                
                        except requests.exceptions.RequestException as e:
                            continue
                            
                except Exception as e:
                    continue
        
        # 所有端点都失败
        print(f"\n❌ 所有API端点尝试失败")
        print(f"可能的原因：")
        print(f"1. 小红书API端点已变更，需要更新代码")
        print(f"2. 需要登录认证或Cookie（小红书有反爬机制）")
        print(f"3. IP被限制或需要验证码")
        print(f"4. 建议使用官方API或联系小红书获取数据授权")
        print(f"\n提示：可以尝试以下方法：")
        print(f"- 使用浏览器开发者工具查看实际请求的API端点")
        print(f"- 添加有效的Cookie到代码中")
        print(f"- 使用Selenium模拟浏览器访问")
        
        return None
    
    def get_topic_notes(self, topic_id: str, page: int = 1, page_size: int = 20) -> Optional[Dict]:
        """
        获取指定话题下的笔记
        
        Args:
            topic_id: 话题ID
            page: 页码
            page_size: 每页数量
            
        Returns:
            笔记列表数据
        """
        try:
            # 话题笔记API（需要根据实际API调整）
            topic_url = f"{self.base_url}/api/sns/web/v1/topic/notes"
            
            params = {
                'topic_id': topic_id,
                'page': page,
                'page_size': page_size,
            }
            
            response = self.session.get(topic_url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取话题笔记失败，状态码: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"获取话题笔记出错: {str(e)}")
            return None
    
    def get_note_detail(self, note_id: str) -> Optional[Dict]:
        """
        获取笔记详情
        
        Args:
            note_id: 笔记ID
            
        Returns:
            笔记详情数据
        """
        try:
            note_url = f"{self.base_url}/api/sns/web/v1/feed"
            
            params = {
                'source_note_id': note_id,
            }
            
            response = self.session.get(note_url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"获取笔记详情失败，状态码: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"获取笔记详情出错: {str(e)}")
            return None
    
    def parse_note_data(self, note_data: Dict) -> Dict:
        """
        解析笔记数据，提取关键信息
        
        Args:
            note_data: 原始笔记数据（可能是item或note_card）
            
        Returns:
            解析后的笔记信息字典
        """
        try:
            # 根据实际API响应结构，数据可能在note_card中
            if 'note_card' in note_data:
                note_card = note_data['note_card']
                note_id = note_data.get('id', note_card.get('note_id', ''))
            else:
                note_card = note_data
                note_id = note_data.get('id', note_data.get('note_id', ''))
            
            # 提取用户信息
            user_info = note_card.get('user', {})
            
            # 提取互动信息
            interact_info = note_card.get('interact_info', {})
            
            # 提取图片列表
            image_list = note_card.get('image_list', [])
            images = []
            for img in image_list:
                # 优先使用url_default，如果没有则使用info_list中的url
                img_url = img.get('url_default', '')
                if not img_url and img.get('info_list'):
                    for info in img['info_list']:
                        if info.get('image_scene') == 'WB_DFT':
                            img_url = info.get('url', '')
                            break
                if img_url:
                    images.append(img_url)
            
            # 提取标签列表
            tag_list = note_card.get('tag_list', [])
            tags = [tag.get('name', '') for tag in tag_list if tag.get('name')]
            
            # 构建笔记信息
            note_info = {
                'note_id': note_id,
                'title': note_card.get('title', ''),
                'desc': note_card.get('desc', ''),
                'type': note_card.get('type', ''),
                'user': {
                    'user_id': user_info.get('user_id', ''),
                    'nickname': user_info.get('nickname', ''),
                    'avatar': user_info.get('avatar', ''),
                },
                'interact_info': {
                    'liked_count': int(interact_info.get('liked_count', 0)) if str(interact_info.get('liked_count', 0)).isdigit() else 0,
                    'collected_count': int(interact_info.get('collected_count', 0)) if str(interact_info.get('collected_count', 0)).isdigit() else 0,
                    'comment_count': int(interact_info.get('comment_count', 0)) if str(interact_info.get('comment_count', 0)).isdigit() else 0,
                    'share_count': int(interact_info.get('share_count', 0)) if str(interact_info.get('share_count', 0)).isdigit() else 0,
                },
                'time': note_card.get('time', ''),
                'last_update_time': note_card.get('last_update_time', ''),
                'images': images,
                'tag_list': tags,
                'cover': images[0] if images else '',  # 使用第一张图片作为封面
            }
            return note_info
        except Exception as e:
            print(f"解析笔记数据出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_to_json(self, data: List[Dict], filename: str):
        """保存数据到JSON文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到: {filename}")
        except Exception as e:
            print(f"保存JSON文件出错: {str(e)}")
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """保存数据到CSV文件"""
        try:
            if not data:
                print("没有数据可保存")
                return
            
            # 获取所有字段名
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            
            # 展开嵌套字典
            flattened_data = []
            for item in data:
                flat_item = {}
                for key, value in item.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_item[f"{key}_{sub_key}"] = sub_value
                    elif isinstance(value, list):
                        flat_item[key] = ', '.join(str(v) for v in value)
                    else:
                        flat_item[key] = value
                flattened_data.append(flat_item)
            
            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(flattened_data[0].keys()) if flattened_data else [])
                writer.writeheader()
                writer.writerows(flattened_data)
            
            print(f"数据已保存到: {filename}")
        except Exception as e:
            print(f"保存CSV文件出错: {str(e)}")
    
    def _generate_search_id(self) -> str:
        """生成搜索ID"""
        return f"{int(time.time() * 1000)}{random.randint(1000, 9999)}"
    
    def _random_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """随机延迟，避免请求过快"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)


def crawl_topic_notes(topic_keyword: str, max_pages: int = 5, output_format: str = 'json', cookie: str = None):
    """
    爬取指定话题下的笔记数据
    
    Args:
        topic_keyword: 话题关键词
        max_pages: 最大爬取页数
        output_format: 输出格式 ('json' 或 'csv')
        cookie: 可选的Cookie字符串（从浏览器开发者工具中获取）
    """
    crawler = XiaohongshuCrawler(cookie=cookie)
    all_notes = []
    
    print(f"开始爬取话题: {topic_keyword}")
    print(f"计划爬取 {max_pages} 页数据")
    
    for page in range(1, max_pages + 1):
        print(f"\n正在爬取第 {page} 页...")
        
        # 搜索话题
        search_result = crawler.search_topic(topic_keyword, page=page)
        
        if not search_result:
            print(f"第 {page} 页获取失败，跳过")
            continue
        
        # 解析搜索结果（根据实际API响应结构调整）
        # 尝试多种可能的数据结构
        notes = []
        
        # 根据实际API响应结构：data.items 包含笔记列表
        # 每个item包含note_card字段
        if 'data' in search_result and 'items' in search_result['data']:
            notes = search_result['data']['items']
            print(f"  找到数据路径: data.items，共 {len(notes)} 条笔记")
        else:
            # 尝试其他可能的数据结构路径
            possible_paths = [
                search_result.get('data', {}).get('items', []),
                search_result.get('data', {}).get('notes', []),
                search_result.get('data', {}).get('result', []),
                search_result.get('data', {}).get('feeds', []),
                search_result.get('items', []),
                search_result.get('notes', []),
                search_result.get('result', []),
                search_result.get('feeds', []),
                search_result.get('data', []),
            ]
            
            for path_data in possible_paths:
                if path_data and isinstance(path_data, list) and len(path_data) > 0:
                    notes = path_data
                    print(f"  找到数据路径，共 {len(notes)} 条笔记")
                    break
        
        if not notes:
            # 打印实际响应结构用于调试
            print(f"第 {page} 页没有更多数据")
            print(f"  响应结构: {list(search_result.keys())[:10] if isinstance(search_result, dict) else type(search_result)}")
            if isinstance(search_result, dict) and 'data' in search_result:
                data_obj = search_result['data']
                if isinstance(data_obj, dict):
                    print(f"  data结构: {list(data_obj.keys())[:10]}")
                else:
                    print(f"  data类型: {type(data_obj)}")
            # 保存响应用于调试
            debug_file = f"debug_response_page_{page}.json"
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(search_result, f, ensure_ascii=False, indent=2)
                print(f"  调试信息已保存到: {debug_file}")
            except:
                pass
            break
        
        # 解析每一条笔记
        for note in notes:
            try:
                note_info = crawler.parse_note_data(note)
                if note_info:
                    all_notes.append(note_info)
                    print(f"  ✓ 获取笔记: {note_info.get('title', '无标题')[:30]}")
            except Exception as e:
                print(f"  ✗ 解析笔记失败: {str(e)}")
        
        # 随机延迟，避免请求过快
        if page < max_pages:
            crawler._random_delay(2.0, 4.0)
    
    # 保存数据
    if all_notes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_keyword = re.sub(r'[^\w\s-]', '', topic_keyword).strip()
        safe_keyword = re.sub(r'[-\s]+', '_', safe_keyword)
        
        if output_format.lower() == 'csv':
            filename = f"xiaohongshu_{safe_keyword}_{timestamp}.csv"
            crawler.save_to_csv(all_notes, filename)
        else:
            filename = f"xiaohongshu_{safe_keyword}_{timestamp}.json"
            crawler.save_to_json(all_notes, filename)
        
        print(f"\n爬取完成！共获取 {len(all_notes)} 条笔记数据")
    else:
        print("\n未获取到任何数据")


def main():
    """主函数"""
    print("=" * 50)
    print("小红书话题笔记爬虫")
    print("=" * 50)
    print("\n⚠️  重要提示：")
    print("1. 本工具仅用于学习和研究目的")
    print("2. 请遵守小红书的使用条款和robots.txt")
    print("3. 建议使用官方API或联系小红书获取数据授权")
    print("4. 请合理控制爬取频率，避免对服务器造成压力")
    print("=" * 50)
    
    # 从命令行参数获取关键词
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
        max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        output_format = sys.argv[3] if len(sys.argv) > 3 else 'json'
    else:
        # 交互式输入
        keyword = input("\n请输入要搜索的话题关键词: ").strip()
        if not keyword:
            print("关键词不能为空！")
            return
        
        try:
            max_pages = int(input("请输入要爬取的页数 (默认5): ").strip() or "5")
        except ValueError:
            max_pages = 5
        
        output_format = input("请输入输出格式 (json/csv，默认json): ").strip().lower() or 'json'
    
    if output_format not in ['json', 'csv']:
        output_format = 'json'
    
    # 开始爬取
    crawl_topic_notes(keyword, max_pages, output_format)


if __name__ == "__main__":
    main()
