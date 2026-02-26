"""
测试DuckDuckGo搜索功能
用于验证duckduckgo_search库的效果
"""
import sys
import os

# 设置控制台编码（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 尝试导入联网搜索工具
try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
    print("[OK] duckduckgo_search 导入成功")
except ImportError as e:
    HAS_DUCKDUCKGO = False
    print(f"[ERROR] duckduckgo_search 导入失败: {e}")
    print("请安装: pip install duckduckgo-search")
    sys.exit(1)


def test_search(query: str, max_results: int = 5):
    """
    测试搜索功能
    
    Args:
        query: 搜索查询
        max_results: 最大返回结果数
    """
    print(f"\n{'='*60}")
    print(f"搜索查询: {query}")
    print(f"最大结果数: {max_results}")
    print(f"{'='*60}\n")
    
    if not HAS_DUCKDUCKGO:
        print("[ERROR] DuckDuckGo搜索不可用")
        return
    
    try:
        with DDGS() as ddgs:
            print("正在搜索...")
            results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                print("[WARNING] 未找到搜索结果")
                return
            
            print(f"[OK] 找到 {len(results)} 条搜索结果\n")
            
            # 显示搜索结果
            for i, result in enumerate(results, 1):
                title = result.get('title', '无标题')
                body = result.get('body', '无内容')
                href = result.get('href', '无链接')
                
                print(f"【结果 {i}】")
                print(f"标题: {title}")
                print(f"链接: {href}")
                print(f"内容: {body[:200]}..." if len(body) > 200 else f"内容: {body}")
                print("-" * 60)
            
            # 返回结果列表格式（用于验证）
            results_list = []
            for result in results:
                results_list.append({
                    'title': result.get('title', '无标题'),
                    'href': result.get('href', '无链接'),
                    'body': result.get('body', '无内容')
                })
            
            return results_list
            
    except Exception as e:
        print(f"[ERROR] 搜索出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_food_search():
    """测试美食相关搜索"""
    print("\n" + "="*60)
    print("测试1: 美食相关搜索")
    print("="*60)
    
    test_cases = [
        "宫保鸡丁 做法",
        "北京美食推荐",
        "如何做红烧肉",
        "健康食谱",
        "意大利菜"
    ]
    
    for query in test_cases:
        print(f"\n测试查询: {query}")
        results = test_search(query, max_results=3)
        if results:
            print(f"[OK] 成功获取 {len(results)} 条结果")
        else:
            print("[FAIL] 搜索失败")
        print()


def test_travel_search():
    """测试旅行相关搜索"""
    print("\n" + "="*60)
    print("测试2: 旅行相关搜索")
    print("="*60)
    
    test_cases = [
        "北京 旅行攻略",
        "去日本旅游",
        "到上海旅游",
        "欧洲旅行推荐",
        "泰国自由行"
    ]
    
    for query in test_cases:
        print(f"\n测试查询: {query}")
        results = test_search(query, max_results=3)
        if results:
            print(f"[OK] 成功获取 {len(results)} 条结果")
        else:
            print("[FAIL] 搜索失败")
        print()


def test_general_search():
    """测试通用搜索"""
    print("\n" + "="*60)
    print("测试3: 通用搜索")
    print("="*60)
    
    test_cases = [
        "Python编程",
        "人工智能",
        "天气查询"
    ]
    
    for query in test_cases:
        print(f"\n测试查询: {query}")
        results = test_search(query, max_results=3)
        if results:
            print(f"[OK] 成功获取 {len(results)} 条结果")
        else:
            print("[FAIL] 搜索失败")
        print()


def interactive_test():
    """交互式测试"""
    print("\n" + "="*60)
    print("交互式测试模式")
    print("="*60)
    print("输入搜索查询（输入 'quit' 退出）")
    
    while True:
        query = input("\n请输入搜索查询: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("退出测试")
            break
        
        if not query:
            print("查询不能为空")
            continue
        
        max_results = input("最大结果数 (默认5): ").strip()
        try:
            max_results = int(max_results) if max_results else 5
        except ValueError:
            max_results = 5
        
        results = test_search(query, max_results=max_results)
        
        if results:
            print(f"\n[OK] 搜索成功，共 {len(results)} 条结果")
            # 显示链接列表
            print("\n链接列表:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   {result['href']}")
        else:
            print("\n[FAIL] 搜索失败或未找到结果")


def main():
    """主函数"""
    print("="*60)
    print("DuckDuckGo 搜索功能测试")
    print("="*60)
    
    if not HAS_DUCKDUCKGO:
        print("\n[ERROR] 无法进行测试，请先安装 duckduckgo-search")
        print("安装命令: pip install duckduckgo-search")
        return
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\n选择测试模式:")
        print("1. 美食相关搜索测试")
        print("2. 旅行相关搜索测试")
        print("3. 通用搜索测试")
        print("4. 交互式测试")
        print("5. 全部测试")
        print("\n提示: 也可以使用命令行参数，如: python test_duckduckgo_search.py 1")
        
        try:
            choice = input("\n请选择 (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            # 非交互式环境，执行快速测试
            print("\n非交互式环境，执行快速测试...")
            choice = '1'
    
    if choice == '1':
        test_food_search()
    elif choice == '2':
        test_travel_search()
    elif choice == '3':
        test_general_search()
    elif choice == '4':
        interactive_test()
    elif choice == '5':
        test_food_search()
        test_travel_search()
        test_general_search()
    else:
        print("无效选择，执行全部测试")
        test_food_search()
        test_travel_search()
        test_general_search()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
