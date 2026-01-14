# 小红书API使用说明

## 问题诊断

出现 **404错误** 的原因：
1. **API端点已变更**：小红书的API端点经常更新
2. **需要认证**：小红书API需要登录Cookie或签名验证
3. **反爬机制**：小红书有严格的反爬虫保护

## 解决方案

### 方案1：获取真实的API端点（推荐）

1. **打开浏览器开发者工具**
   - 访问 https://www.xiaohongshu.com
   - 按 F12 打开开发者工具
   - 切换到 "Network"（网络）标签

2. **搜索关键词**
   - 在小红书网站搜索"职场沟通"
   - 在Network标签中查看请求

3. **找到搜索API请求**
   - 查找类型为 `xhr` 或 `fetch` 的请求
   - 查看请求URL，类似：
     ```
     https://edith.xiaohongshu.com/api/sns/web/v1/search/notes
     ```

4. **复制请求信息**
   - 复制完整的请求URL
   - 复制请求头（Headers），特别是：
     - `Cookie`
     - `x-sign`
     - `x-t`
     - `x-s`
     - `t2`
     - 其他认证相关的header

5. **更新代码**
   - 将真实的API端点更新到 `get_rb_data.py`
   - 将Cookie和认证信息添加到headers中

### 方案2：使用Selenium（更稳定但较慢）

如果API端点无法获取，可以使用Selenium模拟浏览器：

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 使用Selenium访问小红书
driver = webdriver.Chrome()
driver.get("https://www.xiaohongshu.com")
# ... 模拟搜索和获取数据
```

### 方案3：使用官方API（最佳方案）

联系小红书开放平台，申请官方API访问权限。

## 快速修复步骤

1. **获取Cookie**：
   - 登录小红书网站
   - 在浏览器开发者工具中复制Cookie值
   - 更新到代码中的 `self.headers['Cookie']`

2. **更新API端点**：
   - 根据实际请求更新 `base_url` 和 `endpoint`

3. **添加签名验证**（如果需要）：
   - 小红书可能需要 `x-sign`、`x-t`、`x-s` 等签名
   - 这些通常由前端JavaScript生成

## 注意事项

⚠️ **重要提示**：
- 小红书有严格的反爬虫机制
- 频繁请求可能导致IP被封
- 建议使用官方API或联系小红书获取授权
- 仅用于学习和研究目的，遵守相关法律法规

