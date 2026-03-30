"""Image upload encoding and image-derived memory merge."""
import base64
from datetime import datetime

from .auth_memory import deduplicate_items, load_memories, save_memories
from .structured_profile import (
    format_profile_display,
    integrate_all_memories_to_profile,
    merge_structured_profile,
)

# st.rerun() 后「已处理文件」分支不再执行，用 session 保存解析结果以便持续展示
LAST_IMAGE_MEMORY_PARSE_KEY = "last_image_memory_parse"


def store_image_memory_parse_result(session_state, image_info: dict, filename: str) -> None:
    """在即将 st.rerun 前调用，保存本次图片解析结果供下一屏展示。"""
    had_error = bool(image_info.get("error"))
    session_state[LAST_IMAGE_MEMORY_PARSE_KEY] = {
        "filename": filename,
        "title": (image_info.get("title") or "").strip(),
        "description": (image_info.get("description") or "").strip(),
        "events": list(image_info.get("events") or []),
        "profile": list(image_info.get("profile") or []),
        "structured_profile": dict(image_info.get("structured_profile") or {}),
        "had_error": had_error,
        "error_detail": (image_info.get("error_detail") or "").strip(),
    }


def render_last_image_memory_parse(clear_button_key: str) -> None:
    """展示最近一次图片解析结果（若存在）；点击「知道了」后清除。"""
    import streamlit as st

    data = st.session_state.get(LAST_IMAGE_MEMORY_PARSE_KEY)
    if not data:
        return

    with st.expander("🖼️ 图片记忆解析结果（已写入记忆）", expanded=True):
        st.caption(f"文件：`{data.get('filename', '')}`")
        if data.get("had_error"):
            st.warning(
                "多模态识别未完全成功，已尽可能写入记忆。请确认已配置豆包 Vision 或 DeepSeek Vision。"
            )
            desc = data.get("description") or ""
            if desc:
                st.write(desc)
            err_d = data.get("error_detail") or ""
            if err_d:
                with st.expander("错误详情", expanded=False):
                    st.code(err_d[:4000] + ("…" if len(err_d) > 4000 else ""))

        st.markdown("##### 标题")
        st.write(data.get("title") or "—")

        st.markdown("##### 描述 / 解读")
        st.write(data.get("description") or "—")

        events = data.get("events") or []
        profiles = data.get("profile") or []
        if events:
            st.markdown("##### 提取的事件")
            for e in events:
                st.write(f"• {e}")
        if profiles:
            st.markdown("##### 提取的画像")
            for p in profiles:
                st.write(f"• {p}")

        sp = data.get("structured_profile") or {}
        formatted = format_profile_display(sp)
        if formatted:
            st.markdown("##### 结构化画像")
            for line in formatted.split("\n"):
                if line.strip():
                    st.markdown(line)
        elif sp:
            st.markdown("##### 结构化画像（JSON）")
            st.json(sp)

        if not data.get("had_error") and not events and not profiles and not sp:
            st.info("本次解析未返回结构化字段，内容已写入「描述」或记忆摘要。")

        if st.button("知道了", key=clear_button_key, help="关闭本条展示"):
            st.session_state[LAST_IMAGE_MEMORY_PARSE_KEY] = None
            st.rerun()

def process_uploaded_image(uploaded_file):
    """处理上传的图片，转换为base64编码"""
    try:
        # 读取图片文件
        image_bytes = uploaded_file.getvalue()
        
        # 转换为base64编码
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 获取图片格式
        image_format = uploaded_file.type.split('/')[-1]  # 例如: 'png', 'jpeg'
        
        # 构建data URL
        data_url = f"data:image/{image_format};base64,{image_base64}"
        
        return data_url, None
    except Exception as e:
        return None, f"图片处理失败: {str(e)}"

# analyze_image_with_vision 函数已移至 client.py 模块

# 将图片信息添加到用户记忆中
def update_image_memory(image_info, username, filename):
    """将图片信息添加到用户记忆中"""
    memories = load_memories(username)
    
    # 添加图片记录
    image_timestamp = datetime.now().isoformat()
    image_record = {
        "timestamp": image_timestamp,
        "filename": filename,
        "title": image_info.get("title", ""),
        "description": image_info.get("description", ""),
        "events": image_info.get("events", []),
        "profile": image_info.get("profile", []),
        "structured_profile": image_info.get("structured_profile", {}),  # 添加结构化画像
        "type": "image"
    }
    
    # 将图片记录添加到documents列表（统一管理）
    if "documents" not in memories:
        memories["documents"] = []
    memories["documents"].append(image_record)
    
    # 限制记录数量
    if len(memories["documents"]) > 10:
        memories["documents"] = memories["documents"][-10:]
    
    # 合并提取的事件和画像到主记忆（使用去重函数）
    new_events = image_info.get("events", [])
    new_profile = image_info.get("profile", [])
    new_structured_profile = image_info.get("structured_profile", {})
    
    if new_events:
        memories["events"] = deduplicate_items(memories["events"], new_events)
    if new_profile:
        memories["profile"] = deduplicate_items(memories["profile"], new_profile)
    
    # 合并结构化画像信息（传递时间戳用于历史记录）
    if new_structured_profile:
        existing_structured_profile = memories.get("structured_profile", {})
        memories["structured_profile"] = merge_structured_profile(
            existing_structured_profile, 
            new_structured_profile, 
            timestamp=image_timestamp,
            memories=memories  # 传入memories以整合所有历史画像信息
        )
    # 无论是否有新的结构化画像，都整合所有记忆（因为可能有新的旧格式画像或其他信息）
    try:
        memories = integrate_all_memories_to_profile(username, memories=memories)
    except Exception as e:
        print(f"警告：图片记忆更新后整合记忆时出错: {str(e)}")
    
    # 更新记忆摘要
    if memories["summary"] == "这是一位新用户，尚未形成长期记忆。":
        memories["summary"] = f"用户上传了图片：{image_info.get('title', '')}"
    else:
        existing_summary = memories["summary"]
        new_title = image_info.get('title', '')
        
        # 检查是否已经存在相同的图片记忆
        if "【图片记忆】" in existing_summary:
            existing_image_memories = []
            lines = existing_summary.split('\n')
            for line in lines:
                if line.startswith('【图片记忆】'):
                    existing_image_memories.append(line.replace('【图片记忆】', '').strip())
            
            if new_title not in existing_image_memories:
                memories["summary"] += f"\n【图片记忆】{new_title}"
        else:
            memories["summary"] += f"\n【图片记忆】{new_title}"
    
    memories["last_updated"] = datetime.now().isoformat()
    
    # 保存更新
    save_memories(memories, username)
    return memories

