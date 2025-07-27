# -*- coding: utf-8 -*-

import streamlit as st
import requests
import argparse

st.set_page_config(
    page_title="多模式RAG系统浏览器",
    page_icon="🚀",
    layout="wide"
)

# 初始化会话状态变量
st.session_state.setdefault('api_url', '')
st.session_state.setdefault('loaded_modes_info', [])
st.session_state.setdefault('selected_mode', None)
st.session_state.setdefault('query_results', None)
st.session_state.setdefault('doc_store_cache', {})
st.session_state.setdefault('selected_chunk_id', None)


@st.cache_data(ttl=10, show_spinner="正在检查API状态...")
def get_api_status(api_url: str):
    """
    从API服务器获取当前已加载的模式列表。

    通过调用API的 `/status` 端点获取服务器状态信息，包括所有已加载模式的
    详细信息。使用Streamlit的缓存机制，在10秒内避免重复请求。

    Args:
        api_url (str): API服务器的基础URL。

    Returns:
        list: 包含已加载模式信息的列表。如果API请求失败，则返回None。
    """
    try:
        response = requests.get(f"{api_url}/status", timeout=5)
        response.raise_for_status()
        return response.json().get("loaded_modes", [])
    except requests.RequestException:
        return None


@st.cache_data(show_spinner="正在从API加载知识库...")
def get_all_chunks(api_url: str, mode_name: str):
    """
    获取指定模式的完整知识库数据（所有文本块）。

    该函数从API服务器下载指定模式的所有知识块数据，并将其缓存到会话状态中
    以避免在同一会话中重复下载，从而提高性能。

    Args:
        api_url (str): API服务器的基础URL。
        mode_name (str): 需要获取知识库的模式名称。

    Returns:
        dict: 包含知识库数据的字典。如果获取失败，则返回None。
    """
    if mode_name in st.session_state.doc_store_cache:
        return st.session_state.doc_store_cache[mode_name]

    try:
        url = f"{api_url}/modes/get_all_chunks/{mode_name}"
        response = requests.get(url, timeout=180)
        response.raise_for_status()
        doc_store = response.json().get('doc_store', {})
        st.session_state.doc_store_cache[mode_name] = doc_store
        return doc_store
    except requests.RequestException as e:
        st.error(f"获取模式 '{mode_name}' 的知识块失败: {e}")
        return None


def display_context_improved(doc_store: dict, selected_chunk_id: int):
    """
    在界面上显示选定知识块及其周围的上下文信息。

    该函数会定位到选定的知识块，并展示其在原始文档中的前两个和后两个
    相邻块的内容，以帮助用户更好地理解该知识块的语境。

    Args:
        doc_store (dict): 完整的知识库数据字典。
        selected_chunk_id (int): 用户当前选择查看的知识块的整数ID。
    """
    st.subheader("上下文详情")

    if not doc_store or selected_chunk_id is None:
        st.warning("知识库数据不可用或未选择块。")
        return

    selected_chunk_data = doc_store.get(str(selected_chunk_id))
    if not selected_chunk_data:
        st.error(f"错误: 在知识库中找不到块 ID {selected_chunk_id}。")
        return

    selected_doc_id = selected_chunk_data.get('metadata', {}).get('doc_id')

    with st.container(border=True):
        st.markdown(f"**源文件**: `{selected_doc_id}`")

        # 筛选出与选定块来自同一文档的所有兄弟块
        sibling_chunks = [
            chunk for chunk in doc_store.values()
            if chunk.get('metadata', {}).get('doc_id') == selected_doc_id
        ]
        # 按层级和ID排序以确保顺序正确
        sibling_chunks.sort(key=lambda x: (x.get('metadata', {}).get('level', 0), x.get('chunk_id', 0)))

        try:
            current_index = [i for i, c in enumerate(sibling_chunks) if c['chunk_id'] == selected_chunk_id][0]
        except IndexError:
            st.error("错误：无法在文档存储中定位到当前块。")
            return

        # 定义一个窗口来显示前后各2个块
        window = 2
        start_index = max(0, current_index - window)
        end_index = min(len(sibling_chunks), current_index + window + 1)

        # 遍历并显示窗口内的所有块
        for i in range(start_index, end_index):
            chunk = sibling_chunks[i]
            is_selected = (chunk['chunk_id'] == selected_chunk_id)

            with st.container(border=is_selected):
                header_text = f" (整数ID: `{chunk.get('chunk_id', 'N/A')}`)"
                level = chunk.get('metadata', {}).get('level', 'N/A')

                if is_selected:
                    st.markdown(f"**📍 当前块 (层级: {level})**{header_text}")
                    st.markdown(chunk.get('content', '[无内容]'))
                else:
                    st.markdown(f"**上下文块 (层级: {level})**{header_text}")
                    st.write(chunk.get('content', '[无内容]'))

            if i < end_index - 1:
                st.markdown("---")


def main(api_url: str):
    """
    Streamlit Web应用的主函数。

    该函数负责构建整个Web用户界面，包括侧边栏的模式选择和查询控件，
    以及主区域的结果展示面板。它管理着应用的状态和用户交互流程。

    Args:
        api_url (str): 后端RAG API服务器的基础URL。
    """
    st.session_state.api_url = api_url
    st.title("多模式 RAG 系统浏览器")
    st.caption(f"已连接到 API: `{api_url}`")

    # 获取并更新已加载的模式列表
    st.session_state.loaded_modes_info = get_api_status(api_url)

    if st.session_state.loaded_modes_info is None:
        st.error("无法连接到 RAG API 服务器。请确保服务器正在运行。")
        st.stop()

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("模式选择与查询")

        if not st.session_state.loaded_modes_info:
            st.warning("服务器当前未加载任何模式。")
            st.info("请使用客户端加载模式:\n`python client.py load <模式名>`")
        else:
            mode_options = [mode['mode_name'] for mode in st.session_state.loaded_modes_info]

            def on_mode_change():
                # 当用户切换模式时，清空旧的查询结果
                st.session_state.query_results = None
                st.session_state.selected_chunk_id = None

            st.session_state.selected_mode = st.selectbox(
                "选择一个已加载的模式",
                options=mode_options,
                index=mode_options.index(
                    st.session_state.selected_mode
                ) if st.session_state.selected_mode in mode_options else 0,
                on_change=on_mode_change,
                key="mode_selector"
            )

            query = st.text_area(
                "输入你的问题",
                height=150,
                placeholder=f"向模式 '{st.session_state.selected_mode}' 提问..."
            )

            if st.button("执行查询", type="primary", use_container_width=True):
                if query and st.session_state.selected_mode:
                    st.session_state.selected_chunk_id = None
                    with st.spinner("正在向API发送查询..."):
                        response = requests.post(
                            f"{api_url}/ask",
                            json={"mode_name": st.session_state.selected_mode, "query": query},
                            timeout=60
                        )

                    if response.status_code == 200:
                        st.session_state.query_results = response.json()
                    else:
                        st.error(f"API查询失败: {response.json().get('detail', response.text)}")
                        st.session_state.query_results = None
                    st.rerun()  # 强制刷新页面以显示新结果

        if st.button("清除结果", use_container_width=True):
            st.session_state.query_results = None
            st.session_state.selected_chunk_id = None
            st.rerun()

    # --- 主显示区域 ---
    if st.session_state.selected_mode and st.session_state.query_results:
        results = st.session_state.query_results
        doc_store = get_all_chunks(api_url, st.session_state.selected_mode)

        col1, col2 = st.columns(2)

        with col1:
            sources = results.get('sources', [])
            st.subheader(f"参考来源 ({len(sources)}条)")

            if not sources:
                st.info("未能根据您的问题检索到任何相关内容。")
            else:
                for res in sources:
                    with st.container(border=True):
                        st.markdown(f"**源文件**: `{res.get('doc_id', 'N/A')}`")
                        st.metric(label="相关性", value=f"{res.get('score', 0):.3f}")

                        with st.expander("查看块内容"):
                            st.markdown(res.get('content', ''))

                        if st.button(
                                "查看完整上下文",
                                key=f"btn_ctx_{st.session_state.selected_mode}_{res['chunk_id']}",
                                use_container_width=True
                        ):
                            st.session_state.selected_chunk_id = res['chunk_id']
                            st.rerun()

        with col2:
            st.subheader("最终答案")
            with st.container(border=True):
                st.markdown(results.get('answer', ''))

            # 如果用户点击了查看上下文，则在答案下方显示上下文详情
            if st.session_state.selected_chunk_id is not None and doc_store:
                st.markdown("---")
                display_context_improved(doc_store, st.session_state.selected_chunk_id)
    else:
        st.info("请在左侧选择一个已加载的模式，输入问题并执行查询。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG System Web UI")
    parser.add_argument("--api_url", type=str, required=True, help="Base URL of the RAG API server")

    try:
        # 解析命令行参数以获取API URL
        args = parser.parse_args()
        main(args.api_url)
    except SystemExit as e:
        # 避免在未提供参数时streamlit自身抛出错误
        if e.code != 0:
            raise