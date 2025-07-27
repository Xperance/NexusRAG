import argparse
import json
import requests
import subprocess
from typing import Dict, Any
from datetime import datetime
import psutil
import time
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()
CONFIG_FILE_PATH = "deploy/system_config.json"
PID_FILE_PATH = "deploy/app.pid"


def get_system_config() -> Dict:
    """
    读取并返回全局系统配置文件。

    该函数负责从 `system_config.json` 文件中加载配置。如果文件不存在，
    它会创建一个包含默认设置的新配置文件。

    Returns:
        Dict: 包含系统配置的字典。

    Raises:
        None: 错误会打印到控制台，但函数会返回一个默认配置字典。
    """
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 如果配置文件不存在，则创建一个默认配置并返回
        default_config = {"use_multi_query": True, "use_jieba_expansion": False}
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        return default_config
    except Exception as e:
        console.print(f"[bold red]读取系统配置文件 '{CONFIG_FILE_PATH}' 出错: {e}[/bold red]")
        # 在出错时返回一个安全的默认值
        return {"use_multi_query": True, "use_jieba_expansion": False}


def set_system_config(key: str, value: Any):
    """
    设置并保存一个全局系统配置项。

    此函数会更新指定的配置项，并将其持久化到 `system_config.json` 文件中。
    它包含一个特殊逻辑：当 `use_jieba_expansion` 设置为 `True` 时，
    会自动将 `use_multi_query` 设置为 `False`，因为两者功能互斥。

    Args:
        key (str): 要设置的配置项的键。
        value (Any): 要设置的配置项的值。
    """
    config = get_system_config()
    config[key] = value

    # 特殊规则：Jieba分词扩展和LLM多路查询是互斥的
    if key == 'use_jieba_expansion' and value is True:
        if config.get('use_multi_query') is True:
            console.print("[yellow]提示：已自动关闭多路查询模式 (use_multi_query)，以启用分词扩展模式。[/yellow]")
            config['use_multi_query'] = False

    try:
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]✓ 系统配置已更新: '{key}' 设置为 {value}[/green]")
    except Exception as e:
        console.print(f"[bold red]保存系统配置文件 '{CONFIG_FILE_PATH}' 出错: {e}[/bold red]")


def get_api_base_url() -> str:
    """
    从配置文件中读取API服务器的地址和端口，并构建基础URL。

    该函数查找 `deploy/api.json` 文件来获取API服务的连接信息。
    如果文件不存在或读取失败，程序将打印错误信息并退出。

    Returns:
        str: 构建好的API基础URL，格式为 "http://host:port"。
    """
    api_config_path = "deploy/api.json"
    try:
        with open(api_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        host = config.get("api_host", "127.0.0.1")
        port = config.get("api_port", 8000)
        return f"http://{host}:{port}"
    except FileNotFoundError:
        console.print(f"[bold red]错误: 找不到API配置文件 '{api_config_path}'。[/bold red]")
        exit(1)
    except Exception as e:
        console.print(f"[bold red]读取API配置文件时出错: {e}[/bold red]")
        exit(1)


def handle_api_error(response: requests.Response):
    """
    以标准化的格式在控制台显示API错误信息。

    Args:
        response (requests.Response): 来自requests库的响应对象，其中包含错误信息。
    """
    try:
        # 尝试从JSON响应中获取详细错误信息
        detail = response.json().get("detail", response.text)
    except json.JSONDecodeError:
        detail = response.text
    console.print(
        Panel(
            f"[bold red]✗ 操作失败 (状态码: {response.status_code})[/bold red]\n\n{detail}",
            border_style="red",
            title="API 错误"
        )
    )


def display_test_results(result: Dict, mode_name: str, query: str):
    """
    格式化并显示性能评测的结果。

    使用 rich 库的 Panel 和 Table 来美观地展示评测报告，包括核心指标、
    查询扩展详情以及检索到的每个块的详细信息。

    Args:
        result (Dict): 从API `/test` 端点返回的结果字典。
        mode_name (str): 被评测的模式名称。
        query (str): 用于评测的原始查询。
    """
    console.print(
        Panel(
            Text(f"查询: \"{query}\"\n模式: {mode_name}", justify="center"),
            title="[bold blue]API 性能评测报告[/bold blue]",
            border_style="blue"
        )
    )

    # 显示查询扩展的子查询（如果存在）
    sub_queries = result.get('sub_queries_generated', [])
    if sub_queries:
        sub_query_panel_content = ""
        for i, sq in enumerate(sub_queries):
            is_original = "(原始查询)" if sq == query else ""
            sub_query_panel_content += f"{i + 1}. {sq} {is_original}\n"

        console.print(
            Panel(
                sub_query_panel_content.strip(),
                title="[bold cyan]查询扩展详情[/bold cyan]",
                border_style="cyan",
                expand=False
            )
        )

    # 显示核心性能指标
    perf_table = Table(show_header=False, box=None, padding=(0, 2))
    perf_table.add_column()
    perf_table.add_column(style="yellow")
    perf_table.add_row("检索耗时:", f"{result['search_time']:.4f} 秒")
    perf_table.add_row("精确召回率:", f"{result['precision_recall']:.2%} ({result['retrieved']}/{result['ideal']})")
    perf_table.add_row(
        "相关召回率:",
        f"{result['relevance_recall']:.2%} ({result['retrieved_sum']:.3f}/{result['ideal_sum']:.3f})"
    )
    console.print(Panel(perf_table, title="[bold]核心指标[/bold]", border_style="green", expand=False))

    # 显示详细的检索结果列表
    retrieved_results = result.get('retrieved_results', [])
    if not retrieved_results:
        console.print("[yellow]评测未返回详细检索结果。[/yellow]")
        return

    detail_table = Table(
        title="[bold]检索结果详情[/bold]",
        border_style="green",
        show_header=True,
        header_style="bold magenta"
    )
    detail_table.add_column("FAISS ID", style="dim")
    detail_table.add_column("字符串 ID", max_width=25)
    detail_table.add_column("总分", justify="right")
    detail_table.add_column("向量分", justify="right")
    detail_table.add_column("关键词分", justify="right")
    detail_table.add_column("来源文档")
    detail_table.add_column("内容预览", max_width=60)

    for item in retrieved_results:
        metadata = item.get('metadata', {})
        chunk_id_str = metadata.get('chunk_id_str', 'N/A')
        content_preview = item.get('content', '').replace('\n', ' ')[:100] + "..."

        detail_table.add_row(
            str(item.get('chunk_id', 'N/A')),
            chunk_id_str,
            f"{item.get('score', 0):.4f}",
            f"{item.get('vec_score', 0):.4f}",
            f"{item.get('kw_score', 0):.4f}",
            item.get('doc_id', 'N/A'),
            content_preview
        )
    console.print(detail_table)

def is_pid_running(pid: int) -> bool:
    """检查具有给定PID的进程是否仍在运行。"""
    return psutil.pid_exists(pid)

def kill_process_by_pid(pid: int):
    """根据PID安全地终止一个进程及其所有子进程。"""
    try:
        if is_pid_running(pid):
            process = psutil.Process(pid)
            # 终止所有子进程
            for child in process.children(recursive=True):
                child.kill()
            # 终止主进程
            process.kill()
            console.print(f"[yellow]进程 {pid} 已终止。[/yellow]")
    except psutil.NoSuchProcess:
        # 进程已经不存在了，也算成功
        pass
    except Exception as e:
        console.print(f"[bold red]终止进程 {pid} 失败: {e}[/bold red]")

def main():
    """
    RAG 系统 API 客户端的主入口函数。

    该函数负责设置和解析命令行参数，并根据用户输入的命令调用相应的功能，
    如配置管理、模式操作、查询、评测等。
    """
    parser = argparse.ArgumentParser(
        description="RAG 系统 API 客户端 (全功能版)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    system_config = get_system_config()

    # 定义所有可用的子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令', required=True)

    # --- `config` 命令 ---
    p_config = subparsers.add_parser('config', help='管理系统全局配置')
    config_subparsers = p_config.add_subparsers(dest='config_command', help='配置操作', required=True)
    config_subparsers.add_parser('enable-multi-query', help='开启LLM多路查询功能')
    config_subparsers.add_parser('disable-multi-query', help='关闭LLM多路查询功能')
    config_subparsers.add_parser('enable-diff', help='开启Jieba分词查询扩展模式')
    config_subparsers.add_parser('disable-diff', help='关闭Jieba分词查询扩展模式')
    config_subparsers.add_parser('status', help='查看当前配置状态')

    # --- `list` 命令 ---
    p_list = subparsers.add_parser('list', help='列出模式。默认只列出已加载的模式。')
    p_list.add_argument('scope', nargs='?', default='loaded', choices=['loaded', 'all'],
                        help="范围: 'loaded' (默认) 或 'all' (所有已创建的)")

    # --- 模式生命周期管理命令 ---
    p_create = subparsers.add_parser('create', help='请求API服务创建并构建一个新模式')
    p_create.add_argument('mode', help='要创建的模式名称 (蓝图需存在于 blueprint/)')
    p_create_all = subparsers.add_parser('create-all', help='一键创建所有在蓝图文件夹中但尚未创建的模式')
    p_update = subparsers.add_parser('update', help='请求API服务根据最新的蓝图智能更新或重建模式')
    p_update.add_argument('mode', help='要更新的模式名称')
    p_delete = subparsers.add_parser('delete', help='请求API服务删除一个模式')
    p_delete.add_argument('mode', help='要删除的模式名称')
    p_clear = subparsers.add_parser('clear', help='请求API服务删除所有模式')
    p_load = subparsers.add_parser('load', help='命令API服务加载一个模式')
    p_load.add_argument('mode', help='要加载的模式')
    p_load_all = subparsers.add_parser('load-all', help='一键加载所有已创建的模式到服务器')
    p_unload = subparsers.add_parser('unload', help='命令API服务卸载一个已加载的模式')
    p_unload.add_argument('mode', help='要卸载的模式名称')

    # --- 交互与评测命令 ---
    p_test = subparsers.add_parser('test', help='对已加载的模式进行性能评测')
    p_test.add_argument('mode', help='当前API服务加载的模式')
    p_test.add_argument('query', nargs='+', help='用于评测的查询(可以包含空格)')
    p_compare = subparsers.add_parser('compare', help='在客户端对比多个模式的性能')
    p_compare.add_argument('modes', nargs='+', help='至少两个参数: mode1 mode2 ... query')
    p_ask = subparsers.add_parser('ask', help='向已加载的模式提问')
    p_ask.add_argument('mode', help='当前API服务加载的模式')
    p_ask.add_argument('query', nargs='+', help='你的问题(可以包含空格)')
    p_search = subparsers.add_parser('search', help='对已加载的模式执行纯检索')
    p_search.add_argument('mode', help='当前API服务加载的模式')
    p_search.add_argument('query', nargs='+', help='你的查询(可以包含空格)')
    p_search.add_argument('-k', '--top_k', type=int)

    # --- 其他功能命令 ---
    p_show = subparsers.add_parser('show', help='启动一个连接到API的知识树浏览器')
    p_hide = subparsers.add_parser('hide', help='关闭由 show 命令启动的知识树浏览器')
    p_select = subparsers.add_parser('select', help='从API获取数据并生成详细的分块溯源报告')
    p_select.add_argument('mode', help='要生成报告的模式名称')
    p_status = subparsers.add_parser('status', help='检查API服务状态')

    args = parser.parse_args()

    # --- 命令分发逻辑 ---
    if args.command == 'config':
        if args.config_command == 'enable-multi-query':
            set_system_config('use_multi_query', True)
        elif args.config_command == 'disable-multi-query':
            set_system_config('use_multi_query', False)
        elif args.config_command == 'enable-diff':
            set_system_config('use_jieba_expansion', True)
        elif args.config_command == 'disable-diff':
            set_system_config('use_jieba_expansion', False)
        elif args.config_command == 'status':
            current_config = get_system_config()
            mq_status = "开启" if current_config.get("use_multi_query", True) else "关闭"
            diff_status = "开启" if current_config.get("use_jieba_expansion", False) else "关闭"
            console.print(f"当前系统配置:")
            console.print(
                f"- LLM多路查询 (use_multi_query): [bold {'green' if mq_status == '开启' else 'red'}]{mq_status}[/bold {'green' if mq_status == '开启' else 'red'}]")
            console.print(
                f"- Jieba分词扩展 (use_jieba_expansion): [bold {'green' if diff_status == '开启' else 'red'}]{diff_status}[/bold {'green' if diff_status == '开启' else 'red'}]")
        return

    base_url = get_api_base_url()
    response = None

    # --- 参数预处理 ---
    # 将包含空格的查询参数列表合并成一个字符串
    if hasattr(args, 'query') and isinstance(args.query, list):
        args.query = ' '.join(args.query)
    # 为 `compare` 命令分离模式名和查询
    if args.command == 'compare':
        if len(args.modes) < 2:
            console.print("[bold red]错误: 'compare' 命令至少需要提供一个模式名和一个查询。[/bold red]")
            return
        args.query = args.modes[-1]
        args.modes = args.modes[:-1]

    # --- 构建通用的JSON负载 ---
    json_payload = {}
    if args.command in ['ask', 'search', 'test']:
        json_payload = {
            "mode_name": args.mode,
            "query": args.query,
            "use_multi_query": system_config.get("use_multi_query", True),
            "use_jieba_expansion": system_config.get("use_jieba_expansion", False)
        }
        if args.command == 'search' and args.top_k:
            json_payload['top_k'] = args.top_k

    # --- API请求与响应处理 ---
    try:
        if args.command == 'list':
            if args.scope == 'all':
                api_response = requests.get(f"{base_url}/modes/list_available", timeout=10)
                if api_response.status_code == 200:
                    modes = api_response.json().get("modes", [])
                    if not modes:
                        console.print("[yellow]服务器上没有任何已创建的模式。[/yellow]")
                    else:
                        table = Table(title="[bold blue]所有已创建的模式 (来自存储)[/bold blue]", border_style="blue")
                        table.add_column("模式名称", style="green")
                        for mode_name in modes:
                            table.add_row(mode_name)
                        console.print(table)
                else:
                    handle_api_error(api_response)
            else:  # 'loaded'
                api_response = requests.get(f"{base_url}/status", timeout=10)
                if api_response.status_code == 200:
                    data = api_response.json()
                    modes = data.get("loaded_modes", [])
                    if not modes:
                        console.print("[yellow]服务器当前未加载任何模式。[/yellow]")
                    else:
                        table = Table(title="[bold cyan]服务器上已加载的模式[/bold cyan]", border_style="cyan")
                        table.add_column("模式名称", style="green")
                        table.add_column("向量数量")
                        for mode in modes:
                            table.add_row(mode['mode_name'], str(mode['index_vectors']))
                        console.print(table)
                else:
                    handle_api_error(api_response)
            return

        elif args.command == 'create-all':
            console.print("[cyan]开始一键创建所有可用模式...[/cyan]")
            console.print("[yellow]此操作可能会非常耗时，请耐心等待。[/yellow]")
            try:
                # 注意设置一个非常长的超时时间，因为创建多个索引可能需要几分钟甚至更久
                response = requests.post(f"{base_url}/modes/create-all", timeout=1800)  # 30分钟超时
                if response.status_code == 200:
                    data = response.json()
                    console.print(f"[bold green]✓ 批量创建操作完成。[/bold green]")
                    console.print(f"  - 成功创建: [green]{data.get('created', 0)}[/green] 个")
                    console.print(f"  - 自动跳过: [yellow]{data.get('skipped', 0)}[/yellow] 个 (已存在)")
                    console.print(f"  - 创建失败: [red]{data.get('failed', 0)}[/red] 个")

                    failed_details = data.get('failed_details', [])
                    if failed_details:
                        console.print("\n[bold red]失败详情:[/bold red]")
                        for item in failed_details:
                            console.print(f"  - 模式 '{item['mode_name']}': {item['error']}")
                else:
                    handle_api_error(response)
            except requests.exceptions.Timeout:
                console.print(
                    "[bold red]错误: 请求超时。创建所有模式是一个非常耗时的操作，请考虑增加客户端的超时时间或检查服务器状态。[/bold red]")
            except requests.RequestException as e:
                console.print(f"[bold red]无法连接到服务器或发生网络错误: {e}[/bold red]")
            return

        elif args.command == 'load-all':
            console.print("[cyan]开始一键加载所有可用模式...[/cyan]")
            try:
                # 获取所有已创建和已加载的模式，计算出需要加载的模式列表
                list_response = requests.get(f"{base_url}/modes/list_available", timeout=10)
                list_response.raise_for_status()
                available_modes = list_response.json().get("modes", [])

                status_response = requests.get(f"{base_url}/status", timeout=10)
                status_response.raise_for_status()
                loaded_modes = [m['mode_name'] for m in status_response.json().get("loaded_modes", [])]

                modes_to_load = [m for m in available_modes if m not in loaded_modes]

                if not modes_to_load:
                    console.print("[yellow]所有已创建的模式均已加载，无需操作。[/yellow]")
                    return

                console.print(f"发现 {len(modes_to_load)} 个待加载的模式: {', '.join(modes_to_load)}")

                # 循环发送加载请求
                success_count = 0
                fail_count = 0
                for mode_name in modes_to_load:
                    console.print(f"  - 正在加载 [green]{mode_name}[/green]...", end="")
                    try:
                        load_response = requests.post(f"{base_url}/modes/load", json={"mode_name": mode_name}, timeout=120)
                        if load_response.status_code == 200:
                            console.print(" [bold green]✓ 成功[/bold green]")
                            success_count += 1
                        else:
                            console.print(" [bold red]✗ 失败[/bold red]")
                            handle_api_error(load_response)
                            fail_count += 1
                    except requests.RequestException as e:
                        console.print(f" [bold red]✗ 网络错误: {e}[/bold red]")
                        fail_count += 1
                console.print(f"\n[bold]加载完成。成功: {success_count}, 失败: {fail_count}[/bold]")

            except requests.RequestException as e:
                console.print(f"[bold red]无法从服务器获取模式列表: {e}[/bold red]")
            return

        if args.command == 'show':
            if os.path.exists(PID_FILE_PATH):
                with open(PID_FILE_PATH, 'r') as f:
                    pid = int(f.read().strip())
                if is_pid_running(pid):
                    console.print(f"[bold yellow]警告: 知识树浏览器似乎已在运行 (进程ID: {pid})。[/bold yellow]")
                    console.print("[yellow]如果需要重启，请先执行 'python client.py hide'。[/yellow]")
                    return
                else:
                    # 如果进程不存在但PID文件存在，则清理掉无效的PID文件
                    os.remove(PID_FILE_PATH)
            import subprocess
            try:
                with open("deploy/api.json", 'r', encoding='utf-8') as f:
                    api_config = json.load(f)
                app_host = api_config.get("app_host", "127.0.0.1")
                app_port = api_config.get("app_port", 8501)
            except Exception as e:
                console.print(f"[bold red]读取 api.json 中的前端配置失败: {e}[/bold red]")
                app_host, app_port = "127.0.0.1", 8501
            console.print("正在启动知识树浏览器...")
            command = [
                "streamlit", "run", "app.py",
                "--server.address", app_host,
                "--server.port", str(app_port),
                "--", "--api_url", base_url
            ]
            process = subprocess.Popen(command)
            with open(PID_FILE_PATH, 'w') as f:
                f.write(str(process.pid))
            console.print(f"✓ 浏览器界面应在 [cyan]http://{app_host}:{app_port}[/cyan] 打开")
            console.print(f"[dim]进程ID {process.pid} 已记录。使用 'python client.py hide' 来关闭。[/dim]")
            return

        elif args.command == 'hide':
            if not os.path.exists(PID_FILE_PATH):
                console.print("[yellow]找不到正在运行的知识树浏览器实例 (PID文件不存在)。[/yellow]")
                return
            with open(PID_FILE_PATH, 'r') as f:
                pid = int(f.read().strip())
            if not is_pid_running(pid):
                console.print(f"[yellow]记录的进程 {pid} 已不存在。[/yellow]")
            else:
                console.print(f"正在关闭知识树浏览器 (进程ID: {pid})...")
                kill_process_by_pid(pid)
            os.remove(PID_FILE_PATH)
            console.print("[green]✓ 操作完成。[/green]")
            return

        elif args.command == 'ask':
            api_response = requests.post(f"{base_url}/ask", json=json_payload)
            if api_response.status_code == 200:
                data = api_response.json()
                console.print(Panel(data['answer'], title="[bold green]最终答案[/bold green]", border_style="green"))
                if data.get('sources'):
                    table = Table(title="参考来源")
                    table.add_column("ID", style="dim")
                    table.add_column("分数")
                    table.add_column("来源")
                    table.add_column("内容预览", max_width=70)
                    for s in data['sources']:
                        table.add_row(
                            str(s['chunk_id']),
                            f"{s['score']:.4f}",
                            s['doc_id'],
                            s['content'].replace('\n', ' ')[:100] + "..."
                        )
                    console.print(table)
            else:
                handle_api_error(api_response)
            return

        elif args.command == 'search':
            api_response = requests.post(f"{base_url}/search", json=json_payload)
            if api_response.status_code == 200:
                data = api_response.json().get('results', [])
                table = Table(title=f"检索结果: '{args.query}'")
                table.add_column("ID", style="dim")
                table.add_column("分数")
                table.add_column("来源")
                table.add_column("内容预览", max_width=70)
                for r in data:
                    table.add_row(
                        str(r['chunk_id']),
                        f"{r['score']:.4f}",
                        r['doc_id'],
                        r['content'].replace('\n', ' ')[:100] + "..."
                    )
                console.print(table)
            else:
                handle_api_error(api_response)
            return

        elif args.command == 'test':
            api_response = requests.post(f"{base_url}/test", json=json_payload)
            if api_response.status_code == 200:
                display_test_results(api_response.json(), args.mode, args.query)
            else:
                handle_api_error(api_response)
            return

        elif args.command == 'compare':
            query = args.query
            modes_to_compare = args.modes

            console.print(f"--- [bold]开始对已加载的模式进行横向对比[/bold] ---")
            console.print(f"查询: \"{query}\"")
            all_results_data = []

            # 检查待对比的模式是否已加载
            try:
                status_response = requests.get(f"{base_url}/status", timeout=10)
                status_response.raise_for_status()
                loaded_modes_info = status_response.json().get("loaded_modes", [])
                loaded_modes_names = {mode['mode_name'] for mode in loaded_modes_info}
            except requests.RequestException as e:
                console.print(f"[bold red]无法从服务器获取已加载模式列表: {e}[/bold red]")
                return

            # 依次对每个模式进行评测
            for mode_name in modes_to_compare:
                console.print(f"\n[cyan]正在评测模式: {mode_name}[/cyan]")
                if mode_name not in loaded_modes_names:
                    console.print(f"  - [bold yellow]警告: 模式 '{mode_name}' 未在服务器上加载，已跳过。[/bold yellow]")
                    continue
                try:
                    test_response = requests.post(
                        f"{base_url}/test",
                        json={"mode_name": mode_name, "query": query,
                              "use_multi_query": system_config.get("use_multi_query", True),
                              "use_jieba_expansion": system_config.get("use_jieba_expansion", False)},
                        timeout=60
                    )
                    if test_response.status_code == 200:
                        res = test_response.json()
                        res['mode_name'] = mode_name
                        all_results_data.append(res)
                        console.print("  - [green]评测成功![/green]")
                    else:
                        handle_api_error(test_response)
                except requests.RequestException as e:
                    console.print(f"  - [red]评测模式 {mode_name} 时发生网络错误: {e}[/red]")

            if not all_results_data:
                console.print("[yellow]没有可供比较的评测结果。[/yellow]")
                return

            # 汇总对比报告
            summary_table = Table(
                title=f"横向性能对比报告\n查询: \"{query}\"",
                border_style="blue", show_header=True, header_style="bold magenta"
            )
            summary_table.add_column("模式名称")
            summary_table.add_column("耗时(s)")
            summary_table.add_column("精确召回")
            summary_table.add_column("相关召回")

            sorted_summary_data = sorted(all_results_data, key=lambda r: r['precision_recall'], reverse=True)
            for res in sorted_summary_data:
                summary_table.add_row(
                    res['mode_name'], f"{res['search_time']:.4f}",
                    f"{res['precision_recall']:.2%}", f"{res['relevance_recall']:.2%}"
                )
            console.print(summary_table)

            # 展示每个模式的详细检索结果
            console.print("\n--- [bold]各模式检索结果详情[/bold] ---")
            for res_data in sorted_summary_data:
                mode_name = res_data['mode_name']
                retrieved_results = res_data.get('retrieved_results', [])
                detail_table = Table(
                    title=f"模式: [bold green]{mode_name}[/bold green]",
                    border_style="green", show_header=True, header_style="bold magenta"
                )
                detail_table.add_column("FAISS ID", style="dim")
                detail_table.add_column("字符串 ID", max_width=25)
                detail_table.add_column("总分", justify="right")
                detail_table.add_column("向量分", justify="right")
                detail_table.add_column("关键词分", justify="right")
                detail_table.add_column("来源文档")
                detail_table.add_column("内容预览", max_width=60)

                if not retrieved_results:
                    console.print(Panel(f"[yellow]模式 '{mode_name}' 未返回任何检索结果。[/yellow]", border_style="yellow"))
                    continue

                for item in retrieved_results:
                    metadata = item.get('metadata', {})
                    chunk_id_str = metadata.get('chunk_id_str', 'N/A')
                    content_preview = item.get('content', '').replace('\n', ' ')[:100] + "..."
                    detail_table.add_row(
                        str(item.get('chunk_id', 'N/A')), chunk_id_str,
                        f"{item.get('score', 0):.4f}", f"{item.get('vec_score', 0):.4f}",
                        f"{item.get('kw_score', 0):.4f}", item.get('doc_id', 'N/A'), content_preview
                    )
                console.print(detail_table)
            return

        elif args.command == 'select':
            console.print(f"正在从API获取模式 '{args.mode}' 的知识库数据...")
            api_response = requests.get(f"{base_url}/modes/get_all_chunks/{args.mode}")

            if api_response.status_code == 200:
                doc_store = api_response.json().get('doc_store', {})
                if not doc_store:
                    console.print("[yellow]知识库为空或未能获取到数据。[/yellow]")
                    return

                # 生成 Markdown 报告文件
                output_filename = f"select_{args.mode}_details.md"
                markdown_lines = ["# RAG模式 '{args.mode}' 知识库溯源报告",
                                  f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                  f"**总块数:** {len(doc_store)}", "\n---\n"]

                # 按文档ID对块进行分组
                docs_map = {}
                for chunk_id_str_key, chunk_data in doc_store.items():
                    doc_id = chunk_data.get('metadata', {}).get('doc_id', '未知文档')
                    if doc_id not in docs_map:
                        docs_map[doc_id] = []
                    docs_map[doc_id].append(chunk_data)

                # 遍历每个文档并写入其下的所有块
                for doc_id, chunks in sorted(docs_map.items()):
                    markdown_lines.append(f"\n## 文档: {doc_id}\n")
                    sorted_chunks = sorted(chunks, key=lambda c: (
                        c.get('metadata', {}).get('level', 0), c.get('chunk_id', 0)))

                    for chunk in sorted_chunks:
                        chunk_id_int, chunk_id_str = chunk.get('chunk_id', -1), chunk.get('chunk_id_str', 'N/A')
                        metadata, content = chunk.get('metadata', {}), chunk.get('content', '')

                        markdown_lines.append(f"### 块 - 整数ID: `{chunk_id_int}` | 字符串ID: `{chunk_id_str}`")

                        # 根据元数据判断块的来源
                        origin = "原始文档 (文本分块)"  # 默认来源
                        strategy = metadata.get('chunking_strategy')
                        element_type = metadata.get('element_type')

                        if element_type == 'SummaryNode_LLM':
                            origin = "LLM 生成 (RAPTOR 摘要)"
                        elif strategy == 'tabular_summary_codespace':
                            origin = "LLM 生成 (表格摘要)"
                        elif strategy == 'tabular_row_based_semantic':
                            origin = "表格行转换 (程序化语义)"

                        markdown_lines.append(f"- **来源:** {origin}")

                        if metadata.get('child_node_ids'):
                            markdown_lines.append(
                                f"- **参考块:** {', '.join([f'`{ref}`' for ref in metadata['child_node_ids']])}")

                        markdown_lines.append("- **内容:**")
                        markdown_lines.append(f"```\n{content}\n```")
                        markdown_lines.append("- **元数据:**")
                        markdown_lines.append(f"```json\n{json.dumps(metadata, indent=2, ensure_ascii=False)}\n```\n")

                # 将内容写入文件
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(markdown_lines))
                console.print(f"✓ 报告生成成功: [bold cyan]{output_filename}[/bold cyan]")
            else:
                handle_api_error(api_response)
            return

        # --- 简单的 POST/GET/DELETE 请求 ---
        elif args.command == 'load':
            response = requests.post(f"{base_url}/modes/load", json={"mode_name": args.mode}, timeout=120)
        elif args.command == 'unload':
            response = requests.post(f"{base_url}/modes/unload", json={"mode_name": args.mode}, timeout=30)
        elif args.command == 'status':
            response = requests.get(f"{base_url}/status", timeout=10)
        elif args.command == 'create':
            response = requests.post(f"{base_url}/modes/create", json={"mode_name": args.mode}, timeout=600)
        elif args.command == 'update':
            # 为 update 命令添加交互式确认
            if console.input(
                    f"[bold yellow]此操作将根据 'blueprint/{args.mode}.json' 的内容智能更新或完全重建索引。确定要继续吗? (y/N): [/]"
            ).lower() == 'y':
                console.print(f"正在向服务器发送智能更新请求...")
                response = requests.put(f"{base_url}/modes/rebuild", json={"mode_name": args.mode}, timeout=600)
            else:
                console.print("操作已取消。")
                return
        elif args.command == 'delete':
            # 为 delete 命令添加交互式确认
            if console.input(
                    f"[bold red]警告: 这将永久删除模式 '{args.mode}' 的配置文件和索引数据！确定吗? (y/N): [/]"
            ).lower() == 'y':
                response = requests.delete(f"{base_url}/modes/delete", json={"mode_name": args.mode})
            else:
                console.print("操作已取消。")
                return
        elif args.command == 'clear':
            # 为 clear 命令添加交互式确认
            if console.input(
                    f"[bold red]警告: 这将永久删除服务器上所有模式！确定吗? (y/N): [/]"
            ).lower() == 'y':
                response = requests.delete(f"{base_url}/modes/clear_all")
            else:
                console.print("操作已取消。")
                return

    except requests.exceptions.ConnectionError:
        console.print(f"[bold red]无法连接到API服务: {base_url}\n请确保已使用 'python api.py' 启动服务。[/bold red]")
        return
    except Exception as e:
        console.print(f"[bold red]发生未知客户端错误: {e}[/bold red]")
        return

    # --- 统一处理最终响应 ---
    if response is not None:
        if response.status_code < 300:
            console.print(
                Panel(
                    f"[bold green]✓ 操作成功![/bold green]\n\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}",
                    border_style="green"
                )
            )
        else:
            handle_api_error(response)


if __name__ == "__main__":
    main()