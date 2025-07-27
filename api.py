import os
import psutil
import json
import uvicorn
import shutil
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 将项目根目录添加到Python路径中
sys.path.append(str(Path(__file__).resolve().parent))

from rag_system.config import Config, ModeManager
from rag_system.engine import Engine
from rag_system.evaluator import Evaluator
from rag_system.generator import Generator

# 用于在内存中存储已加载的引擎实例
ENGINES: Dict[str, Engine] = {}
PID_FILE_PATH = Path("deploy/app.pid")

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG-API")
logging.getLogger("unstructured").setLevel(logging.ERROR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器。在应用启动时运行yield之前的部分，
    在应用关闭时（如Ctrl+C）运行yield之后的部分。
    """
    # 应用启动时执行的代码
    logger.info("RAG API 服务已启动。")
    yield
    # 应用关闭时执行的代码
    logger.info("RAG API 服务正在关闭...")
    if PID_FILE_PATH.exists():
        logger.info(f"检测到活动的知识树浏览器 (PID文件存在)，将尝试关闭它...")
        try:
            with open(PID_FILE_PATH, 'r') as f:
                pid = int(f.read().strip())

            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                for child in process.children(recursive=True):
                    child.kill()
                process.kill()
                logger.info(f"已成功终止知识树浏览器进程 {pid}。")

            # 清理PID文件
            os.remove(PID_FILE_PATH)
        except psutil.NoSuchProcess:
            logger.warning(f"记录的进程 {pid} 在尝试关闭前已经不存在。")
            os.remove(PID_FILE_PATH)  # 同样清理掉
        except Exception as e:
            logger.error(f"在API关闭期间清理知识树浏览器进程失败: {e}")

# 将 lifespan 管理器添加到 FastAPI 应用中
app = FastAPI(
    title="RAG 系统 API",
    version="3.2.0",  # 版本号可以更新一下
    lifespan=lifespan
)

# 配置CORS中间件，允许所有来源的跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic 模型定义 ---

class ModeRequest(BaseModel):
    """用于模式操作（如加载、卸载、创建）的请求体。"""
    mode_name: str


class AskRequest(BaseModel):
    """用于问答（/ask）的请求体。"""
    mode_name: str
    query: str
    top_k: Optional[int] = None
    use_multi_query: bool = True
    use_jieba_expansion: bool = False


class SearchRequest(BaseModel):
    """用于纯检索（/search）的请求体。"""
    mode_name: str
    query: str
    top_k: Optional[int] = None
    use_multi_query: bool = True
    use_jieba_expansion: bool = False


class TestRequest(BaseModel):
    """用于性能评测（/test）的请求体。"""
    mode_name: str
    query: str
    use_multi_query: bool = True
    use_jieba_expansion: bool = False


# --- 辅助函数 ---

def get_engine(mode_name: str) -> Engine:
    """
    从内存中获取指定模式的引擎实例。

    这是一个内部辅助函数，用于在处理API请求时安全地获取一个已加载的`Engine`对象。

    Args:
        mode_name (str): 需要获取的模式名称。

    Returns:
        Engine: 对应模式的引擎实例。

    Raises:
        HTTPException: 如果模式未加载或尚未准备就绪，则抛出404错误。
    """
    engine = ENGINES.get(mode_name)
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=404, detail=f"模式 '{mode_name}' 未加载或未准备好。请先调用 /modes/load 端点。")
    return engine

# --- API 端点定义 ---

@app.get("/status", summary="检查所有已加载模式的状态")
def get_status():
    """
    获取API服务的当前状态和所有已加载模式的摘要信息。

    Returns:
        dict: 包含服务状态和每个已加载模式的名称及其索引向量数量的字典。
    """
    if not ENGINES:
        return {"status": "idle", "loaded_modes": []}

    loaded_modes_details = []
    for mode_name, engine in ENGINES.items():
        loaded_modes_details.append({
            "mode_name": mode_name,
            "index_vectors": engine.indexer.index.ntotal if engine.indexer.index else 0
        })
    return {"status": "active", "loaded_modes": loaded_modes_details}


@app.post("/modes/load", summary="加载一个新模式到内存")
def load_mode(request: ModeRequest):
    """
    根据配置文件加载一个已存在的模式到服务器内存中，使其可以接受查询。

    Args:
        request (ModeRequest): 包含要加载的`mode_name`。

    Returns:
        dict: 操作成功的消息。

    Raises:
        HTTPException: 如果模式已加载（409）或加载过程中发生错误（500）。
    """
    mode_name = request.mode_name
    if mode_name in ENGINES:
        raise HTTPException(status_code=409, detail=f"模式 '{mode_name}' 已经加载。")

    try:
        logger.info(f"正在加载模式: {mode_name}...")
        mode_manager = ModeManager()
        config = mode_manager.load_mode(mode_name)

        # 仅当提供了有效的LLM配置时，才初始化Generator
        llm_generator = None
        if config.llm_api_key and config.llm_base_url and not config.llm_api_key.startswith("sk-xxxx"):
            logger.info(f"为模式 '{mode_name}' 初始化Generator...")
            model_config = {
                "api_key": config.llm_api_key,
                "base_url": config.llm_base_url,
                "model_name": config.llm_model_name
            }
            llm_generator = Generator(model_config)

        engine = Engine(config, generator=llm_generator)
        engine.load(mode_name)

        ENGINES[mode_name] = engine
        logger.info(f"模式 '{mode_name}' 加载成功。")
        return {"status": "success", "message": f"模式 '{mode_name}' 已成功加载。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模式 '{mode_name}' 失败: {e}")


@app.post("/modes/unload", summary="从内存中卸载指定模式")
def unload_mode(request: ModeRequest):
    """
    从服务器内存中卸载一个已加载的模式，以释放资源。

    Args:
        request (ModeRequest): 包含要卸载的`mode_name`。

    Returns:
        dict: 操作成功的消息。

    Raises:
        HTTPException: 如果模式未被加载（404）。
    """
    mode_name = request.mode_name
    if mode_name not in ENGINES:
        raise HTTPException(status_code=404, detail=f"模式 '{mode_name}' 未被加载。")

    del ENGINES[mode_name]
    logger.info(f"模式 '{mode_name}' 已卸载。")
    return {"status": "success", "message": f"模式 '{mode_name}' 已卸载。"}


@app.post("/modes/create", summary="创建并构建新模式")
def create_mode(request: ModeRequest):
    """
    根据蓝图配置文件（.json）创建一个全新的模式，包括数据处理和索引构建。

    Args:
        request (ModeRequest): 包含要创建的`mode_name`。

    Returns:
        dict: 操作成功的消息。

    Raises:
        HTTPException: 如果模式存储已存在（409）、蓝图文件不存在（404）或构建失败（500）。
    """
    start_time = time.time()
    try:
        mode_manager = ModeManager()
        mode_name = request.mode_name

        storage_path = Path(mode_manager.storage_dir) / mode_name
        if storage_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"模式 '{mode_name}' 的存储目录已存在，无法创建。请先删除或更新。"
            )

        if not mode_manager.mode_exists(mode_name):
            raise HTTPException(status_code=404, detail=f"模式蓝图 'blueprint/{mode_name}.json' 不存在。")

        logger.info(f"开始创建并构建新模式: {mode_name}")
        config = Config.load(f"blueprint/{mode_name}.json")

        # 如果启用了RAPTOR且配置了LLM，则为构建过程初始化一个专用的Generator
        llm_generator_for_build = None
        if (config.proc_enable_raptor and config.llm_api_key and
                config.llm_base_url and not config.llm_api_key.startswith("sk-xxxx")):
            logger.info("检测到RAPTOR已启用，正在为构建过程初始化Generator...")
            try:
                model_config = {
                    "api_key": config.llm_api_key,
                    "base_url": config.llm_base_url,
                    "model_name": config.llm_model_name
                }
                llm_generator_for_build = Generator(model_config)
                logger.info("构建用的Generator初始化成功。")
            except Exception as e:
                logger.warning(f"构建用的Generator初始化失败: {e}。将无法生成RAPTOR摘要层，仅构建基础索引。")

        temp_engine = Engine(config, generator=llm_generator_for_build)
        temp_engine.build(name=mode_name, save=True)

        # 构建完成后，记录并打印一份总结报告
        end_time = time.time()
        total_time = end_time - start_time
        final_stats = temp_engine.indexer.get_stats()
        total_chunks = final_stats.get('total_vectors', 0)

        summary_chunks = 0
        if config.proc_enable_raptor and temp_engine.indexer.doc_store:
            summary_chunks = sum(
                1 for chunk in temp_engine.indexer.doc_store.values()
                if chunk.get('metadata', {}).get('level', 0) > 0
            )

        logger.info("=" * 50)
        logger.info(f"模式 '{mode_name}' 构建流程完成 - 最终报告")
        logger.info(f"  - 总耗时: {total_time:.2f} 秒")
        logger.info(f"  - 总块数 (已存入索引): {total_chunks}")
        if config.proc_enable_raptor:
            logger.info(f"  - 基础块数 (Level 0): {total_chunks - summary_chunks}")
            logger.info(f"  - LLM生成的摘要块数 (Level > 0): {summary_chunks}")
        logger.info("=" * 50)

        return {"status": "success", "message": f"模式 '{mode_name}' 创建并构建成功。"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"创建模式 '{request.mode_name}' 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建失败: {e}")


@app.post("/modes/create-all", summary="一键创建所有可用模式")
def create_all_modes():
    """
    扫描蓝图目录，为所有尚未创建存储的模式执行构建流程。
    该操作会自动跳过那些已经存在存储目录的模式。

    Returns:
        dict: 包含操作结果摘要的字典，如成功、跳过、失败的数量。
    """
    logger.info("收到一键创建所有模式的请求...")
    mode_manager = ModeManager()

    try:
        available_blueprints = mode_manager.list_modes()
        if not available_blueprints:
            logger.info("在 'blueprint/' 目录中未找到任何模式蓝图。")
            return {"status": "noop", "message": "在 'blueprint/' 目录中未找到任何模式蓝图。", "created": 0,
                    "skipped": 0, "failed": 0, "failed_details": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描蓝图目录失败: {e}")

    storage_path = Path("./storage")
    existing_modes = [p.name for p in storage_path.iterdir() if p.is_dir() and (p / "faiss.index").exists()]

    modes_to_create = [bp for bp in available_blueprints if bp not in existing_modes]

    logger.info(
        f"共发现 {len(available_blueprints)} 个蓝图。其中 {len(existing_modes)} 个已存在，{len(modes_to_create)} 个待创建。")

    success_count = 0
    failed_count = 0
    failed_details = []

    for mode_name in modes_to_create:
        try:
            logger.info(f"--- 开始创建模式: {mode_name} ---")
            start_time = time.time()

            config = Config.load(f"blueprint/{mode_name}.json")

            llm_generator_for_build = None
            if (config.proc_enable_raptor and config.llm_api_key and
                    config.llm_base_url and not config.llm_api_key.startswith("sk-xxxx")):
                logger.info(f"为 '{mode_name}' 构建过程初始化Generator...")
                model_config = {"api_key": config.llm_api_key, "base_url": config.llm_base_url,
                                "model_name": config.llm_model_name}
                llm_generator_for_build = Generator(model_config)

            temp_engine = Engine(config, generator=llm_generator_for_build)
            temp_engine.build(name=mode_name, save=True)

            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"--- 模式 '{mode_name}' 创建成功，耗时: {total_time:.2f} 秒 ---")
            success_count += 1

        except Exception as e:
            logger.error(f"--- 创建模式 '{mode_name}' 失败: {e} ---", exc_info=True)
            failed_count += 1
            failed_details.append({"mode_name": mode_name, "error": str(e)})

    message = f"批量创建操作完成。成功: {success_count}, 失败: {failed_count}, 已跳过: {len(existing_modes)}."
    logger.info(message)

    return {
        "status": "completed",
        "message": message,
        "created": success_count,
        "skipped": len(existing_modes),
        "failed": failed_count,
        "failed_details": failed_details
    }

@app.put("/modes/rebuild", summary="智能更新或重建模式")
def rebuild_mode(request: ModeRequest):
    """
    根据最新的蓝图配置文件，智能地更新或完全重建一个模式。
    如果核心参数（如嵌入模型、分块大小）未变，则执行热更新；否则执行完全重建。

    Args:
        request (ModeRequest): 包含要更新的`mode_name`。

    Returns:
        dict: 操作成功的消息，会指明是热更新还是完全重建。

    Raises:
        HTTPException: 如果蓝图文件不存在（404）或更新过程中发生错误（500）。
    """
    mode_manager = ModeManager()
    mode_name = request.mode_name

    if not mode_manager.mode_exists(mode_name):
        raise HTTPException(status_code=404, detail=f"模式蓝图 'blueprint/{mode_name}.json' 不存在。")

    # 如果索引不存在，直接转到创建流程
    storage_path = Path("./storage") / mode_name
    if not storage_path.exists():
        logger.info(f"模式 '{mode_name}' 的索引不存在，将执行首次创建流程。")
        return create_mode(request)

    try:
        new_config = Config.load(f"blueprint/{mode_name}.json")
        old_config = ENGINES[mode_name].config if mode_name in ENGINES else None

        # 定义一组核心参数，这些参数的改变将触发完全重建
        core_params = [
            'embed_model', 'embed_model_source', 'embed_dim',
            'proc_chunk_token_size', 'proc_enable_raptor', 'use_code_space',
            'index_type', 'metric',
            'hnsw_m', 'hnsw_ef_con', 'ivf_nlist'
        ]

        # 检查核心参数是否发生变化
        needs_rebuild = False
        if not old_config:
            needs_rebuild = True
            logger.info(f"模式 '{mode_name}' 未加载，无法进行智能更新对比，将执行完全重构。")
        else:
            for param in core_params:
                if getattr(new_config, param) != getattr(old_config, param):
                    needs_rebuild = True
                    logger.info(
                        f"检测到核心参数 '{param}' 发生改变 "
                        f"('{getattr(old_config, param)}' -> '{getattr(new_config, param)}')."
                    )
                    break

        if needs_rebuild:
            logger.info(f"正在对模式 '{mode_name}' 执行完全重构...")
            start_time = time.time()

            # 卸载旧引擎并删除旧存储
            if mode_name in ENGINES:
                del ENGINES[mode_name]
                logger.info(f"已从内存中卸载旧模式 '{mode_name}'。")
            shutil.rmtree(storage_path)
            logger.info(f"已删除旧的存储目录: {storage_path}")

            # 使用新配置从头构建
            llm_generator = None
            if (new_config.proc_enable_raptor and new_config.llm_api_key and
                    not new_config.llm_api_key.startswith("sk-xxxx")):
                model_config = {"api_key": new_config.llm_api_key, "base_url": new_config.llm_base_url,
                                "model_name": new_config.llm_model_name}
                llm_generator = Generator(model_config)

            temp_engine = Engine(new_config, generator=llm_generator)
            temp_engine.build(name=mode_name, save=True)

            # 打印重构总结报告
            end_time = time.time()
            total_time = end_time - start_time
            final_stats = temp_engine.indexer.get_stats()
            total_chunks = final_stats.get('total_vectors', 0)
            summary_chunks = 0
            if new_config.proc_enable_raptor and temp_engine.indexer.doc_store:
                summary_chunks = sum(
                    1 for chunk in temp_engine.indexer.doc_store.values()
                    if chunk.get('metadata', {}).get('level', 0) > 0
                )

            logger.info("=" * 50)
            logger.info(f"模式 '{mode_name}' 完全重构完成 - 最终报告")
            logger.info(f"  - 总耗时: {total_time:.2f} 秒")
            logger.info(f"  - 总块数 (已存入索引): {total_chunks}")
            if new_config.proc_enable_raptor:
                logger.info(f"  - 基础块数 (Level 0): {total_chunks - summary_chunks}")
                logger.info(f"  - LLM生成的摘要块数 (Level > 0): {summary_chunks}")
            logger.info("=" * 50)
            return {"status": "success", "message": f"模式 '{mode_name}' 已成功完全重建。"}

        else:
            logger.info(f"核心参数未变，正在对模式 '{mode_name}' 执行热更新...")
            engine_to_update = ENGINES[mode_name]

            # 更新引擎的整体配置对象
            engine_to_update.config = new_config

            # 更新可热更新的组件，如检索器权重和Faiss运行时参数
            engine_to_update.retriever.update_weights(new_config.vec_weight, new_config.kw_weight)
            if new_config.index_type == "HNSW":
                engine_to_update.indexer.index.hnsw.efSearch = new_config.hnsw_ef
                logger.info(f"HNSW efSearch已热更新为: {new_config.hnsw_ef}")
            elif new_config.index_type == "IVF":
                engine_to_update.indexer.index.nprobe = new_config.ivf_nprobe
                logger.info(f"IVF nprobe已热更新为: {new_config.ivf_nprobe}")

            # 检查并热更新LLM Generator
            if (new_config.llm_api_key != old_config.llm_api_key or
                    new_config.llm_base_url != old_config.llm_base_url or
                    new_config.llm_model_name != old_config.llm_model_name):
                logger.info("LLM配置发生变化，正在重新初始化Generator...")
                try:
                    if new_config.llm_api_key and not new_config.llm_api_key.startswith("sk-xxxx"):
                        model_config = {"api_key": new_config.llm_api_key, "base_url": new_config.llm_base_url,
                                        "model_name": new_config.llm_model_name}
                        engine_to_update.generator = Generator(model_config)
                        logger.info("Generator已成功热更新。")
                    else:
                        engine_to_update.generator = None
                        logger.info("新的LLM配置无效，Generator已置为None。")
                except Exception as e:
                    logger.warning(f"重新初始化Generator失败: {e}")
                    engine_to_update.generator = None

            return {"status": "success", "message": f"模式 '{mode_name}' 的参数已成功热更新，无需重构索引。"}

    except Exception as e:
        logger.error(f"更新模式 '{request.mode_name}' 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新失败: {e}")


@app.get("/modes/list_available", summary="列出服务器上所有已创建的模式")
def list_available_modes():
    """
    扫描存储目录，列出所有已经成功构建索引的模式。

    Returns:
        dict: 包含一个`modes`键，其值为所有可用模式名称的列表。

    Raises:
        HTTPException: 如果扫描存储目录时发生错误（500）。
    """
    try:
        storage_path = Path("./storage")
        if not storage_path.exists():
            return {"modes": []}
        available_modes = [
            p.name for p in storage_path.iterdir()
            if p.is_dir() and (p / "faiss.index").exists()
        ]
        return {"modes": sorted(available_modes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描可用模式时出错: {e}")


@app.delete("/modes/delete", summary="删除一个模式")
def delete_mode(request: ModeRequest):
    """
    从服务器上永久删除一个模式的磁盘存储数据。

    重要提示：如果模式当前处于加载状态，此操作将失败。请先卸载模式。

    Args:
        request (ModeRequest): 包含要删除的`mode_name`。

    Returns:
        dict: 操作成功的消息。

    Raises:
        HTTPException: 如果模式当前已加载 (409 Conflict)。
    """
    mode_name = request.mode_name

    if mode_name in ENGINES:
        raise HTTPException(
            status_code=409,
            detail=f"模式 '{mode_name}' 当前已加载，无法删除。请先使用 /modes/unload 端点将其卸载。"
        )

    mode_manager = ModeManager()

    if mode_manager.delete_mode(mode_name):
        logger.info(f"模式 '{mode_name}' 的存储数据已成功删除。")
        return {"status": "success", "message": f"模式 '{mode_name}' 的存储数据已成功删除。"}
    else:
        logger.warning(f"尝试删除的模式存储 '{mode_name}' 未找到，无需操作。")
        return {"status": "success", "message": f"模式 '{mode_name}' 的存储数据未找到，无需删除。"}


@app.delete("/modes/clear_all", summary="删除所有模式")
def clear_all_modes():
    """
    清空服务器上的所有模式，包括内存和磁盘存储。这是一个危险操作。

    Returns:
        dict: 包含已清除模式数量的成功消息。
    """
    # 清空内存
    ENGINES.clear()
    logger.info("已从内存中卸载所有模式。")
    # 清空存储
    mode_manager = ModeManager()
    count = mode_manager.clear_all_modes()
    return {"status": "success", "message": f"已成功清除 {count} 个模式的存储数据。"}


@app.post("/ask", summary="向指定模式问答")
def ask_question(request: AskRequest):
    """
    处理一个完整的问答请求。该过程包括检索相关文档块，然后使用LLM根据
    这些块的内容生成最终答案。

    Args:
        request (AskRequest): 包含模式名、问题和查询选项。

    Returns:
        dict: 包含最终答案和参考来源列表的字典。

    Raises:
        HTTPException: 如果问答过程中发生内部错误（500）。
    """
    engine = get_engine(request.mode_name)
    try:
        response = engine.search(
            request.query,
            top_k=request.top_k,
            use_multi_query=request.use_multi_query,
            use_jieba_expansion=request.use_jieba_expansion
        )
        if not response.results:
            return {"answer": "抱歉，在知识库中找不到与您问题相关的信息。", "sources": []}

        # 将检索到的块内容格式化为LLM的上下文
        context_chunks = []
        for res in response.results:
            source_path = res.metadata.get('path', '未知路径')
            chunk_info = (f"来源文档: {res.doc_id}\n"
                          f"文件路径: {source_path}\n"
                          f"内容: {res.content}")
            context_chunks.append(chunk_info)

        if not engine.generator:
            return {
                "answer": "该模式的LLM生成器未初始化，无法生成答案。",
                "sources": [vars(r) for r in response.results]
            }

        answer = engine.generator.answer_question_from_context(request.query, context_chunks)
        return {"answer": answer, "sources": [vars(r) for r in response.results]}
    except Exception as e:
        logger.error(f"问答流程失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="对指定模式进行纯检索")
def search_documents(request: SearchRequest):
    """
    执行一个纯粹的文档检索操作，不经过LLM生成答案，直接返回最相关的文档块。

    Args:
        request (SearchRequest): 包含模式名、查询和检索选项。

    Returns:
        dict: 包含一个`results`键，其值为检索到的文档块列表。
    """
    engine = get_engine(request.mode_name)
    response = engine.search(
        request.query,
        top_k=request.top_k,
        use_multi_query=request.use_multi_query,
        use_jieba_expansion=request.use_jieba_expansion
    )
    return {"results": [vars(r) for r in response.results]}


@app.post("/test", summary="对指定模式进行性能评测")
def test_mode(request: TestRequest):
    """
    对指定模式的检索性能进行评测。

    Args:
        request (TestRequest): 包含模式名、评测查询和查询选项。

    Returns:
        dict: 包含详细评测指标（如召回率、耗时）和检索结果的字典。

    Raises:
        HTTPException: 如果评测过程中发生错误（500）。
    """
    engine = get_engine(request.mode_name)
    try:
        evaluator = Evaluator(engine, engine.config)
        eval_result = evaluator.evaluate(
            request.query,
            use_multi_query=request.use_multi_query,
            use_jieba_expansion=request.use_jieba_expansion
        )
        eval_result_dict = vars(eval_result)
        eval_result_dict['retrieved_results'] = [vars(r) for r in eval_result_dict['retrieved_results']]
        return eval_result_dict
    except Exception as e:
        logger.error(f"评测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/modes/get_all_chunks/{mode_name}", summary="获取模式的所有知识块")
def get_all_chunks(mode_name: str):
    """
    获取指定模式的完整知识库，即所有文档块及其元数据。
    主要用于前端进行上下文追溯和展示。

    Args:
        mode_name (str): 目标模式的名称。

    Returns:
        dict: 包含一个`doc_store`键，其值为知识库字典。

    Raises:
        HTTPException: 如果模式的索引文件不存在（404）或加载失败（500）。
    """
    try:
        mode_manager = ModeManager()
        config = mode_manager.load_mode(mode_name)
        # 临时加载一个引擎以访问其doc_store
        temp_engine = Engine(config)
        temp_engine.load(mode_name)
        doc_store = temp_engine.indexer.doc_store
        return {"doc_store": doc_store}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"模式 '{mode_name}' 的索引文件不存在。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取知识块失败: {e}")


if __name__ == "__main__":
    """
    API服务的应用程序入口点。

    此脚本从 `rag_system/modes/api.json` 配置文件中读取主机和端口设置，
    然后启动Uvicorn服务器来运行FastAPI应用。
    """
    api_config_path = Path("deploy/api.json")
    try:
        with open(api_config_path, 'r', encoding='utf-8') as f:
            api_config = json.load(f)

        host = api_config.get("api_host", "127.0.0.1")
        port = api_config.get("api_port", 8000)

        logger.info(f"即将根据 'api.json' 的配置启动服务在 http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)

    except FileNotFoundError:
        logger.error(f"错误：找不到API配置文件 '{api_config_path}'。无法启动服务。")
    except Exception as e:
        logger.error(f"启动服务失败: {e}")