# -*- coding: utf-8 -*-
"""
file_flow 包内模块分工（建议阅读顺序）：

**PDF 全文（非 LLM）**
    ``pdf_text_extract`` / ``opendataloader_adapter``：把单个 PDF 转成纯文本；由 ``pdf_prepare`` 调用。

**第一步：装配工作 JSON（``pdf_prepare``）**
    读 PDF 目录 + ``schema`` → 写出 ``*_work.json`` + 同目录 ``*_fulltext.txt``。
    若 ``file_flow_steps`` 中含独立步骤 ``schema_llm_extract``，本步**仅**骨架与全文，大模型摘录在下一步；
    否则在 ``file_flow_llm_extract=true`` 时于本步内联调用 ``schema_llm_extract`` 写各字段 ``content``。

**第二步：schema 大模型摘录（``schema_llm_extract``）**
    读 ``*_work.json`` 与 ``{{案卷名}}_fulltext.txt``，按 field_name + description 写 ``content``（不读 ``standards``）。

**第三步：清单级评审（``standards_llm_review``）**
    按 ``standards.json`` 每条，将整份工作 JSON 嵌入请求，对照 ``standard`` 作答，写入 ``standards_review``。

**第四步：可视化**
    ``render_html`` 等；另有 ``document_export`` 等辅助模块。

**编排与配置**
    ``pipeline_merge``（``run_file_flow``）、``pipeline_config``、``naming``、``step_dotenv``、``llm_openai``。

管线配置仅来自 ``file_flow/pipeline.json`` 已声明键；不与仓库根 ``pipeline.json``、不与 ``defaults_from_environment()`` 合并。
环境加载见 ``file_flow.step_dotenv``。
"""
