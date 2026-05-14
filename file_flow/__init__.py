# -*- coding: utf-8 -*-
"""
file_flow 包内模块分工（建议阅读顺序）：

**PDF 全文（非 LLM）**
    ``pdf_text_extract`` / ``opendataloader_adapter``：把单个 PDF 转成纯文本；由 ``pdf_prepare`` 调用。

**第一步：装配工作 JSON（``pdf_prepare``）**
    读 PDF 目录 + ``schema`` → 写出 ``*_work.json`` + 同目录 ``*_fulltext.txt``。
    若 ``file_flow_llm_extract=true``，在本步内调用 ``schema_llm_extract``：**仅**按 field_name + description
    从全文摘录写入各字段 ``content``（不引入 ``standards``）。

**第二步：字段级对照评审（``llm_fill``）**
    读 ``*_work.json``：用各字段已有 ``content`` + ``standards.json`` **顶层数组按下标**取 ``standard`` 作为
    「评审问题」，调用大模型写入 ``answer``。这是「摘录后的对照说明」，**不是**再次做全文 schema 抽取。

**第三步：清单级评审（``standards_llm_review``）**
    按 ``standards.json`` 每条（category / subcategory / … + standard）结合案卷摘录汇总调用大模型，
    写入 ``standards_review``。

**编排与配置**
    ``pipeline_merge``（``run_file_flow``）、``pipeline_config``、``naming``、``step_dotenv``、``llm_openai``。

**其它**
    ``render_html`` 可视化；``document_export`` 等。

管线配置仅来自 ``file_flow/pipeline.json`` 已声明键；不与仓库根 ``pipeline.json``、不与 ``defaults_from_environment()`` 合并。
环境加载见 ``file_flow.step_dotenv``。
"""
