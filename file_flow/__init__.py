# -*- coding: utf-8 -*-
"""
纯文件模式流水线（``pdf_prepare`` → ``llm_fill`` → ``standards_llm_review`` → ``render_html``）：schema 抽取按 field_name+description；栏目标注的评审问题来自 ``standards.json`` 与字段顺序对齐的 ``standard``；清单评审见 ``standards_llm_review``。
schema 根结构须含 ``document_types``（与 ``out/schema_example.json`` 一致）。

管线配置与 ``pipeline_config`` 均在本包目录；**编排侧**仅从 ``file_flow/pipeline.json`` 读入已声明键，不与仓库根 ``pipeline.json``、不与 ``defaults_from_environment()`` 做字典合并。
不依赖 ``review_standard_llm_fill``、``vlm_client``、仓库根 ``step_dotenv.py``；环境加载见 ``file_flow.step_dotenv``（先本目录 .env，再上一级 .env 补缺，最后本目录 ``环节变量.env`` 覆盖），
LLM 见 ``file_flow.llm_openai``，编排入口见 ``file_flow.pipeline_merge``（``run_file_flow`` / ``load_merged_pipeline_config``）。
按 ``document_types`` 与字段 ``content`` 的全文抽取见 ``file_flow.schema_llm_extract``；
按 ``standards_example.json`` 清单逐项评审见 ``file_flow.standards_llm_review``。
"""
